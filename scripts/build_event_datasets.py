from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.events import add_basic_event_metrics, merge_player_events_with_lineups
from mundialytics.data.adapters import (
    statsbomb_events_to_lineups,
    statsbomb_events_to_player_events,
    statsbomb_events_to_tactical_shifts,
    statsbomb_events_to_team_events,
    statsbomb_open_data_match_metadata,
    wyscout_events_to_player_events,
    wyscout_events_to_team_events,
    wyscout_matches_to_lineups,
)
from mundialytics.data.event_quality import diagnose_player_event_dataset
from mundialytics.data.competition_taxonomy import enrich_competition_metadata


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write(df: pd.DataFrame, path: str | None) -> None:
    if not path:
        return
    out = _resolve(path)
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows -> {out}")


def _statsbomb_event_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    events_dir = root / "events" if (root / "events").exists() else root
    return sorted(events_dir.glob("*.json"))


def build_statsbomb(args: argparse.Namespace) -> None:
    root = _resolve(args.input)
    assert root is not None
    files = _statsbomb_event_files(root)
    if not files:
        raise ValueError(f"No StatsBomb event JSON files found in {root}")

    metadata = statsbomb_open_data_match_metadata(root)
    player_frames = []
    team_frames = []
    lineup_frames = []
    tactical_frames = []
    limit = args.limit or len(files)
    for fp in files[:limit]:
        match_id = fp.stem
        meta = metadata.get(str(match_id), {})
        date = meta.get("date")
        competition = args.competition or meta.get("competition") or "StatsBomb Open Data"
        player_frames.append(statsbomb_events_to_player_events(fp, match_id=match_id, date=date, team_scope=args.team_scope, competition=competition))
        team_frames.append(statsbomb_events_to_team_events(fp, match_id=match_id, date=date, team_scope=args.team_scope, competition=competition))
        lineup_frames.append(statsbomb_events_to_lineups(fp, match_id=match_id, date=date, team_scope=args.team_scope, competition=competition))
        tactical_frames.append(statsbomb_events_to_tactical_shifts(fp, match_id=match_id, date=date, team_scope=args.team_scope, competition=competition))

    player_df = pd.concat(player_frames, ignore_index=True) if player_frames else pd.DataFrame()
    lineup_df = pd.concat(lineup_frames, ignore_index=True) if lineup_frames else pd.DataFrame()
    player_df = add_basic_event_metrics(merge_player_events_with_lineups(player_df, lineup_df))
    team_df = pd.concat(team_frames, ignore_index=True) if team_frames else pd.DataFrame()
    tactical_df = pd.concat(tactical_frames, ignore_index=True) if tactical_frames else pd.DataFrame()
    player_df = enrich_competition_metadata(player_df, overwrite=True) if not player_df.empty else player_df
    team_df = enrich_competition_metadata(team_df, overwrite=True) if not team_df.empty else team_df
    lineup_df = enrich_competition_metadata(lineup_df, overwrite=True) if not lineup_df.empty else lineup_df
    tactical_df = enrich_competition_metadata(tactical_df, overwrite=True) if not tactical_df.empty else tactical_df
    _write(player_df, args.player_events_out)
    _write(team_df, args.team_events_out)
    _write(lineup_df, args.lineups_out)
    _write(tactical_df, args.tactical_out)
    if args.diagnostic_out:
        import json
        diag = diagnose_player_event_dataset(player_df, lineups=lineup_df)
        out = _resolve(args.diagnostic_out)
        assert out is not None
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote event diagnostic -> {out}")


def build_wyscout(args: argparse.Namespace) -> None:
    events = _resolve(args.events)
    if events is None:
        raise ValueError("--events is required for Wyscout")
    matches = _resolve(args.matches)
    players = _resolve(args.players)
    teams = _resolve(args.teams)

    pe = wyscout_events_to_player_events(
        events,
        matches_json=matches,
        players_json=players,
        teams_json=teams,
        competition=args.competition,
        season=args.season,
        team_scope=args.team_scope,
    )
    te = wyscout_events_to_team_events(
        events,
        matches_json=matches,
        teams_json=teams,
        competition=args.competition,
        season=args.season,
        team_scope=args.team_scope,
    )
    lu = pd.DataFrame()
    if matches:
        lu = wyscout_matches_to_lineups(
            matches,
            players_json=players,
            teams_json=teams,
            competition=args.competition,
            season=args.season,
            team_scope=args.team_scope,
        )
    pe = add_basic_event_metrics(merge_player_events_with_lineups(pe, lu))
    pe = enrich_competition_metadata(pe, overwrite=True) if not pe.empty else pe
    te = enrich_competition_metadata(te, overwrite=True) if not te.empty else te
    lu = enrich_competition_metadata(lu, overwrite=True) if not lu.empty else lu
    _write(pe, args.player_events_out)
    _write(te, args.team_events_out)
    _write(lu, args.lineups_out)


def main() -> None:
    p = argparse.ArgumentParser(description="Build player/team event datasets from open event-data providers.")
    sub = p.add_subparsers(dest="source", required=True)

    sb = sub.add_parser("statsbomb", help="Convert StatsBomb Open Data event JSON files")
    sb.add_argument("--input", required=True, help="Path to StatsBomb data/events directory, a single event JSON, or data root containing events/")
    sb.add_argument("--competition", default=None, help="Override competition name; if omitted, StatsBomb match metadata is used when available")
    sb.add_argument("--team-scope", default="club", choices=["club", "national"])
    sb.add_argument("--limit", type=int, default=None, help="Optional cap for quick experiments")
    sb.add_argument("--player-events-out", default="data/processed/statsbomb_player_events.csv")
    sb.add_argument("--team-events-out", default="data/processed/statsbomb_team_events.csv")
    sb.add_argument("--lineups-out", default="data/processed/statsbomb_lineups.csv")
    sb.add_argument("--tactical-out", default="data/processed/statsbomb_tactical_shifts.csv")
    sb.add_argument("--diagnostic-out", default="outputs/statsbomb_event_diagnostic.json")
    sb.set_defaults(func=build_statsbomb)

    wy = sub.add_parser("wyscout", help="Convert Wyscout public event dataset JSON files")
    wy.add_argument("--events", required=True, help="Path to events JSON, e.g. events_England.json")
    wy.add_argument("--matches", default=None, help="Optional matches JSON")
    wy.add_argument("--players", default=None, help="Optional players JSON")
    wy.add_argument("--teams", default=None, help="Optional teams JSON")
    wy.add_argument("--competition", default="Wyscout Public Data")
    wy.add_argument("--season", default=None)
    wy.add_argument("--team-scope", default="club", choices=["club", "national"])
    wy.add_argument("--player-events-out", default="data/processed/wyscout_player_events.csv")
    wy.add_argument("--team-events-out", default="data/processed/wyscout_team_events.csv")
    wy.add_argument("--lineups-out", default="data/processed/wyscout_lineups.csv")
    wy.set_defaults(func=build_wyscout)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
