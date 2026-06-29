from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.adapters.espn import DEFAULT_WORLD_CUP_LEAGUE, ESPNClient, scoreboard_response_to_df
from mundialytics.data.adapters.sofascore import SofaScoreClient, fetch_scheduled_events_window, scheduled_events_response_to_df
from mundialytics.data.free_fixtures import filter_by_matchday_date, filter_contains, resolved_local_date, world_cup_filters
from mundialytics.matchday.player_inputs import PlayerInputFetchConfig, fetch_player_inputs_for_fixtures
from mundialytics.matchday.today_builder import build_matchday_fixtures, write_matchday_inputs


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _apply_competition_filter(df: pd.DataFrame, competition: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not competition or competition == "all":
        return df, {"competition_filter": "all"}
    if competition in {"world_cup", "fifa_world_cup"}:
        include, exclude = world_cup_filters()
        out, report = filter_contains(df, include=include, exclude=exclude)
        report.update({"competition_filter": competition, "include": include, "exclude": exclude})
        return out, report
    out, report = filter_contains(df, include=[competition.replace("_", " ")], exclude=[])
    report.update({"competition_filter": competition})
    return out, report


def _date_window(center_date: str, days_before: int, days_after: int) -> list[str]:
    from datetime import datetime, timedelta
    center = datetime.fromisoformat(center_date).date()
    return [(center + timedelta(days=o)).isoformat() for o in range(-int(days_before), int(days_after) + 1)]


def _fetch_provider(provider: str, local_date: str, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if provider == "sofascore":
        client = SofaScoreClient()
        payloads, fetched_dates = fetch_scheduled_events_window(
            client,
            center_date=local_date,
            days_before=args.days_before,
            days_after=args.days_after,
        )
        df = scheduled_events_response_to_df(payloads)
        return df, {"provider": "sofascore", "fetched_dates": fetched_dates, "rows_raw": int(len(df))}
    if provider == "espn":
        client = ESPNClient()
        frames: list[pd.DataFrame] = []
        fetched_dates: list[str] = []
        for d in _date_window(local_date, args.days_before, args.days_after):
            dates = d.replace("-", "")
            payload = client.scoreboard(league=args.espn_league, dates=dates, limit=args.espn_limit)
            frames.append(scoreboard_response_to_df(payload, league_slug=args.espn_league))
            fetched_dates.append(dates)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not df.empty:
            key = "match_id" if "match_id" in df.columns else "fixture_id"
            df = df.drop_duplicates(subset=[key]).reset_index(drop=True)
        return df, {"provider": "espn", "dates": fetched_dates, "rows_raw": int(len(df))}
    raise ValueError(f"unknown provider {provider}")


def _load_or_fetch(args: argparse.Namespace, local_date: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.fixtures_source:
        src = _resolve(args.fixtures_source)
        if src is None or not src.exists():
            raise SystemExit(f"fixtures source not found: {args.fixtures_source}")
        df = pd.read_csv(src)
        return df, {"mode": "fixtures_source", "path": str(src), "rows_raw": int(len(df))}

    providers = [args.provider] if args.provider != "auto" else ["sofascore", "espn"]
    attempts: list[dict[str, Any]] = []
    for provider in providers:
        try:
            df, raw = _fetch_provider(provider, local_date, args)
            before_comp = len(df)
            df, comp_report = _apply_competition_filter(df, args.competition)
            df, local_filter_warning = filter_by_matchday_date(df, local_date=local_date, timezone=args.timezone, date_mode=args.date_mode)
            attempt = {
                "provider": provider,
                "status": "ok",
                "rows_raw": int(raw.get("rows_raw", before_comp)),
                "rows_after_competition_filter": int(len(df) + comp_report.get("exclude_removed_rows", 0)),
                "rows_final": int(len(df)),
                "competition_report": comp_report,
                "local_filter_warning": local_filter_warning,
            }
            attempts.append(attempt)
            if not df.empty or args.provider != "auto":
                raw.update({"mode": "fetch", "attempts": attempts, "chosen_provider": provider})
                return df, raw
        except Exception as exc:
            attempts.append({"provider": provider, "status": "error", "error": repr(exc)})
            if args.provider != "auto":
                raise
    return pd.DataFrame(), {"mode": "fetch", "chosen_provider": None, "attempts": attempts}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build run_statistical_matchday-compatible inputs for today's fixtures.")
    date_group = p.add_mutually_exclusive_group()
    date_group.add_argument("--date", default=None, help="Local calendar date, YYYY-MM-DD.")
    date_group.add_argument("--today", action="store_true")
    date_group.add_argument("--tomorrow", action="store_true")
    p.add_argument("--timezone", default="Europe/Madrid", help="IANA timezone used to decide the user-local matchday.")
    p.add_argument("--date-mode", choices=["user", "event", "event_or_user", "utc", "any"], default="event_or_user", help="Calendar date filter: user timezone, event/venue timezone, UTC, or combined. Default includes both user-local and event-local dates.")
    p.add_argument("--competition", default="world_cup", help="world_cup, all, or a text contains filter.")
    p.add_argument("--provider", choices=["auto", "sofascore", "espn"], default="auto")
    p.add_argument("--fixtures-source", default=None, help="Use an already fetched provider fixtures CSV instead of calling a provider.")
    p.add_argument("--days-before", type=int, default=1, help="SofaScore fetch window before selected date.")
    p.add_argument("--days-after", type=int, default=1, help="SofaScore fetch window after selected date.")
    p.add_argument("--espn-league", default=DEFAULT_WORLD_CUP_LEAGUE)
    p.add_argument("--espn-limit", type=int, default=500)
    p.add_argument("--out-dir", default="data/input/generated")
    p.add_argument("--include-live", action="store_true", default=True)
    p.add_argument("--exclude-live", dest="include_live", action="store_false")
    p.add_argument("--include-completed", action="store_true", default=False)
    p.add_argument("--exclude-unknown-status", dest="include_unknown_status", action="store_false", default=True)
    p.add_argument("--fetch-player-inputs", dest="fetch_player_inputs", action="store_true", default=True, help="Try to fetch lineups/squads from the selected provider after fixtures are built.")
    p.add_argument("--no-fetch-player-inputs", dest="fetch_player_inputs", action="store_false")
    p.add_argument("--no-fetch-squads", dest="fetch_squads", action="store_false", default=True)
    p.add_argument("--no-fetch-lineups", dest="fetch_lineups", action="store_false", default=True)
    p.add_argument("--no-empty-player-inputs", dest="write_empty_player_inputs", action="store_false", default=True)
    p.add_argument("--print-run-command", action="store_true", default=True)
    p.add_argument("--no-print-run-command", dest="print_run_command", action="store_false")
    args = p.parse_args(argv)

    if not (args.date or args.today or args.tomorrow):
        args.today = True
    local_date, timezone_warning = resolved_local_date(date=args.date, today=args.today, tomorrow=args.tomorrow, timezone=args.timezone)
    provider_df, source_report = _load_or_fetch(args, local_date)
    source_report["resolved_local_date"] = local_date
    source_report["timezone"] = args.timezone
    source_report["date_mode"] = args.date_mode
    if timezone_warning:
        source_report["timezone_warning"] = timezone_warning

    out_dir = _resolve(args.out_dir)
    assert out_dir is not None
    player_lineups = pd.DataFrame()
    player_squads = pd.DataFrame()
    player_input_report: dict[str, Any] = {"status": "not_requested"}
    if args.fetch_player_inputs and not provider_df.empty:
        # Build a provisional fixtures frame so provider IDs and match IDs are aligned
        # before fetching lineups/squads.
        provisional_fixtures, provisional_report = build_matchday_fixtures(
            provider_df,
            local_date=local_date,
            timezone=args.timezone,
            date_mode=args.date_mode,
            include_live=args.include_live,
            include_completed=args.include_completed,
            include_unknown_status=args.include_unknown_status,
        )
        player_lineups, player_squads, player_input_report = fetch_player_inputs_for_fixtures(
            provider_df,
            provisional_fixtures,
            config=PlayerInputFetchConfig(
                provider="auto" if args.provider == "auto" else args.provider,
                espn_league=args.espn_league,
                fetch_lineups=args.fetch_lineups,
                fetch_squads=args.fetch_squads,
            ),
        )
        player_input_report["provisional_fixture_rows"] = int(len(provisional_fixtures))
        player_input_report["provisional_builder_status"] = provisional_report.get("status")

    result = write_matchday_inputs(
        provider_df,
        out_dir=out_dir,
        local_date=local_date,
        timezone=args.timezone,
        date_mode=args.date_mode,
        include_live=args.include_live,
        include_completed=args.include_completed,
        include_unknown_status=args.include_unknown_status,
        write_empty_player_inputs=args.write_empty_player_inputs,
        source_report=source_report,
        lineups_df=player_lineups,
        squads_df=player_squads,
        player_input_report=player_input_report,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    if args.print_run_command:
        print("\nNext command:")
        lineups = result.get("lineups_csv")
        squads = result.get("squads_csv")
        print("python scripts/run_statistical_matchday.py `")
        print(f"  --fixtures {result['fixtures_csv']} `")
        if lineups:
            print(f"  --lineups {lineups} `")
        if squads:
            print(f"  --squads {squads} `")
        print("  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `")
        print("  --model-config outputs/rolling_model_lab_current/best_rolling_model_config.json `")
        print("  --event-model-config outputs/player_prop_champion_full/prediction_registry.json `")
        print("  --out-dir outputs/statistical_matchday_today `")
        print("  --clean-out-dir `")
        print("  --no-demo-picks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
