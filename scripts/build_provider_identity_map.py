from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.provider_identity import (
    IDENTITY_MAP_COLUMNS,
    standardize_provider_players,
    normalize_provider,
)
from mundialytics.data.identity import PlayerIdentityResolver, canonical_player_name, normalize_text
from mundialytics.inference.safe_props import prepare_player_events_for_model
from mundialytics.models.player_event_model import PlayerEventModel


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _player_catalog_from_provider_csv(path: Path, default_provider: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return standardize_provider_players(raw, default_provider=default_provider)


def _historical_catalog_and_model(player_events: pd.DataFrame) -> tuple[pd.DataFrame, PlayerEventModel]:
    hist = prepare_player_events_for_model(player_events)
    model = PlayerEventModel().fit(hist)
    baselines = model.baselines if model.baselines is not None else pd.DataFrame()
    if baselines.empty:
        return baselines, model
    out = baselines.copy()
    out["historical_player_id_global"] = out["player_id_global"]
    out["historical_player_name"] = out["player"]
    out["historical_player_name_norm"] = out["player"].map(normalize_text)
    return out, model


def build_identity_map(provider_players: pd.DataFrame, historical_player_events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    provider = standardize_provider_players(provider_players)
    hist_catalog, model = _historical_catalog_and_model(historical_player_events)
    resolver = model.identity_resolver or PlayerIdentityResolver.from_baselines(pd.DataFrame())
    rows: list[dict] = []
    for _, r in provider.iterrows():
        match = resolver.resolve(r.get("provider_player_name"), None)
        if match.status == "matched":
            hist_hit = hist_catalog[hist_catalog["historical_player_id_global"].astype(str) == str(match.matched_player_id_global)] if not hist_catalog.empty else pd.DataFrame()
            h = hist_hit.iloc[0].to_dict() if not hist_hit.empty else {}
        else:
            h = {}
        rows.append({
            "canonical_player_id": r.get("canonical_player_id"),
            "provider": normalize_provider(r.get("provider")),
            "provider_player_id": r.get("provider_player_id"),
            "provider_player_name": r.get("provider_player_name"),
            "provider_player_name_norm": r.get("provider_player_name_norm"),
            "historical_player_id_global": match.matched_player_id_global if match.status == "matched" else None,
            "historical_player_name": match.matched_player if match.status == "matched" else None,
            "historical_player_name_norm": normalize_text(match.matched_player) if match.matched_player else None,
            "match_method": match.method,
            "match_confidence": match.confidence,
            "match_status": match.status,
            "match_reason": match.reason,
            "team": r.get("team"),
            "team_type": r.get("team_type"),
            "competition": r.get("competition"),
            "position": r.get("position"),
            "first_seen_date": r.get("date"),
            "last_seen_date": r.get("date"),
            "minutes_sample": h.get("minutes_sample"),
            "club_minutes_sample": h.get("club_minutes_sample"),
            "national_minutes_sample": h.get("national_minutes_sample"),
        })
    out = pd.DataFrame(rows)
    for c in IDENTITY_MAP_COLUMNS:
        if c not in out.columns:
            out[c] = None
    out = out[IDENTITY_MAP_COLUMNS]
    summary = {
        "provider_rows": int(len(provider)),
        "identity_rows": int(len(out)),
        "historical_players": int(hist_catalog["historical_player_id_global"].nunique()) if not hist_catalog.empty else 0,
        "match_status_counts": out["match_status"].value_counts(dropna=False).to_dict() if not out.empty else {},
        "match_method_counts": out["match_method"].value_counts(dropna=False).to_dict() if not out.empty else {},
        "unmatched_players": out.loc[out["match_status"].astype(str) != "matched", "provider_player_name"].dropna().head(25).tolist() if not out.empty else [],
    }
    return out, summary


def main() -> None:
    p = argparse.ArgumentParser(description="Build canonical provider ID ↔ historical player identity map.")
    p.add_argument("--provider-players", required=True, help="CSV from API-Football lineups/players with provider_player_id/player columns.")
    p.add_argument("--historical-player-events", required=True, help="Clean historical player events CSV used by player props.")
    p.add_argument("--provider", default="api_football")
    p.add_argument("--out", default="data/identity/player_identity_map.csv")
    p.add_argument("--report", default=None)
    args = p.parse_args()

    provider_path = _resolve(args.provider_players)
    hist_path = _resolve(args.historical_player_events)
    out_path = _resolve(args.out)
    report_path = _resolve(args.report) if args.report else out_path.with_suffix(".report.json")
    assert provider_path and hist_path and out_path and report_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    provider_df = pd.read_csv(provider_path)
    provider_df = standardize_provider_players(provider_df, default_provider=args.provider)
    hist_df = pd.read_csv(hist_path)
    identity_map, summary = build_identity_map(provider_df, hist_df)
    identity_map.to_csv(out_path, index=False)
    report_path.write_text(json.dumps({"status": "PROVIDER_IDENTITY_MAP_BUILT", **summary, "out": str(out_path)}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "PROVIDER_IDENTITY_MAP_BUILT", **summary, "out": str(out_path), "report": str(report_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
