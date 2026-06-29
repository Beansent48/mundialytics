#!/usr/bin/env python3
"""Fetch and normalize OddsPapi historical odds into Mundialytics historical_odds_input.csv."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.data.adapters.oddspapi import OddsPapiClient, flatten_oddspapi_odds_response


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch OddsPapi historical odds timelines with hard call budget.")
    parser.add_argument("--fixture-mapping", required=True, help="fixture_mapping_selected.csv reviewed/approved")
    parser.add_argument("--markets", default=None, help="soccer_markets.csv from oddspapi_probe.py; strongly recommended")
    parser.add_argument("--out-dir", default="outputs/oddspapi_historical_odds_current")
    parser.add_argument("--cache-dir", default="data/raw/oddspapi/cache")
    parser.add_argument("--ledger-path", default="data/raw/oddspapi/request_ledger.jsonl")
    parser.add_argument("--monthly-budget", type=int, default=250)
    parser.add_argument("--mode", choices=["direct", "rapidapi"], default=None)
    parser.add_argument("--provider-config", default=None, help="External config path. Defaults to MUNDIALYTICS_API_CONFIG or config/mundialytics_api_config.local.yaml")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--bookmaker", default="pinnacle", help="Exactly one bookmaker per historical call when oddsIds is not provided")
    parser.add_argument("--odds-ids-column", default=None, help="Optional column containing oddsIds to restrict response")
    parser.add_argument("--snapshot-policy", choices=["all", "closing", "pre_kickoff"], default="closing")
    parser.add_argument("--pre-kickoff-seconds", type=int, default=3600)
    parser.add_argument("--max-api-calls", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(_resolve(args.fixture_mapping), low_memory=False)
    if "provider_fixture_id" not in mapping.columns:
        raise SystemExit("fixture mapping must include provider_fixture_id")
    mapping = mapping.dropna(subset=["provider_fixture_id"]).drop_duplicates("provider_fixture_id")
    todo = mapping.head(args.max_api_calls).copy()
    if args.dry_run:
        plan = todo[[c for c in ["match_id", "provider_fixture_id", "home_team", "away_team", "provider_kickoff_utc"] if c in todo.columns]].copy()
        plan["endpoint"] = "/v4/historical-odds"
        plan["bookmaker"] = args.bookmaker
        plan.to_csv(out_dir / "historical_odds_fetch_plan_preview.csv", index=False)
        print(f"Dry run. Planned calls written to {out_dir / 'historical_odds_fetch_plan_preview.csv'}")
        return 0

    markets_df = pd.read_csv(_resolve(args.markets), low_memory=False) if args.markets else None
    client = OddsPapiClient.from_env(mode=args.mode, base_url=args.base_url, cache_dir=_resolve(args.cache_dir), ledger_path=_resolve(args.ledger_path), max_calls=args.max_api_calls, monthly_budget=args.monthly_budget, provider_config=args.provider_config)
    frames = []
    for _, row in todo.iterrows():
        fixture_id = str(row["provider_fixture_id"])
        odds_ids = str(row.get(args.odds_ids_column, "")) if args.odds_ids_column else None
        if odds_ids == "nan":
            odds_ids = None
        payload = client.fixture_historical_odds(fixture_id=fixture_id, bookmaker=args.bookmaker, odds_ids=odds_ids)
        (raw_dir / f"historical_{fixture_id.replace('/', '_')}_{args.bookmaker}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        frame = flatten_oddspapi_odds_response(payload, markets_df=markets_df, snapshot_policy=args.snapshot_policy, pre_kickoff_seconds=args.pre_kickoff_seconds)
        if not frame.empty:
            frame["match_id"] = row.get("match_id", "")
            frame["internal_match_id"] = row.get("match_id", "")
            # Keep internal teams if present; provider names remain in notes/raw if needed.
            if "home_team" in row and pd.notna(row["home_team"]):
                frame["home_team"] = row["home_team"]
            if "away_team" in row and pd.notna(row["away_team"]):
                frame["away_team"] = row["away_team"]
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(out_dir / "historical_odds_input.csv", index=False)
    summary = {
        "version": "v0.42_oddspapi_fetch_historical_odds",
        "calls_made": client.calls_made,
        "rows": int(len(out)),
        "priced_matches": int(out["match_id"].nunique()) if not out.empty and "match_id" in out.columns else 0,
        "snapshot_policy": args.snapshot_policy,
        "bookmaker": args.bookmaker,
        "warning": "Inspect unmapped/empty market_key rows before using value_edges as ROI evidence.",
    }
    (out_dir / "historical_odds_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
