#!/usr/bin/env python3
"""Backfill OddsPapi historical odds raw JSON and normalized tick odds.

This is the disciplined historical downloader:
- one fixtureId + one bookmaker per call unless oddsIds are explicitly supplied
- raw JSON is always stored before normalization
- resume skips existing raw payloads unless --force
- local filtering keeps only Mundialytics target markets for model training/backtesting
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.data.adapters.oddspapi import OddsPapiClient, flatten_oddspapi_odds_response
from mundialytics.betting.historical_odds_backfill import TARGET_MARKET_KEYS, stable_json_name


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def _raw_path(raw_dir: Path, fixture_id: str, bookmaker: str) -> Path:
    return raw_dir / stable_json_name("historical_odds", [fixture_id, bookmaker])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill historical odds from OddsPapi for matched fixtures.")
    parser.add_argument("--fixture-mapping", required=True, help="fixture_mapping_selected.csv")
    parser.add_argument("--markets", required=True, help="soccer_markets.csv from oddspapi_probe.py")
    parser.add_argument("--out-dir", default="outputs/oddspapi_historical_odds_backfill_current")
    parser.add_argument("--provider-config", default=None)
    parser.add_argument("--mode", choices=["direct", "rapidapi"], default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--cache-dir", default="data/raw/oddspapi/cache")
    parser.add_argument("--ledger-path", default="data/raw/oddspapi/request_ledger.jsonl")
    parser.add_argument("--bookmaker", default="bet365", help="Bookmaker slug. Mundialytics v0.46.4 default: bet365 only.")
    parser.add_argument("--max-api-calls", type=int, default=25)
    parser.add_argument("--monthly-budget", type=int, default=None)
    parser.add_argument("--min-interval-sec", type=float, default=5.1, help="Historical endpoint cooldown is 5000ms; keep >=5.0.")
    parser.add_argument("--max-fixtures", type=int, default=None)
    parser.add_argument("--keep-unmapped", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    raw_dir = out_dir / "raw" / args.bookmaker
    raw_dir.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(_resolve(args.fixture_mapping), low_memory=False)
    if "provider_fixture_id" not in mapping.columns:
        raise SystemExit("fixture mapping must include provider_fixture_id")
    mapping = mapping.dropna(subset=["provider_fixture_id"]).drop_duplicates("provider_fixture_id").copy()
    if args.max_fixtures:
        mapping = mapping.head(args.max_fixtures).copy()

    plan = mapping[[c for c in ["match_id", "provider_fixture_id", "home_team", "away_team", "provider_kickoff_utc", "match_confidence"] if c in mapping.columns]].copy()
    plan["bookmaker"] = args.bookmaker
    plan["raw_path"] = plan["provider_fixture_id"].map(lambda x: str(_raw_path(raw_dir, str(x), args.bookmaker)))
    plan["already_downloaded"] = plan["raw_path"].map(lambda p: Path(p).exists())
    plan.to_csv(out_dir / "historical_odds_backfill_plan.csv", index=False)
    if args.dry_run:
        print(f"Dry run. Planned {len(plan)} fixtures -> {out_dir / 'historical_odds_backfill_plan.csv'}")
        return 0

    markets_df = pd.read_csv(_resolve(args.markets), low_memory=False)
    client = OddsPapiClient.from_env(
        mode=args.mode,
        base_url=args.base_url,
        cache_dir=_resolve(args.cache_dir),
        ledger_path=_resolve(args.ledger_path),
        max_calls=args.max_api_calls,
        monthly_budget=args.monthly_budget,
        provider_config=args.provider_config,
    )
    client.min_interval_sec = max(float(client.min_interval_sec or 0.0), float(args.min_interval_sec))

    frames = []
    downloaded = 0
    reused = 0
    for _, row in mapping.iterrows():
        fixture_id = str(row["provider_fixture_id"])
        raw_path = _raw_path(raw_dir, fixture_id, args.bookmaker)
        if raw_path.exists() and not args.force:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            reused += 1
        else:
            if downloaded >= args.max_api_calls:
                break
            payload = client.fixture_historical_odds(fixture_id=fixture_id, bookmaker=args.bookmaker)
            raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            downloaded += 1
        frame = flatten_oddspapi_odds_response(payload, markets_df=markets_df, snapshot_policy="all", include_unmapped=args.keep_unmapped)
        if not frame.empty:
            frame["match_id"] = row.get("match_id", "")
            frame["internal_match_id"] = row.get("match_id", "")
            frame["provider_event_id"] = fixture_id
            frame["kickoff_utc"] = row.get("provider_kickoff_utc", row.get("kickoff_utc", ""))
            frame["match_confidence"] = row.get("match_confidence", "")
            if "home_team" in row and pd.notna(row["home_team"]):
                frame["home_team"] = row["home_team"]
            if "away_team" in row and pd.notna(row["away_team"]):
                frame["away_team"] = row["away_team"]
            if not args.keep_unmapped and "market_key" in frame.columns:
                frame = frame[frame["market_key"].astype(str).isin(TARGET_MARKET_KEYS)].copy()
            frames.append(frame)

    ticks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    ticks.to_csv(out_dir / "historical_odds_ticks.csv", index=False)
    summary = {
        "version": "v0.46.4_oddspapi_historical_odds_backfill_bet365_default",
        "bookmaker": args.bookmaker,
        "fixtures_in_plan": int(len(mapping)),
        "api_calls_made": int(client.calls_made),
        "cache_hits": int(client.cache_hits),
        "raw_downloaded_this_run": int(downloaded),
        "raw_reused_from_disk": int(reused),
        "tick_rows": int(len(ticks)),
        "priced_matches": int(ticks["match_id"].nunique()) if not ticks.empty and "match_id" in ticks.columns else 0,
        "markets": sorted(ticks["market_key"].dropna().unique().tolist()) if not ticks.empty and "market_key" in ticks.columns else [],
        "leakage_policy": "Raw/tick odds may include many timestamps; use oddspapi_build_snapshot_odds.py before training/backtesting.",
        "warning": "Historical endpoint cooldown is 5000ms; this script enforces --min-interval-sec >= 5.1 by default.",
    }
    (out_dir / "historical_odds_backfill_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
