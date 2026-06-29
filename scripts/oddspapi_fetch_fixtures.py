#!/usr/bin/env python3
"""Fetch OddsPapi fixtures for pre-planned windows, with cache and call budget."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.data.adapters.oddspapi import OddsPapiClient, fixtures_to_frame


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch OddsPapi fixtures from fixture_search_windows.csv.")
    parser.add_argument("--windows", required=True, help="fixture_search_windows.csv from oddspapi_build_fixture_request_plan.py")
    parser.add_argument("--out-dir", default="outputs/oddspapi_fixtures_current")
    parser.add_argument("--cache-dir", default="data/raw/oddspapi/cache")
    parser.add_argument("--ledger-path", default="data/raw/oddspapi/request_ledger.jsonl")
    parser.add_argument("--monthly-budget", type=int, default=250)
    parser.add_argument("--mode", choices=["direct", "rapidapi"], default=None)
    parser.add_argument("--provider-config", default=None, help="External config path. Defaults to MUNDIALYTICS_API_CONFIG or config/mundialytics_api_config.local.yaml")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--sport-id", type=int, default=10)
    parser.add_argument("--tournament-id", type=int, default=None)
    parser.add_argument("--bookmakers", default=None, help="Optional bookmaker filter, e.g. pinnacle or bet365")
    parser.add_argument("--max-api-calls", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(_resolve(args.windows))
    if args.dry_run:
        preview = windows.head(args.max_api_calls).copy()
        preview["endpoint"] = "/fixtures"
        preview["sportId"] = args.sport_id
        preview["tournamentId"] = args.tournament_id
        preview["bookmakers"] = args.bookmakers
        preview.to_csv(out_dir / "fixture_fetch_plan_preview.csv", index=False)
        print(f"Dry run. Planned calls written to {out_dir / 'fixture_fetch_plan_preview.csv'}")
        return 0

    client = OddsPapiClient.from_env(mode=args.mode, base_url=args.base_url, cache_dir=_resolve(args.cache_dir), ledger_path=_resolve(args.ledger_path), max_calls=args.max_api_calls, monthly_budget=args.monthly_budget, provider_config=args.provider_config)
    frames = []
    for _, row in windows.head(args.max_api_calls).iterrows():
        payload = client.fixtures(
            sport_id=args.sport_id,
            start_time_from=int(row["startTimeFrom"]),
            start_time_to=int(row["startTimeTo"]),
            tournament_id=args.tournament_id,
            bookmakers=args.bookmakers,
        )
        (raw_dir / f"fixtures_window_{int(row['window_id']):04d}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        frame = fixtures_to_frame(payload)
        if not frame.empty:
            frame["window_id"] = int(row["window_id"])
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out = out.drop_duplicates("provider_fixture_id")
    out.to_csv(out_dir / "oddspapi_fixtures.csv", index=False)
    summary = {
        "version": "v0.42_oddspapi_fetch_fixtures",
        "calls_made": client.calls_made,
        "fixture_rows": int(len(out)),
        "unique_fixtures": int(out["provider_fixture_id"].nunique()) if not out.empty else 0,
    }
    (out_dir / "fixture_fetch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
