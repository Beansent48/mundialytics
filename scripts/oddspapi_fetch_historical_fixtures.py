#!/usr/bin/env python3
"""Fetch OddsPapi historical fixtures for planned windows and normalize them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.data.adapters.oddspapi import OddsPapiClient, fixtures_to_frame
from mundialytics.betting.historical_odds_backfill import stable_json_name


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch OddsPapi fixtures by date windows for historical matching.")
    parser.add_argument("--windows", required=True)
    parser.add_argument("--out-dir", default="outputs/oddspapi_historical_fixtures_current")
    parser.add_argument("--provider-config", default=None)
    parser.add_argument("--mode", choices=["direct", "rapidapi"], default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--cache-dir", default="data/raw/oddspapi/cache")
    parser.add_argument("--ledger-path", default="data/raw/oddspapi/request_ledger.jsonl")
    parser.add_argument("--sport-id", type=int, default=10)
    parser.add_argument("--bookmakers", default="bet365", help="Bookmaker slug. Mundialytics v0.46.4 default: bet365 only.")
    parser.add_argument("--status-id", type=int, default=2, help="2=finished per OddsPapi docs")
    parser.add_argument("--has-odds", action="store_true", default=True)
    parser.add_argument("--param-style", choices=["docs_v4", "v5_epoch"], default="docs_v4", help="Use docs_v4 from/to ISO params by default. Use v5_epoch only if your RapidAPI snippet requires startTimeFrom/startTimeTo.")
    parser.add_argument("--max-api-calls", type=int, default=25)
    parser.add_argument("--monthly-budget", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    windows_all = pd.read_csv(_resolve(args.windows), low_memory=False).copy()
    if "span_hours" in windows_all.columns:
        too_wide = pd.to_numeric(windows_all["span_hours"], errors="coerce").ge(240)
        if too_wide.any() and args.param_style == "docs_v4":
            raise ValueError(
                f"{int(too_wide.sum())} fixture windows are >=240h. "
                "OddsPapi sportId+from/to fixture searches must stay under 10 days. "
                "Rebuild the plan with --chunk-hours 216 --pad-hours 6 or lower."
            )
    windows = windows_all.head(args.max_api_calls).copy()
    if args.dry_run:
        preview = windows.copy()
        preview["endpoint"] = "/fixtures"
        preview["sportId"] = args.sport_id
        preview["statusId"] = args.status_id
        preview["hasOdds"] = bool(args.has_odds)
        preview["bookmakers"] = args.bookmakers
        preview.to_csv(out_dir / "fixture_fetch_plan_preview.csv", index=False)
        print(f"Dry run. Planned {len(preview)} fixture calls -> {out_dir / 'fixture_fetch_plan_preview.csv'}")
        return 0

    client = OddsPapiClient.from_env(
        mode=args.mode,
        base_url=args.base_url,
        cache_dir=_resolve(args.cache_dir),
        ledger_path=_resolve(args.ledger_path),
        max_calls=args.max_api_calls,
        monthly_budget=args.monthly_budget,
        provider_config=args.provider_config,
    )
    frames = []
    for _, w in windows.iterrows():
        if args.param_style == "docs_v4":
            params = {
                "sportId": args.sport_id,
                "from": w["from"],
                "to": w["to"],
                "statusId": args.status_id,
                "hasOdds": str(bool(args.has_odds)).lower(),
                "bookmakers": args.bookmakers,
            }
        else:
            params = {
                "sportId": args.sport_id,
                "startTimeFrom": int(w["startTimeFrom"]),
                "startTimeTo": int(w["startTimeTo"]),
                "statusId": args.status_id,
                "hasOdds": str(bool(args.has_odds)).lower(),
                "bookmakers": args.bookmakers,
            }
        payload = client.get("/fixtures", params, force=args.force)
        raw_path = raw_dir / stable_json_name("fixtures", [w.get("window_id"), w.get("from"), w.get("to"), args.bookmakers])
        raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        frame = fixtures_to_frame(payload)
        if not frame.empty:
            frame["source_window_id"] = w.get("window_id")
            frames.append(frame)
    fixtures = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not fixtures.empty:
        fixtures = fixtures.drop_duplicates("provider_fixture_id").sort_values("kickoff_utc")
    fixtures.to_csv(out_dir / "oddspapi_historical_fixtures.csv", index=False)
    summary = {
        "version": "v0.46_oddspapi_historical_fixtures",
        "calls_made": client.calls_made,
        "cache_hits": client.cache_hits,
        "fixture_rows": int(len(fixtures)),
        "unique_fixtures": int(fixtures["provider_fixture_id"].nunique()) if not fixtures.empty and "provider_fixture_id" in fixtures.columns else 0,
        "param_style": args.param_style,
        "bookmakers": args.bookmakers,
        "outputs": ["oddspapi_historical_fixtures.csv", "raw/*.json"],
    }
    (out_dir / "historical_fixtures_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
