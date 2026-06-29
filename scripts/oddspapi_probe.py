#!/usr/bin/env python3
"""Small, safe OddsPapi probe: sports, soccer markets and bookmaker capability.

Default budget is 3 calls. This is intentionally tiny so the free plan is not burned.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.adapters.oddspapi import OddsPapiClient, bookmakers_to_frame, build_market_mapping_frame, markets_to_frame, sports_to_frame


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe OddsPapi/RapidAPI with a hard request budget.")
    parser.add_argument("--mode", choices=["direct", "rapidapi"], default=None, help="direct uses ODDSPAPI_API_KEY; rapidapi uses RAPIDAPI_KEY + RAPIDAPI_ODDSPAPI_HOST")
    parser.add_argument("--provider-config", default=None, help="External config path. Defaults to MUNDIALYTICS_API_CONFIG or config/mundialytics_api_config.local.yaml")
    parser.add_argument("--base-url", default=None, help="Override base URL. For RapidAPI copy this from the RapidAPI code snippet.")
    parser.add_argument("--out-dir", default="outputs/oddspapi_probe_current")
    parser.add_argument("--cache-dir", default="data/raw/oddspapi/cache")
    parser.add_argument("--ledger-path", default="data/raw/oddspapi/request_ledger.jsonl")
    parser.add_argument("--monthly-budget", type=int, default=250)
    parser.add_argument("--max-api-calls", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        plan = {
            "mode": args.mode or "direct",
            "planned_calls": ["GET /v4/sports", "GET /v4/bookmakers?playerProps=true", "GET /v4/markets?sportId=10"],
            "max_api_calls": args.max_api_calls,
            "note": "Dry run only: no API calls made.",
        }
        (out_dir / "probe_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(json.dumps(plan, indent=2))
        return 0

    client = OddsPapiClient.from_env(mode=args.mode, base_url=args.base_url, cache_dir=_resolve(args.cache_dir), ledger_path=_resolve(args.ledger_path), max_calls=args.max_api_calls, monthly_budget=args.monthly_budget, provider_config=args.provider_config)
    sports = client.sports()
    bookmakers = client.bookmakers(player_props=True)
    markets = client.markets(sport_id=10)

    (out_dir / "sports_raw.json").write_text(json.dumps(sports, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "bookmakers_player_props_raw.json").write_text(json.dumps(bookmakers, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "soccer_markets_raw.json").write_text(json.dumps(markets, indent=2, ensure_ascii=False), encoding="utf-8")

    sports_to_frame(sports).to_csv(out_dir / "sports.csv", index=False)
    bookmakers_to_frame(bookmakers).to_csv(out_dir / "bookmakers_player_props.csv", index=False)
    markets_df = markets_to_frame(markets)
    markets_df.to_csv(out_dir / "soccer_markets.csv", index=False)
    build_market_mapping_frame(markets_df).to_csv(out_dir / "soccer_market_mapping_suggested.csv", index=False)

    summary = {
        "version": "v0.42_oddspapi_probe",
        "calls_made": client.calls_made,
        "sports_rows": int(len(sports_to_frame(sports))),
        "bookmakers_player_props_rows": int(len(bookmakers_to_frame(bookmakers))),
        "soccer_market_outcome_rows": int(len(markets_df)),
        "outputs": ["sports.csv", "bookmakers_player_props.csv", "soccer_markets.csv", "soccer_market_mapping_suggested.csv"],
    }
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
