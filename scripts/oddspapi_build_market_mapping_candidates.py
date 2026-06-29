#!/usr/bin/env python3
"""Build/review OddsPapi market mapping candidates for Mundialytics target markets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.data.adapters.oddspapi import build_market_mapping_frame
from mundialytics.betting.historical_odds_backfill import filter_target_market_mapping, TARGET_MARKET_KEYS


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create target market mapping candidates from OddsPapi markets.csv.")
    parser.add_argument("--markets", required=True, help="soccer_markets.csv from oddspapi_probe.py")
    parser.add_argument("--out-dir", default="outputs/oddspapi_market_mapping_current")
    parser.add_argument("--exclude-review-required", action="store_true")
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markets = pd.read_csv(_resolve(args.markets), low_memory=False)
    mapping = build_market_mapping_frame(markets)
    targets = filter_target_market_mapping(mapping, include_review_required=not args.exclude_review_required)
    mapping.to_csv(out_dir / "oddspapi_market_mapping_all.csv", index=False)
    targets.to_csv(out_dir / "oddspapi_target_market_mapping_candidates.csv", index=False)
    summary = {
        "version": "v0.46_oddspapi_market_mapping_candidates",
        "input_markets_rows": int(len(markets)),
        "mapped_rows": int(len(mapping)),
        "target_candidate_rows": int(len(targets)),
        "target_market_keys": sorted(TARGET_MARKET_KEYS),
        "target_market_keys_found": sorted(targets["internal_market_key"].dropna().unique().tolist()) if not targets.empty and "internal_market_key" in targets.columns else [],
        "mapping_confidence_counts": targets["mapping_confidence"].value_counts(dropna=False).to_dict() if not targets.empty and "mapping_confidence" in targets.columns else {},
        "warning": "Review low/medium name-based mappings before using them for ROI evidence, especially team/player stat props.",
    }
    (out_dir / "market_mapping_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
