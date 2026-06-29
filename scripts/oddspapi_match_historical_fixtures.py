#!/usr/bin/env python3
"""Match internal historical matches to OddsPapi historical fixtures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.betting.historical_odds_backfill import match_internal_to_provider, select_best_fixture_matches


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fuzzy match internal historical matches to OddsPapi fixtures.")
    parser.add_argument("--internal-matches", required=True, help="internal_matches_prepared.csv from fixture plan.")
    parser.add_argument("--provider-fixtures", required=True, help="oddspapi_historical_fixtures.csv")
    parser.add_argument("--out-dir", default="outputs/oddspapi_historical_fixture_mapping_current")
    parser.add_argument("--auto-threshold", type=float, default=0.86)
    parser.add_argument("--max-hours-diff", type=float, default=30.0)
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    internal = pd.read_csv(_resolve(args.internal_matches), low_memory=False)
    provider = pd.read_csv(_resolve(args.provider_fixtures), low_memory=False)
    candidates = match_internal_to_provider(internal, provider, max_hours_diff=args.max_hours_diff, auto_threshold=args.auto_threshold)
    selected = select_best_fixture_matches(candidates, auto_threshold=args.auto_threshold)
    manual = candidates[~candidates.get("match_id", pd.Series(dtype=str)).isin(selected.get("match_id", pd.Series(dtype=str)))].copy() if not candidates.empty else pd.DataFrame()
    candidates.to_csv(out_dir / "fixture_mapping_candidates.csv", index=False)
    selected.to_csv(out_dir / "fixture_mapping_selected.csv", index=False)
    manual.to_csv(out_dir / "fixture_mapping_manual_review.csv", index=False)
    summary = {
        "version": "v0.46_oddspapi_match_historical_fixtures",
        "internal_matches": int(len(internal)),
        "provider_fixtures": int(len(provider)),
        "candidate_rows": int(len(candidates)),
        "selected_matches": int(len(selected)),
        "selected_coverage": round(len(selected) / len(internal), 4) if len(internal) else 0.0,
        "auto_threshold": args.auto_threshold,
        "manual_review_matches": int(internal[~internal["match_id"].isin(selected.get("match_id", []))]["match_id"].nunique()) if "match_id" in internal.columns else 0,
        "outputs": ["fixture_mapping_candidates.csv", "fixture_mapping_selected.csv", "fixture_mapping_manual_review.csv"],
    }
    (out_dir / "fixture_mapping_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
