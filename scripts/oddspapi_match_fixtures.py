#!/usr/bin/env python3
"""Fuzzy-match internal model lines to OddsPapi fixture IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.data.adapters.oddspapi import match_model_lines_to_provider_fixtures


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Match model_market_lines.csv with oddspapi_fixtures.csv.")
    parser.add_argument("--model-lines", required=True)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--out-dir", default="outputs/oddspapi_fixture_mapping_current")
    parser.add_argument("--auto-threshold", type=float, default=0.86)
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = pd.read_csv(_resolve(args.model_lines), low_memory=False)
    fixtures = pd.read_csv(_resolve(args.fixtures), low_memory=False)
    candidates = match_model_lines_to_provider_fixtures(model, fixtures)
    if not candidates.empty:
        candidates["auto_match"] = candidates["best_score"].ge(args.auto_threshold) & candidates["date_diff_days"].le(1)
    selected = candidates[candidates["auto_match"]].drop_duplicates("match_id") if not candidates.empty else pd.DataFrame()
    candidates.to_csv(out_dir / "fixture_mapping_candidates.csv", index=False)
    selected.to_csv(out_dir / "fixture_mapping_selected.csv", index=False)
    summary = {
        "version": "v0.42_oddspapi_match_fixtures",
        "internal_matches": int(model["match_id"].nunique()) if "match_id" in model.columns else 0,
        "provider_fixtures": int(fixtures["provider_fixture_id"].nunique()) if "provider_fixture_id" in fixtures.columns else int(len(fixtures)),
        "candidate_rows": int(len(candidates)),
        "auto_selected_matches": int(len(selected)),
        "warning": "Review fixture_mapping_candidates.csv manually before fetching historical odds at scale.",
    }
    (out_dir / "fixture_mapping_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
