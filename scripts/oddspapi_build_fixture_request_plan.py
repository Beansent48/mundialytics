#!/usr/bin/env python3
"""Build an OddsPapi fixture-discovery request plan from model_market_lines.csv."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.data.adapters.oddspapi import build_fixture_windows_from_model_lines


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan low-call OddsPapi fixture discovery windows.")
    parser.add_argument("--model-lines", required=True, help="outputs/odds_ready_current/model_market_lines.csv")
    parser.add_argument("--out-dir", default="outputs/oddspapi_request_plan_current")
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--pad-hours", type=int, default=12)
    parser.add_argument("--max-planned-calls", type=int, default=25)
    parser.add_argument("--allow-over-budget", action="store_true")
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_lines = pd.read_csv(_resolve(args.model_lines), low_memory=False)
    windows = build_fixture_windows_from_model_lines(model_lines, chunk_days=args.chunk_days, pad_hours=args.pad_hours)
    estimated_calls = int(len(windows))
    if estimated_calls > args.max_planned_calls and not args.allow_over_budget:
        windows.head(args.max_planned_calls).to_csv(out_dir / "fixture_search_windows_preview.csv", index=False)
        raise SystemExit(f"Planned calls {estimated_calls} exceed --max-planned-calls {args.max_planned_calls}. Increase chunk-days or pass --allow-over-budget intentionally.")
    windows.to_csv(out_dir / "fixture_search_windows.csv", index=False)
    unique_matches = int(model_lines["match_id"].nunique()) if "match_id" in model_lines.columns else int(len(model_lines))
    summary = {
        "version": "v0.42_oddspapi_fixture_request_plan",
        "model_line_rows": int(len(model_lines)),
        "unique_matches": unique_matches,
        "chunk_days": args.chunk_days,
        "estimated_fixture_discovery_calls": estimated_calls,
        "max_planned_calls": args.max_planned_calls,
        "recommendation": "Use weekly or monthly chunks, then fuzzy-match locally. Do not call /fixtures once per internal match.",
    }
    (out_dir / "fixture_request_plan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
