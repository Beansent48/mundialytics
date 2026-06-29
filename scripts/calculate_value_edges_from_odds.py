#!/usr/bin/env python3
"""Attach historical/bookmaker odds to odds-ready model lines and calculate EV/edge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.betting.odds_contract import merge_model_lines_with_odds, standard_odds_input_frame


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calculate value edges from model_market_lines.csv and historical odds input.")
    parser.add_argument("--model-lines", required=True, help="model_market_lines.csv from build_odds_ready_shortlist.py")
    parser.add_argument("--historical-odds", required=True, help="Provider/bookmaker odds mapped to historical_odds_input schema")
    parser.add_argument("--out-dir", default="outputs/value_edges_current")
    parser.add_argument("--min-ev", type=float, default=0.03)
    parser.add_argument("--min-edge", type=float, default=0.02)
    args = parser.parse_args(argv)

    model_path = _resolve(args.model_lines)
    odds_path = _resolve(args.historical_odds)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_lines = pd.read_csv(model_path, low_memory=False)
    raw_odds = pd.read_csv(odds_path, low_memory=False)
    odds = standard_odds_input_frame(raw_odds)
    edges = merge_model_lines_with_odds(model_lines, odds)
    if not edges.empty:
        # Apply user-specified value threshold labels after the generic merge classification.
        ev = pd.to_numeric(edges["ev"], errors="coerce")
        edge = pd.to_numeric(edges["edge"], errors="coerce")
        value_mask = ev.ge(args.min_ev) & edge.ge(args.min_edge)
        high_mask = ev.ge(max(args.min_ev, 0.08)) & edge.ge(max(args.min_edge, 0.04))
        edges.loc[value_mask, "value_label"] = "value"
        edges.loc[high_mask, "value_label"] = "high_value"
        edges.loc[edges["bookmaker_odds"].isna(), "value_label"] = "no_odds"
    edges.to_csv(out_dir / "value_edges.csv", index=False)
    summary = {
        "version": "v0.40_value_edges",
        "model_lines_rows": int(len(model_lines)),
        "odds_rows": int(len(odds)),
        "value_edges_rows": int(len(edges)),
        "priced_rows": int(edges["bookmaker_odds"].notna().sum()) if "bookmaker_odds" in edges.columns else 0,
        "value_label_counts": edges["value_label"].value_counts(dropna=False).to_dict() if "value_label" in edges.columns else {},
        "mode": "historical_odds_value_eval" if "actual_win" in edges.columns else "odds_comparison_only",
        "warning": "ROI/profit only has meaning if model lines contain settled actual_win and odds snapshots are realistic historical prices.",
    }
    if "actual_win" in edges.columns and edges["bookmaker_odds"].notna().any():
        priced = edges[edges["bookmaker_odds"].notna()].copy()
        for label in ["value", "high_value"]:
            sel = priced[priced["value_label"].eq(label)]
            if not sel.empty and "profit_1u" in sel.columns:
                summary[f"{label}_n"] = int(len(sel))
                summary[f"{label}_hit_rate"] = float(pd.to_numeric(sel["actual_win"], errors="coerce").mean())
                summary[f"{label}_profit_1u"] = float(pd.to_numeric(sel["profit_1u"], errors="coerce").sum())
                summary[f"{label}_roi_1u"] = float(pd.to_numeric(sel["profit_1u"], errors="coerce").mean())
    (out_dir / "value_edges_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("MUNDIALYTICS VALUE EDGES v0.40")
    print(f"Model lines: {len(model_lines)}")
    print(f"Odds rows: {len(odds)}")
    print(f"Priced rows: {summary['priced_rows']}")
    print(f"Output: {out_dir / 'value_edges.csv'}")
    print("Value labels:")
    print(json.dumps(summary.get("value_label_counts", {}), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
