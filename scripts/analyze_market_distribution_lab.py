from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mundialytics.statistical_core.market_distribution_lab import read_line_signals, write_outputs


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze exact-count, range and over/under quality for bookmaker-style stat markets.")
    parser.add_argument("--line-signals", required=True, help="settled_event_line_signals.csv from build_event_line_backtest.py")
    parser.add_argument("--out-dir", default="outputs/market_distribution_lab_current")
    parser.add_argument("--line-min-model-prob", type=float, default=0.52, help="Prefilter lines by model probability for large files. Use 0.0 to analyze every line.")
    parser.add_argument("--line-markets", default=None, help="Optional comma-separated market filter, e.g. corners,team_corners,goalkeeper_saves")
    parser.add_argument("--line-target-quality", default="all", help="Optional target quality filter: real_target,match_total,derived_target,unknown_quality,all")
    parser.add_argument("--line-max-rows", type=int, default=0, help="Optional deterministic cap after filtering for quick runs. 0 means no cap.")
    parser.add_argument("--split-mode", default="auto", choices=["auto", "chronological", "hash", "stratified_hash"], help="Split mode. auto uses dates when coverage is good and deterministic hash when many dates are missing.")
    parser.add_argument("--min-sample", type=int, default=100, help="Minimum validation/test sample for decision matrix rows.")
    parser.add_argument("--write-range-rows", action="store_true", help="Write row-level interval predictions. Can be large.")
    parser.add_argument("--top-print", type=int, default=20)
    args = parser.parse_args(argv)

    line_path = _resolve(args.line_signals)
    out_dir = _resolve(args.out_dir)
    signals = read_line_signals(
        line_path,
        min_model_probability=args.line_min_model_prob,
        markets=args.line_markets,
        target_quality=args.line_target_quality,
        max_rows=args.line_max_rows,
        split_mode=args.split_mode,
    )
    if signals.empty:
        print("No evaluable line signals after filters.")
        return 1
    summary = write_outputs(signals, out_dir, min_sample=args.min_sample, write_range_rows=args.write_range_rows, split_mode=args.split_mode)
    print("MUNDIALYTICS MARKET DISTRIBUTION LAB v0.39.1")
    print(f"Input: {line_path}")
    print(f"Output dir: {out_dir}")
    print(f"Signals analyzed: {len(signals)}")
    print(f"Unique stat predictions: {summary['unique_stat_predictions']}")
    print(f"Split mode: {args.split_mode}")
    print("Target quality:")
    print(pd.Series(summary.get("target_quality", {})).to_string())
    print("Decision counts:")
    print(pd.Series(summary.get("decision_counts", {})).to_string())
    matrix_path = out_dir / "market_side_decision_matrix.csv"
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path)
        if not matrix.empty:
            cols = [
                "signal_group", "target_quality", "fair_odds_bucket", "validation_n", "test_n",
                "validation_hit_rate", "test_hit_rate", "test_avg_model_probability",
                "test_calibration_gap", "test_avg_fair_odds", "decision", "reason_codes",
            ]
            cols = [c for c in cols if c in matrix.columns]
            print("\nTop decision rows:")
            print(matrix[cols].head(args.top_print).to_string(index=False))
    print("\nGenerated files:")
    for name in summary.get("outputs", []):
        print(f"- {out_dir / name}")
    print("\nOJO: esto mide calidad estadística, rangos y calibración. No mide ROI real sin cuotas históricas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
