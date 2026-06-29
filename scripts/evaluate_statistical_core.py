from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mundialytics.statistical_core.evaluation import (  # noqa: E402
    TemporalEvaluationConfig,
    evaluate_match_model_temporal,
    write_evaluation_artifacts,
)
from mundialytics.statistical_core.schemas import read_csv_optional  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate and calibrate Mundialytics v0.26 statistical core with temporal holdout.")
    p.add_argument("--historical-events", required=True, help="Processed historical event CSV")
    p.add_argument("--out-dir", required=True, help="Output directory for evaluation artifacts")
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--min-train-matches", type=int, default=20)
    p.add_argument("--calibration-bins", type=int, default=10)
    p.add_argument("--clean-out-dir", action="store_true")
    p.add_argument("--model-config", default=None, help="Optional JSON file with MatchOutcomeModel kwargs")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.clean_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events = read_csv_optional(args.historical_events)
    model_config = {}
    if args.model_config:
        model_config = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
        # Accept either raw kwargs or the best_model_config.json wrapper from run_model_lab.py
        if isinstance(model_config, dict) and "model_config" in model_config:
            model_config = model_config.get("model_config") or {}
    cfg = TemporalEvaluationConfig(
        test_fraction=args.test_fraction,
        min_train_matches=args.min_train_matches,
        calibration_bins=args.calibration_bins,
        model_config=model_config,
    )
    predictions, bins, summary, calibration = evaluate_match_model_temporal(events, cfg)
    files = write_evaluation_artifacts(out_dir, predictions, bins, summary, calibration)
    print("Statistical core evaluation complete")
    print(f"Status: {summary.get('status')}")
    print(f"Backtest predictions: {files['match_backtest_predictions']}")
    print(f"Calibration model: {files['match_calibration_model']}")
    if summary.get("metrics"):
        print(pd.Series(summary["metrics"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
