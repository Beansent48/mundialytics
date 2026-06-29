from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.loaders import load_matches
from mundialytics.evaluation.backtest_runner import BacktestConfig, walk_forward_backtest


def main() -> None:
    p = argparse.ArgumentParser(description="Run leakage-safe walk-forward backtest from canonical matches.")
    p.add_argument("--matches", required=True)
    p.add_argument("--out", default="outputs/backtest_predictions.csv")
    p.add_argument("--summary-out", default="outputs/backtest_summary.json")
    p.add_argument("--min-train-matches", type=int, default=10)
    p.add_argument("--model-type", choices=["poisson", "random_forest_lambda"], default="poisson")
    p.add_argument("--retrain-every", type=int, default=5)
    args = p.parse_args()

    matches_path = ROOT / args.matches if not Path(args.matches).is_absolute() else Path(args.matches)
    matches = load_matches(matches_path)
    pred, summary = walk_forward_backtest(matches, BacktestConfig(args.min_train_matches, args.model_type, args.retrain_every))

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False)
    sout = ROOT / args.summary_out if not Path(args.summary_out).is_absolute() else Path(args.summary_out)
    sout.parent.mkdir(parents=True, exist_ok=True)
    sout.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved backtest predictions to {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
