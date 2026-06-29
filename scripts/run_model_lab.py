from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mundialytics.statistical_core.model_lab import run_model_lab  # noqa: E402
from mundialytics.statistical_core.schemas import read_csv_optional  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run automatic Mundialytics v0.26 model lab experiments.")
    p.add_argument("--historical-events", required=True, help="Processed historical event CSV")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--clean-out-dir", action="store_true")
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--min-train-matches", type=int, default=50)
    p.add_argument("--calibration-bins", type=int, default=10)
    p.add_argument("--max-test-matches", type=int, default=None, help="Optional quick-lab cap on holdout rows per trial")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.clean_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events = read_csv_optional(args.historical_events)
    leaderboard, best = run_model_lab(
        events,
        out_dir,
        n_trials=args.n_trials,
        test_fraction=args.test_fraction,
        min_train_matches=args.min_train_matches,
        calibration_bins=args.calibration_bins,
        max_test_matches=args.max_test_matches,
    )
    print("Mundialytics model lab complete")
    print(f"Trials completed: {len(leaderboard)}")
    print(f"Best trial: {best.get('trial_id')} {best.get('trial_name')}")
    print(f"Report: {best.get('report')}")
    if not leaderboard.empty:
        cols = ["trial_id", "trial_name", "objective", "calibrated_log_loss_1x2", "accuracy_pick_max", "calibrated_log_loss_over25", "calibrated_log_loss_btts"]
        print(leaderboard[[c for c in cols if c in leaderboard.columns]].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
