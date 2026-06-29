from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mundialytics.statistical_core.rolling_validation import RollingMatchConfig, run_rolling_model_lab
from mundialytics.statistical_core.schemas import read_csv_optional


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Mundialytics v0.27 rolling-origin match model lab.")
    p.add_argument("--historical-events", required=True)
    p.add_argument("--out-dir", default="outputs/rolling_model_lab_current")
    p.add_argument("--clean-out-dir", action="store_true")
    p.add_argument("--n-trials", type=int, default=14)
    p.add_argument("--min-train-matches", type=int, default=900)
    p.add_argument("--calibration-matches", type=int, default=500)
    p.add_argument("--test-matches", type=int, default=250)
    p.add_argument("--step-matches", type=int, default=250)
    p.add_argument("--max-folds", type=int, default=6)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    if args.clean_out_dir and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    events = read_csv_optional(args.historical_events)
    cfg = RollingMatchConfig(
        min_train_matches=args.min_train_matches,
        calibration_matches=args.calibration_matches,
        test_matches=args.test_matches,
        step_matches=args.step_matches,
        max_folds=args.max_folds,
    )
    leaderboard, best = run_rolling_model_lab(events, out, n_trials=args.n_trials, cfg=cfg)
    print("Mundialytics rolling model lab complete")
    print(f"Trials completed: {len(leaderboard)}")
    print(f"Best trial: {best.get('trial_id')} {best.get('trial_name')}")
    print(f"Report: {best.get('report')}")
    if not leaderboard.empty:
        cols = ["trial_id", "trial_name", "objective", "folds", "test_matches", "calibrated_log_loss_1x2", "accuracy_pick_max", "calibrated_log_loss_over25", "calibrated_log_loss_btts"]
        print(leaderboard[[c for c in cols if c in leaderboard.columns]].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
