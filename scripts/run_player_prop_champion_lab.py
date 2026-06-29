from __future__ import annotations

import argparse
import shutil
import sys
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mundialytics.statistical_core.player_prop_champion import ChampionPropConfig, run_player_prop_champion_lab, write_player_prop_champion_outputs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run v0.26 champion/challenger player-prop model lab.")
    p.add_argument("--historical-events", required=True)
    p.add_argument("--out-dir", default="outputs/player_prop_champion_current")
    p.add_argument("--clean-out-dir", action="store_true")
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--calibration-fraction-within-train", type=float, default=0.35)
    p.add_argument("--min-train-matches", type=int, default=50)
    p.add_argument("--max-test-matches", type=int, default=None)
    p.add_argument("--max-calibration-matches", type=int, default=None)
    p.add_argument("--min-calibration-rows", type=int, default=300)
    p.add_argument("--min-group-rows", type=int, default=400)
    p.add_argument("--min-segment-rows", type=int, default=120)
    p.add_argument("--n-trials", type=int, default=None, help="Limit number of candidate prop architectures for quick smoke runs")
    return p


def main() -> int:
    args = build_parser().parse_args()
    out = Path(args.out_dir)
    if args.clean_out_dir and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.historical_events)
    cfg = ChampionPropConfig(
        test_fraction=args.test_fraction,
        calibration_fraction_within_train=args.calibration_fraction_within_train,
        min_train_matches=args.min_train_matches,
        max_test_matches=args.max_test_matches,
        max_calibration_matches=args.max_calibration_matches,
        min_calibration_rows=args.min_calibration_rows,
        min_group_rows=args.min_group_rows,
        min_segment_rows=args.min_segment_rows,
        n_trials=args.n_trials,
    )
    leaderboard, champion_summary, segment_metrics, payload = run_player_prop_champion_lab(events, cfg)
    paths = write_player_prop_champion_outputs(out, leaderboard, champion_summary, segment_metrics, payload)
    print("Mundialytics player prop champion lab complete")
    print(f"Status: {payload.get('status')}")
    print(f"Report: {paths.get('player_prop_champion_report.html')}")
    if not champion_summary.empty:
        cols = [c for c in ["market", "champion_trial_name", "log_loss", "baseline_log_loss", "brier", "baseline_brier", "probability_bias", "policy"] if c in champion_summary.columns]
        print(champion_summary[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
