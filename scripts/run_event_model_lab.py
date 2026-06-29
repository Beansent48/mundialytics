from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mundialytics.statistical_core.event_model_lab import run_event_model_lab


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run an automatic lab for Mundialytics event/team/player prop models.")
    p.add_argument("--historical-events", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--clean-out-dir", action="store_true")
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--max-test-matches", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    events = pd.read_csv(args.historical_events)
    leaderboard, best = run_event_model_lab(events, args.out_dir, n_trials=args.n_trials, clean_out_dir=args.clean_out_dir, max_test_matches=args.max_test_matches)
    print("Mundialytics event model lab complete")
    print(f"Trials completed: {len(leaderboard)}")
    print(f"Best trial: {best.get('trial_id')} {best.get('trial_name')}")
    print(f"Report: {best.get('report')}")
    if not leaderboard.empty:
        cols = [c for c in ["trial_id", "trial_name", "objective", "team_shots_mae", "team_sot_mae", "team_fouls_mae", "team_yellow_cards_mae", "player_shots_brier", "player_sot_brier", "player_fouls_brier", "player_yellow_card_brier"] if c in leaderboard.columns]
        print(leaderboard[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(code or 0))
