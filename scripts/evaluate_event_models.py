from __future__ import annotations

import argparse
import os
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mundialytics.statistical_core.event_evaluation import EventEvaluationConfig, evaluate_event_models_temporal, write_event_evaluation_outputs


def _load_json(path: str | None) -> dict:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "event_model_config" in payload:
        payload = payload["event_model_config"]
    return payload if isinstance(payload, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Mundialytics team events and player props on a temporal holdout.")
    p.add_argument("--historical-events", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--clean-out-dir", action="store_true")
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--min-train-matches", type=int, default=50)
    p.add_argument("--max-test-matches", type=int, default=None)
    p.add_argument("--event-model-config", default=None, help="JSON config from run_event_model_lab.py or manual event model config")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.clean_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _load_json(args.event_model_config)
    cfg = EventEvaluationConfig(
        test_fraction=args.test_fraction,
        min_train_matches=args.min_train_matches,
        max_test_matches=args.max_test_matches,
        team_model_config=config.get("team_model_config", {}),
        player_model_config=config.get("player_model_config", {}),
    )
    events = pd.read_csv(args.historical_events)
    team_scored, team_lines, player_scored, player_lines, summary = evaluate_event_models_temporal(events, cfg)
    paths = write_event_evaluation_outputs(out_dir, team_scored, team_lines, player_scored, player_lines, summary)
    print("Event model evaluation complete")
    print(f"Status: {summary.get('status')}")
    print(f"Report: {paths.get('event_evaluation_report.html')}")
    team = summary.get("team_event_performance", {}).get("count_metrics_by_market", {})
    player = summary.get("player_prop_performance", {}).get("prop_metrics_by_market", {})
    if team:
        print("\nTeam events:")
        print(pd.DataFrame.from_dict(team, orient="index")[["mae", "baseline_mae", "mae_improvement_vs_baseline", "poisson_nll", "baseline_poisson_nll"]].to_string())
    if player:
        print("\nPlayer props:")
        print(pd.DataFrame.from_dict(player, orient="index")[["brier_1plus", "baseline_brier_1plus", "brier_improvement_vs_baseline", "log_loss_1plus", "baseline_log_loss_1plus"]].to_string())
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(code or 0))
