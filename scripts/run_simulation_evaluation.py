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

from mundialytics.statistical_core.simulation_evaluation import (  # noqa: E402
    SIMULATION_EVALUATION_VERSION,
    evaluate_simulation_predictions,
    write_simulation_evaluation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Mundialytics simulator predictions against known actual results, offline only."
    )
    parser.add_argument("--predictions", required=True, help="match_predictions.csv generated before or for the evaluated fixtures")
    parser.add_argument("--actual-results", default=None, help="CSV with match_id, home_goals and away_goals. If omitted, outputs are marked not_available.")
    parser.add_argument("--scorelines", default=None, help="Optional scoreline_distribution.csv")
    parser.add_argument("--dynamic-lines", default=None, help="Optional dynamic_market_lines.csv")
    parser.add_argument("--out-dir", required=True, help="Directory where evaluation outputs will be written")
    parser.add_argument(
        "--evaluation-mode",
        default="retrospective_backtest",
        choices=["sample_smoke_evaluation", "retrospective_backtest", "forward_evaluation"],
        help="Evaluation provenance label. Use forward_evaluation only for predictions logged before kickoff.",
    )
    parser.add_argument("--min-matches-for-calibration", type=int, default=30)
    parser.add_argument("--clean-out-dir", action="store_true", help="Delete out-dir before writing outputs")
    return parser


def _read_required_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required input not found: {p}")
    return pd.read_csv(p)


def _read_optional_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None or str(path).strip() == "":
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.clean_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = _read_required_csv(args.predictions)
    actual_results = _read_optional_csv(args.actual_results)
    scorelines = _read_optional_csv(args.scorelines)
    dynamic_lines = _read_optional_csv(args.dynamic_lines)

    outputs = evaluate_simulation_predictions(
        predictions=predictions,
        actual_results=actual_results,
        scoreline_distribution=scorelines,
        dynamic_market_lines=dynamic_lines,
        evaluation_mode=args.evaluation_mode,
        min_matches_for_calibration=args.min_matches_for_calibration,
    )
    written = write_simulation_evaluation_outputs(out_dir=out_dir, outputs=outputs)

    print(f"{SIMULATION_EVALUATION_VERSION}: {outputs.metrics.get('status')}")
    print(f"Matches evaluated: {outputs.metrics.get('matches_evaluated', 0)}")
    print(f"Report: {written.get('simulation_evaluation_report.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
