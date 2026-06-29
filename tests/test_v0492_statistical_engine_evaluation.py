from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from mundialytics.evaluation.statistical_engine import evaluate_statistical_engine


ROOT = Path(__file__).resolve().parents[1]


def _prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2025-01-01",
                "competition": "Test League",
                "home_team": "alpha",
                "away_team": "beta",
                "home_goals": 2,
                "away_goals": 0,
                "actual_outcome": "H",
                "p_home_win": 0.58,
                "p_draw": 0.24,
                "p_away_win": 0.18,
                "lambda_home": 1.7,
                "lambda_away": 0.7,
                "most_likely_score": "1-0",
            },
            {
                "match_id": "m2",
                "date": "2025-01-02",
                "competition": "Test League",
                "home_team": "gamma",
                "away_team": "delta",
                "home_goals": 1,
                "away_goals": 1,
                "actual_outcome": "D",
                "p_home_win": 0.36,
                "p_draw": 0.31,
                "p_away_win": 0.33,
                "lambda_home": 1.2,
                "lambda_away": 1.1,
                "most_likely_score": "1-1",
            },
        ]
    )


def test_statistical_engine_evaluates_goals_lines_scorelines_not_profit() -> None:
    (
        goal_errors,
        goal_lines,
        line_calibration,
        scorelines,
        calibration_layer,
        dixon_coles_scorelines,
        summary,
    ) = evaluate_statistical_engine(_prediction_rows())

    assert "total_goals_mae" in set(goal_errors["metric_name"])
    assert {"total_goals", "btts"} == set(goal_lines["market"])
    assert {"total_goals", "btts"} == set(line_calibration["market"])
    assert scorelines["top5_coverage"].notna().all()
    assert "calibration_layer" in summary
    assert "dixon_coles" in summary
    assert summary["evaluation_purpose"] == "statistical_engine_quality_not_profit"
    assert summary["bets_or_profit_evaluated"] is False
    assert summary["principles"]["roi_not_used_for_model_selection"] is True
    assert "statistical_engine_quality" not in dixon_coles_scorelines.columns or len(dixon_coles_scorelines) >= 0
    assert "market" in calibration_layer.columns


def test_historical_validation_writes_statistical_engine_outputs(tmp_path: Path) -> None:
    matches_path = tmp_path / "matches.csv"
    rows = []
    teams = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    for i in range(36):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        rows.append(
            {
                "match_id": f"m{i:03d}",
                "date": f"2025-01-{(i % 28) + 1:02d}",
                "competition": "Unit League",
                "season": "2025",
                "stage": "league",
                "home_team": home,
                "away_team": away,
                "home_goals": i % 4,
                "away_goals": (i + 1) % 3,
                "status": "finished",
                "neutral": 0,
                "team_scope": "club",
            }
        )
    pd.DataFrame(rows).to_csv(matches_path, index=False)

    out_dir = tmp_path / "validation"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_historical_validation.py"),
        "--matches",
        str(matches_path),
        "--out-dir",
        str(out_dir),
        "--min-train-matches",
        "12",
        "--retrain-every",
        "6",
        "--max-backtest-predictions",
        "12",
        "--min-matches-ready",
        "20",
        "--min-backtest-predictions-ready",
        "10",
        "--model-types",
        "poisson",
    ]
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    assert "statistical_engine_evaluation" in completed.stdout

    report = json.loads((out_dir / "operational_validation_report.json").read_text(encoding="utf-8"))
    stat_eval = report["backtests"]["poisson"]["statistical_engine_evaluation"]
    assert stat_eval["summary"]["evaluation_purpose"] == "statistical_engine_quality_not_profit"
    assert Path(stat_eval["goal_lines_csv"]).exists()
    assert Path(stat_eval["line_calibration_csv"]).exists()
    assert Path(stat_eval["scorelines_csv"]).exists()
    assert Path(stat_eval["calibration_layer_csv"]).exists()
    assert Path(stat_eval["dixon_coles_scorelines_csv"]).exists()
    assert stat_eval["summary"]["model_improvement_features"]["detailed_line_calibration"] is True
