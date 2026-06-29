from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_sample_prediction_outputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    predictions_path = tmp_path / "match_predictions.csv"
    actuals_path = tmp_path / "actual_results.csv"
    scorelines_path = tmp_path / "scoreline_distribution.csv"
    dynamic_lines_path = tmp_path / "dynamic_market_lines.csv"

    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2026-06-21",
                "competition": "sample",
                "home_team": "alpha",
                "away_team": "beta",
                "p_home_win": 0.62,
                "p_draw": 0.23,
                "p_away_win": 0.15,
                "lambda_home": 1.8,
                "lambda_away": 0.7,
            },
            {
                "match_id": "m2",
                "date": "2026-06-22",
                "competition": "sample",
                "home_team": "gamma",
                "away_team": "delta",
                "p_home_win": 0.28,
                "p_draw": 0.31,
                "p_away_win": 0.41,
                "lambda_home": 1.0,
                "lambda_away": 1.3,
            },
        ]
    ).to_csv(predictions_path, index=False)

    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2026-06-21",
                "competition": "sample",
                "home_team": "alpha",
                "away_team": "beta",
                "home_goals": 2,
                "away_goals": 0,
                "status": "finished",
                "source": "unit_sample",
            },
            {
                "match_id": "m2",
                "date": "2026-06-22",
                "competition": "sample",
                "home_team": "gamma",
                "away_team": "delta",
                "home_goals": 1,
                "away_goals": 1,
                "status": "finished",
                "source": "unit_sample",
            },
        ]
    ).to_csv(actuals_path, index=False)

    pd.DataFrame(
        [
            {"match_id": "m1", "home_goals": 2, "away_goals": 0, "probability": 0.19},
            {"match_id": "m1", "home_goals": 1, "away_goals": 0, "probability": 0.16},
            {"match_id": "m1", "home_goals": 1, "away_goals": 1, "probability": 0.09},
            {"match_id": "m2", "home_goals": 1, "away_goals": 1, "probability": 0.13},
            {"match_id": "m2", "home_goals": 0, "away_goals": 1, "probability": 0.12},
            {"match_id": "m2", "home_goals": 1, "away_goals": 2, "probability": 0.10},
        ]
    ).to_csv(scorelines_path, index=False)

    pd.DataFrame(
        [
            {"match_id": "m1", "market": "goals", "scope": "match", "side": "both", "line": 2.5, "over_under": "under", "model_probability": 0.58},
            {"match_id": "m1", "market": "goals", "scope": "match", "side": "both", "line": 1.5, "over_under": "over", "model_probability": 0.64},
            {"match_id": "m2", "market": "goals", "scope": "match", "side": "both", "line": 2.5, "over_under": "under", "model_probability": 0.61},
            {"match_id": "m2", "market": "goals", "scope": "match", "side": "both", "line": 1.5, "over_under": "over", "model_probability": 0.55},
        ]
    ).to_csv(dynamic_lines_path, index=False)

    return predictions_path, actuals_path, scorelines_path, dynamic_lines_path


def test_v0484_simulation_evaluation_outputs_baselines_and_missing_actuals(tmp_path: Path) -> None:
    predictions_path, actuals_path, scorelines_path, dynamic_lines_path = _write_sample_prediction_outputs(tmp_path)
    evaluation_dir = tmp_path / "evaluation"
    missing_actuals_dir = tmp_path / "evaluation_missing_actuals"

    evaluation_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_simulation_evaluation.py"),
        "--predictions",
        str(predictions_path),
        "--actual-results",
        str(actuals_path),
        "--scorelines",
        str(scorelines_path),
        "--dynamic-lines",
        str(dynamic_lines_path),
        "--out-dir",
        str(evaluation_dir),
        "--evaluation-mode",
        "sample_smoke_evaluation",
        "--clean-out-dir",
    ]
    result = subprocess.run(evaluation_cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr

    expected_files = {
        "simulation_metrics.json",
        "simulation_evaluation.csv",
        "calibration_1x2.csv",
        "goal_error_metrics.csv",
        "scoreline_evaluation.csv",
        "baseline_comparison.csv",
        "line_evaluation.csv",
        "simulation_evaluation_report.html",
    }
    missing = [name for name in expected_files if not (evaluation_dir / name).exists()]
    assert not missing, f"Missing expected evaluation outputs: {missing}"

    metrics = json.loads((evaluation_dir / "simulation_metrics.json").read_text(encoding="utf-8"))
    assert metrics["version"] == "v0.48.4_simulation_evaluation_report"
    assert metrics["status"] == "completed"
    assert metrics["evaluation_mode"] == "sample_smoke_evaluation"
    assert metrics["matches_evaluated"] == 2
    assert metrics["principles"]["offline_only"] is True
    assert metrics["principles"]["model_logic_changed"] is False
    assert metrics["principles"]["betting_recommendations"] is False
    assert metrics["principles"]["player_props_deep_evaluation"] is False
    assert "accuracy_1x2" in metrics["metrics"]
    assert "log_loss_1x2" in metrics["metrics"]
    assert "brier_1x2" in metrics["metrics"]
    assert "minimum_match_results_columns" in metrics["next_data_phase"]

    evaluation = pd.read_csv(evaluation_dir / "simulation_evaluation.csv")
    assert len(evaluation) == 2
    assert {
        "match_id",
        "actual_outcome",
        "predicted_outcome",
        "prediction_correct",
        "actual_scoreline_probability",
        "actual_scoreline_top3",
    }.issubset(evaluation.columns)

    baselines = pd.read_csv(evaluation_dir / "baseline_comparison.csv")
    assert {"accuracy_1x2", "log_loss_1x2", "brier_1x2"}.issubset(set(baselines["metric_name"]))

    calibration = pd.read_csv(evaluation_dir / "calibration_1x2.csv")
    assert {"home_win", "draw", "away_win"}.issubset(set(calibration["outcome_class"]))

    line_eval = pd.read_csv(evaluation_dir / "line_evaluation.csv")
    assert not line_eval.empty
    assert {"line", "over_under", "accuracy_excluding_pushes"}.issubset(line_eval.columns)

    html = (evaluation_dir / "simulation_evaluation_report.html").read_text(encoding="utf-8")
    assert "Mundialytics Simulation Evaluation Report" in html
    assert "Model vs Baselines" in html
    assert "1X2 Calibration Bins" in html
    assert "Goal Error Metrics" in html
    assert "Scoreline Evaluation" in html
    assert "Dynamic Line Evaluation" in html
    assert "Next Data Foundation Requirements" in html

    missing_actuals_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_simulation_evaluation.py"),
        "--predictions",
        str(predictions_path),
        "--out-dir",
        str(missing_actuals_dir),
        "--evaluation-mode",
        "retrospective_backtest",
        "--clean-out-dir",
    ]
    result = subprocess.run(missing_actuals_cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr

    missing_metrics = json.loads((missing_actuals_dir / "simulation_metrics.json").read_text(encoding="utf-8"))
    assert missing_metrics["status"] == "not_available"
    assert missing_metrics["matches_evaluated"] == 0
    assert "actual_results_not_available" in missing_metrics["warnings"]

    missing_evaluation = pd.read_csv(missing_actuals_dir / "simulation_evaluation.csv")
    assert missing_evaluation.empty

    missing_html = (missing_actuals_dir / "simulation_evaluation_report.html").read_text(encoding="utf-8")
    assert "not_available" in missing_html
