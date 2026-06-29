
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_audit_inputs(tmp_path: Path) -> dict[str, Path]:
    fixtures = tmp_path / "fixtures.csv"
    actuals = tmp_path / "actual_results.csv"
    lineups = tmp_path / "lineups.csv"
    squads = tmp_path / "squads.csv"
    player_events = tmp_path / "player_events.csv"
    predictions = tmp_path / "match_predictions.csv"
    dynamic_lines = tmp_path / "dynamic_market_lines.csv"

    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2026-06-21",
                "competition": "sample",
                "home_team": "Alpha",
                "away_team": "Beta",
                "team_scope": "national",
            },
            {
                "match_id": "m2",
                "date": "2026-06-22",
                "competition": "sample",
                "home_team": "Gamma",
                "away_team": "Delta",
                "team_scope": "national",
            },
        ]
    ).to_csv(fixtures, index=False)

    pd.DataFrame(
        [
            {"match_id": "m1", "date": "2026-06-21", "home_team": "Alpha", "away_team": "Beta", "home_goals": 2, "away_goals": 0, "status": "finished"},
        ]
    ).to_csv(actuals, index=False)

    pd.DataFrame(
        [
            {"match_id": "m1", "team": "Alpha", "player": "Player One", "position": "ST", "started": 1, "expected_minutes": 80},
            {"match_id": "m1", "team": "Beta", "player": "Player Two", "position": "GK", "started": 1, "expected_minutes": 90},
            {"match_id": "m2", "team": "Gamma", "player": "Historical Player", "position": "ST", "started": 1, "expected_minutes": 70},
        ]
    ).to_csv(lineups, index=False)

    pd.DataFrame(
        [
            {"team": "Alpha", "player": "Player One", "position": "ST", "status": "current", "expected_minutes": 80},
            {"team": "Beta", "player": "Player Two", "position": "GK", "status": "current", "expected_minutes": 90},
            {"team": "Gamma", "player": "Old Striker", "position": "ST", "status": "inactive", "expected_minutes": 0},
        ]
    ).to_csv(squads, index=False)

    pd.DataFrame(
        [
            {"match_id": "old1", "date": "2025-01-01", "team": "Alpha", "opponent": "Beta", "player": "Player One", "minutes": 82, "goals": 1, "shots": 3, "shots_on_target": 2},
            {"match_id": "old2", "date": "2025-01-02", "team": "Gamma", "opponent": "Delta", "player": "Old Striker", "minutes": 60, "goals": 0, "shots": 1, "shots_on_target": 0},
        ]
    ).to_csv(player_events, index=False)

    pd.DataFrame(
        [
            {"match_id": "m1", "date": "2026-06-21", "home_team": "Alpha", "away_team": "Beta", "p_home_win": 0.60, "p_draw": 0.25, "p_away_win": 0.15},
            {"match_id": "m2", "date": "2026-06-22", "home_team": "Gamma", "away_team": "Delta", "p_home_win": 0.40, "p_draw": 0.30, "p_away_win": 0.30},
        ]
    ).to_csv(predictions, index=False)

    pd.DataFrame(
        [
            {"match_id": "m1", "market": "goals", "scope": "match", "line": 2.5, "over_under": "under", "model_probability": 0.58},
            {"match_id": "m2", "market": "goals", "scope": "match", "line": 2.5, "over_under": "over", "model_probability": 0.52},
        ]
    ).to_csv(dynamic_lines, index=False)

    return {
        "fixtures": fixtures,
        "actuals": actuals,
        "lineups": lineups,
        "squads": squads,
        "player_events": player_events,
        "predictions": predictions,
        "dynamic_lines": dynamic_lines,
    }


def test_v0490_data_audit_outputs_reports_and_guardrails(tmp_path: Path) -> None:
    paths = _write_audit_inputs(tmp_path)
    out_dir = tmp_path / "data_audit"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_data_audit.py"),
        "--fixtures",
        str(paths["fixtures"]),
        "--actual-results",
        str(paths["actuals"]),
        "--lineups",
        str(paths["lineups"]),
        "--squads",
        str(paths["squads"]),
        "--player-events",
        str(paths["player_events"]),
        "--predictions",
        str(paths["predictions"]),
        "--dynamic-lines",
        str(paths["dynamic_lines"]),
        "--out-dir",
        str(out_dir),
        "--run-label",
        "unit_data_audit",
        "--clean-out-dir",
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr

    expected_files = {
        "data_audit_summary.json",
        "data_audit_report.csv",
        "coverage_report.csv",
        "data_gaps_report.csv",
        "entity_quality_report.csv",
        "feature_availability_matrix.csv",
        "next_data_requirements.csv",
        "data_audit_report.html",
    }
    missing = [name for name in expected_files if not (out_dir / name).exists()]
    assert not missing, f"Missing data audit outputs: {missing}"

    summary = json.loads((out_dir / "data_audit_summary.json").read_text(encoding="utf-8"))
    assert summary["version"] == "v0.49.0_data_audit_report"
    assert summary["principles"]["offline_only"] is True
    assert summary["principles"]["model_logic_changed"] is False
    assert summary["principles"]["external_api_calls"] is False
    assert summary["principles"]["player_props_conservative"] is True

    report = pd.read_csv(out_dir / "data_audit_report.csv")
    assert {"fixtures", "lineups", "squads", "player_events", "actual_results"}.issubset(set(report["dataset"]))
    fixtures_row = report[report["dataset"].eq("fixtures")].iloc[0]
    assert int(fixtures_row["row_count"]) == 2
    assert str(fixtures_row["data_quality_flag"]) in {"high", "medium"}

    coverage = pd.read_csv(out_dir / "coverage_report.csv")
    assert "lineups_match_coverage_vs_fixtures" in set(coverage["coverage_area"])
    assert "current_squad_player_coverage_for_lineups" in set(coverage["coverage_area"])

    entity_quality = pd.read_csv(out_dir / "entity_quality_report.csv")
    assert "lineup_player_not_in_current_squad" in set(entity_quality["issue"])

    availability = pd.read_csv(out_dir / "feature_availability_matrix.csv")
    assert "golden_boot_projection" in set(availability["feature"])
    assert "player_props_current_inference" in set(availability["feature"])

    requirements = pd.read_csv(out_dir / "next_data_requirements.csv")
    assert "player_events" in set(requirements["area"])
    assert "current_squads_lineups" in set(requirements["area"])
    assert "forward_prediction_snapshots" in set(requirements["area"])

    html = (out_dir / "data_audit_report.html").read_text(encoding="utf-8")
    assert "Mundialytics Data Audit Report" in html
    assert "Dataset Health" in html
    assert "Coverage Report" in html
    assert "Feature Availability Matrix" in html
    assert "Next Data Requirements" in html
