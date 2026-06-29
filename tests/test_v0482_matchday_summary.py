from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_v0482_matchday_summary_outputs_and_report_sections(tmp_path: Path) -> None:
    out_dir = tmp_path / "matchday_summary_v0482"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_statistical_matchday.py"),
        "--fixtures",
        str(ROOT / "data" / "input" / "fixtures.csv"),
        "--lineups",
        str(ROOT / "data" / "input" / "current_lineups.csv"),
        "--squads",
        str(ROOT / "data" / "input" / "squads.csv"),
        "--odds",
        str(ROOT / "data" / "input" / "odds.csv"),
        "--tournament-config",
        str(ROOT / "data" / "input" / "tournament_config.csv"),
        "--historical-events",
        str(ROOT / "data" / "sample" / "sample_player_events.csv"),
        "--out-dir",
        str(out_dir),
        "--n-simulations",
        "30",
        "--seed",
        "42",
        "--clean-out-dir",
    ]

    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr

    summary_csv = out_dir / "matchday_summary.csv"
    summary_json = out_dir / "matchday_summary.json"
    report_path = out_dir / "daily_report.html"
    contract_path = out_dir / "simulation_contract_report.json"
    audit_path = out_dir / "audit_report.json"

    assert summary_csv.exists()
    assert summary_json.exists()
    assert report_path.exists()
    assert contract_path.exists()
    assert audit_path.exists()

    summary = pd.read_csv(summary_csv)
    assert not summary.empty

    required_columns = {
        "ranking_category",
        "rank",
        "match_id",
        "match",
        "metric_name",
        "metric_value",
        "statistical_label",
        "data_quality_flag",
        "short_structured_reason",
    }
    assert required_columns.issubset(summary.columns)

    categories = set(summary["ranking_category"].dropna().astype(str))
    expected_categories = {
        "high_goal_expectation",
        "low_goal_environment",
        "most_balanced_matches",
        "strongest_favorites",
        "highest_uncertainty",
        "btts_lean",
        "top_dynamic_statistical_signals",
    }
    assert expected_categories.issubset(categories)

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["version"] == "v0.48.2_matchday_summary_rankings"
    assert payload["paper_mode"] is True
    assert payload["principles"]["odds_required"] is False
    assert payload["principles"]["betting_recommendations"] is False
    assert payload["summary_rows"] == len(summary)
    assert "high_goal_expectation" in payload["categories"]

    html = report_path.read_text(encoding="utf-8")
    assert "Mundialytics Statistical Simulator v0.48.4" in html
    assert "Matchday Summary Rankings" in html
    assert "High Goal Expectation" in html
    assert "Most Balanced Matches" in html
    assert "Top Dynamic Statistical Signals" in html
    assert "These sections order the day statistically; they are not betting picks." in html
    assert "Tournament Visual Report" in html

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["contract_version"] == "v0.48.4_simulation_evaluation_foundation_contract"
    assert contract["status"] == "ok"
    assert contract["missing_files"] == []
    assert contract["schema_failures"] == []
    assert contract["html_report_sections"]["status"] == "ok"
    assert "matchday_summary.csv" in contract["files"]
    assert "matchday_summary.json" in contract["files"]
    assert "tournament_report.csv" in contract["files"]
    assert "tournament_report.json" in contract["files"]

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["version"] == "v0.48.4_simulation_evaluation_foundation"
    assert audit["matchday_summary"]["status"] == "completed"
    assert audit["matchday_summary"]["rows"] == len(summary)
