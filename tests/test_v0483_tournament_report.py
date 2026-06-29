from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_v0483_tournament_visual_report_outputs_and_contract(tmp_path: Path) -> None:
    out_dir = tmp_path / "tournament_visual_report_v0483"

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
        "40",
        "--seed",
        "42",
        "--clean-out-dir",
    ]

    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr

    report_csv_path = out_dir / "tournament_report.csv"
    report_json_path = out_dir / "tournament_report.json"
    html_path = out_dir / "daily_report.html"
    contract_path = out_dir / "simulation_contract_report.json"
    audit_path = out_dir / "audit_report.json"

    assert report_csv_path.exists()
    assert report_json_path.exists()
    assert html_path.exists()
    assert contract_path.exists()
    assert audit_path.exists()

    report = pd.read_csv(report_csv_path)
    assert not report.empty

    required_columns = {
        "report_section",
        "rank",
        "team",
        "metric_name",
        "metric_value",
        "statistical_label",
        "data_quality_flag",
        "short_structured_reason",
    }
    assert required_columns.issubset(report.columns)

    sections = set(report["report_section"].dropna().astype(str))
    expected_sections = {
        "championship_race",
        "qualification_race",
        "group_winner_race",
        "expected_group_table",
        "uncertainty_watchlist",
    }
    assert expected_sections.issubset(sections)

    payload = json.loads(report_json_path.read_text(encoding="utf-8"))
    assert payload["version"] == "v0.48.4_simulation_evaluation_foundation"
    assert payload["paper_mode"] is True
    assert payload["principles"]["odds_required"] is False
    assert payload["principles"]["betting_recommendations"] is False
    assert payload["principles"]["model_logic_changed"] is False
    assert payload["summary_rows"] == len(report)
    assert "championship_race" in payload["categories"]

    html = html_path.read_text(encoding="utf-8")
    assert "Mundialytics Statistical Simulator v0.48.4" in html
    assert "Tournament Visual Report" in html
    assert "Championship Race" in html
    assert "Qualification Race" in html
    assert "Expected Group Tables" in html
    assert "Uncertainty Watchlist" in html
    assert "This is not a betting screen." in html

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["contract_version"] == "v0.48.4_simulation_evaluation_foundation_contract"
    assert contract["status"] == "ok"
    assert contract["missing_files"] == []
    assert contract["schema_failures"] == []
    assert contract["html_report_sections"]["status"] == "ok"
    assert "tournament_report.csv" in contract["files"]
    assert "tournament_report.json" in contract["files"]

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["version"] == "v0.48.4_simulation_evaluation_foundation"
    assert audit["tournament_report"]["status"] == "completed"
    assert audit["tournament_report"]["rows"] == len(report)
