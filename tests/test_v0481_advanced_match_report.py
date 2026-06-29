from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v0481_advanced_match_report_sections_and_contract(tmp_path: Path) -> None:
    out_dir = tmp_path / "advanced_match_report_v0481"

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

    report_path = out_dir / "daily_report.html"
    contract_path = out_dir / "simulation_contract_report.json"
    audit_path = out_dir / "audit_report.json"

    assert report_path.exists()
    assert contract_path.exists()
    assert audit_path.exists()
    assert (out_dir / "matchday_summary.csv").exists()
    assert (out_dir / "matchday_summary.json").exists()
    assert (out_dir / "tournament_report.csv").exists()
    assert (out_dir / "tournament_report.json").exists()

    html = report_path.read_text(encoding="utf-8")
    required_sections = [
        "Mundialytics Statistical Simulator v0.48.4",
        "Advanced Match Report",
        "Executive Summary",
        "Matchday Summary Rankings",
        "Match Probabilities",
        "Advanced Match Cards",
        "Top Scorelines",
        "Dynamic Goal Lines",
        "Not Available Markets",
        "Team Statistics",
        "Player Statistics",
        "Data Quality",
        "Tournament Visual Report",
        "Simulation Metadata",
    ]
    for section in required_sections:
        assert section in html

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert audit["version"] == "v0.48.4_simulation_evaluation_foundation"
    assert audit["status"] == "completed"
    assert audit["focus"] == "statistical_simulator_first"
    assert audit["simulation_contract"]["status"] == "ok"

    assert contract["contract_version"] == "v0.48.4_simulation_evaluation_foundation_contract"
    assert contract["status"] == "ok"
    assert contract["html_report_sections"]["status"] == "ok"
    assert contract["report_section_failures"] == []
    assert contract["recommended_large_run_n_simulations"] == 50000
