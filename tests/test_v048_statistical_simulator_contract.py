from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_v048_statistical_simulator_contract_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "statistical_simulator_v048"

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
        "25",
        "--seed",
        "42",
        "--clean-out-dir",
    ]

    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr

    audit_path = out_dir / "audit_report.json"
    contract_path = out_dir / "simulation_contract_report.json"
    report_path = out_dir / "daily_report.html"

    assert audit_path.exists()
    assert contract_path.exists()
    assert report_path.exists()
    assert (out_dir / "matchday_summary.csv").exists()
    assert (out_dir / "matchday_summary.json").exists()
    assert (out_dir / "tournament_report.csv").exists()
    assert (out_dir / "tournament_report.json").exists()

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert audit["version"] == "v0.48.4_simulation_evaluation_foundation"
    assert audit["focus"] == "statistical_simulator_first"
    assert audit["paper_mode"] is True
    assert audit["status"] == "completed"
    assert audit["simulation_policy"]["recommended_large_run_n_simulations"] == 50000
    assert audit["tournament_simulator"]["n_simulations"] == 25
    assert audit["tournament_simulator"]["seed"] == 42
    assert audit["simulation_contract"]["status"] == "ok"

    assert contract["contract_version"] == "v0.48.4_simulation_evaluation_foundation_contract"
    assert contract["status"] == "ok"
    assert contract["recommended_large_run_n_simulations"] == 50000
    assert contract["missing_files"] == []
    assert contract["schema_failures"] == []

    tournament = pd.read_csv(out_dir / "tournament_simulation.csv")
    assert not tournament.empty
    assert set(["team", "champion_probability", "qualify_group_probability", "simulations", "seed"]).issubset(tournament.columns)
    assert int(tournament["simulations"].max()) == 25
    assert int(tournament["seed"].max()) == 42

    html = report_path.read_text(encoding="utf-8")
    assert "Mundialytics Statistical Simulator v0.48.4" in html
    assert "Advanced Match Report" in html
    assert "Executive Summary" in html
    assert "Matchday Summary Rankings" in html
    assert "Match Probabilities" in html
    assert "Top Scorelines" in html
    assert "Dynamic Goal Lines" in html
    assert "Data Quality" in html
    assert "Tournament Visual Report" in html
    assert "Simulation Metadata" in html

    assert contract["html_report_sections"]["status"] == "ok"
    assert contract["report_section_failures"] == []
