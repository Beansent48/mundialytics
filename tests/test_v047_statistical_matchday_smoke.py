from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v047_statistical_matchday_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "statistical_matchday_v047"

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

    expected_files = {
        "match_predictions.csv",
        "scoreline_distribution.csv",
        "team_stats_predictions.csv",
        "player_event_predictions.csv",
        "dynamic_market_lines.csv",
        "tournament_simulation.csv",
        "audit_report.json",
        "daily_report.html",
        "simulation_contract_report.json",
        "matchday_summary.csv",
        "matchday_summary.json",
        "tournament_report.csv",
        "tournament_report.json",
    }
    missing = [name for name in expected_files if not (out_dir / name).exists()]
    assert not missing, f"Missing expected outputs: {missing}"

    audit = json.loads((out_dir / "audit_report.json").read_text(encoding="utf-8"))
    assert audit["version"] in {"v0.47_prediction_simulation_consolidation", "v0.48_statistical_simulator_upgrade", "v0.48.1_advanced_match_report", "v0.48.2_matchday_summary_rankings", "v0.48.4_simulation_evaluation_foundation"}
    assert audit["paper_mode"] is True
    assert audit["status"] == "completed"
    assert "corners_not_available_not_invented" in audit["warnings"]
    assert "goalkeeper_saves_not_available_not_invented" in audit["warnings"]
