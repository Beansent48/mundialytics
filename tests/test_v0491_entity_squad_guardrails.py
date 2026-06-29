
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_guardrail_inputs(tmp_path: Path) -> dict[str, Path]:
    fixtures = tmp_path / "fixtures.csv"
    lineups = tmp_path / "lineups.csv"
    squads = tmp_path / "squads.csv"
    player_events = tmp_path / "player_events.csv"

    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2026-06-21",
                "home_team": "Alpha",
                "away_team": "Beta",
                "team_scope": "national",
                "provider": "sample_provider",
                "provider_fixture_id": "fx_conflict",
            },
            {
                "match_id": "m2",
                "date": "2026-06-22",
                "home_team": "Gamma",
                "away_team": "Delta",
                "team_scope": "national",
                "provider": "sample_provider",
                "provider_fixture_id": "fx_conflict",
            },
        ]
    ).to_csv(fixtures, index=False)

    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "team": "Alpha",
                "player": "Player One",
                "player_id": "p1",
                "team_scope": "national",
                "lineup_status": "starter",
            },
            {
                "match_id": "m1",
                "team": "Alpha",
                "player": "No Id Forward",
                "player_id": "",
                "team_scope": "national",
                "lineup_status": "starter",
            },
            {
                "match_id": "m2",
                "team": "Gamma",
                "player": "Old Striker",
                "player_id": "p_old",
                "team_scope": "national",
                "lineup_status": "starter",
            },
            {
                "match_id": "m2",
                "team": "Club Only FC",
                "player": "Wrong Context Player",
                "player_id": "p_wrong",
                "team_scope": "club",
                "lineup_status": "starter",
            },
        ]
    ).to_csv(lineups, index=False)

    pd.DataFrame(
        [
            {
                "team": "Alpha",
                "player": "Player One",
                "player_id": "p1",
                "current_squad_flag": True,
                "availability_status": "available",
                "team_scope": "national",
            },
            {
                "team": "Gamma",
                "player": "Old Striker",
                "player_id": "p_old",
                "current_squad_flag": False,
                "availability_status": "unavailable",
                "team_scope": "national",
            },
        ]
    ).to_csv(squads, index=False)

    pd.DataFrame(
        [
            {
                "match_id": "old1",
                "date": "2025-01-01",
                "team": "Alpha",
                "opponent": "Beta",
                "player": "Retired Ace",
                "player_id": "p_retired",
                "minutes": 70,
                "goals": 1,
            },
            {
                "match_id": "old2",
                "date": "2025-01-02",
                "team": "Gamma",
                "opponent": "Delta",
                "player": "Old Striker",
                "player_id": "p_old",
                "minutes": 60,
                "goals": 0,
            },
        ]
    ).to_csv(player_events, index=False)

    return {
        "fixtures": fixtures,
        "lineups": lineups,
        "squads": squads,
        "player_events": player_events,
    }


def test_v0491_entity_squad_guardrails_report_reason_codes(tmp_path: Path) -> None:
    paths = _write_guardrail_inputs(tmp_path)
    out_dir = tmp_path / "guardrail_audit"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_data_audit.py"),
        "--fixtures",
        str(paths["fixtures"]),
        "--lineups",
        str(paths["lineups"]),
        "--squads",
        str(paths["squads"]),
        "--player-events",
        str(paths["player_events"]),
        "--out-dir",
        str(out_dir),
        "--run-label",
        "unit_v0491_guardrails",
        "--clean-out-dir",
    ]

    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr

    expected_files = {
        "entity_guardrails_report.csv",
        "squad_guardrails_report.csv",
        "guardrail_summary.json",
        "data_audit_summary.json",
        "data_audit_report.html",
    }
    missing = [name for name in expected_files if not (out_dir / name).exists()]
    assert not missing, f"Missing guardrail outputs: {missing}"

    entity_guardrails = pd.read_csv(out_dir / "entity_guardrails_report.csv")
    squad_guardrails = pd.read_csv(out_dir / "squad_guardrails_report.csv")
    summary = json.loads((out_dir / "guardrail_summary.json").read_text(encoding="utf-8"))
    audit_summary = json.loads((out_dir / "data_audit_summary.json").read_text(encoding="utf-8"))

    entity_reasons = set(entity_guardrails["reason_code"])
    squad_reasons = set(squad_guardrails["reason_code"])

    assert "provider_fixture_id_conflict" in entity_reasons
    assert "team_scope_mismatch" in entity_reasons
    assert "team_not_in_fixture" in entity_reasons

    assert "lineup_player_without_player_id" in squad_reasons
    assert "missing_current_eligibility" in squad_reasons
    assert "player_not_in_current_squad" in squad_reasons
    assert "historical_only_player_for_current_inference" in squad_reasons

    assert summary["version"] == "v0.49.1_entity_squad_guardrails"
    assert summary["status"] == "blocked"
    assert summary["unsafe_for_player_props"] is True
    assert audit_summary["guardrails"]["status"] == "blocked"

    html = (out_dir / "data_audit_report.html").read_text(encoding="utf-8")
    assert "Entity Guardrails" in html
    assert "Squad Guardrails" in html
