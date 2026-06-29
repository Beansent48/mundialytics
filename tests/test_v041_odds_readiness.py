from __future__ import annotations

import pandas as pd

from mundialytics.betting.odds_readiness import (
    audit_historical_odds_coverage,
    file_readiness_report,
    market_requirements_frame,
    summarize_odds_template,
)


def test_market_requirements_include_priority_targets():
    req = market_requirements_frame()
    keys = set(req["market_key"])
    assert "team_yellow_cards" in keys
    assert "team_fouls" in keys
    assert "goalkeeper_saves" in keys


def test_file_readiness_report_is_non_throwing(tmp_path):
    present = tmp_path / "present.csv"
    present.write_text("a\n1\n", encoding="utf-8")
    missing = tmp_path / "missing.csv"
    report = file_readiness_report({"present": present, "missing": missing})
    assert report["ready"] is False
    assert report["files"]["present"]["exists"] is True
    assert report["files"]["missing"]["exists"] is False


def test_summarize_odds_template_flags_missing_subjects():
    template = pd.DataFrame([
        {"match_id": "m1", "market_key": "team_yellow_cards", "market": "team_yellow_cards", "scope": "team", "line": 1.5, "side": "over", "subject_team": "Spain"},
        {"match_id": "m2", "market_key": "goalkeeper_saves", "market": "goalkeeper_saves", "scope": "player", "line": 2.5, "side": "under", "subject_player": ""},
    ])
    shopping, summary = summarize_odds_template(template)
    assert summary["rows"] == 2
    assert "missing_subject_player" in summary["readiness_flags"]
    assert shopping.loc[shopping["market_key"].eq("team_yellow_cards"), "readiness_flag"].iloc[0] == "ok"


def test_audit_historical_odds_coverage_uses_strict_keys():
    model = pd.DataFrame([
        {"match_id": "m1", "market_key": "team_yellow_cards", "market": "team_yellow_cards", "scope": "team", "subject_team": "Spain", "line": 1.5, "side": "over", "model_probability": 0.70},
        {"match_id": "m1", "market_key": "team_yellow_cards", "market": "team_yellow_cards", "scope": "team", "subject_team": "Spain", "line": 2.5, "side": "over", "model_probability": 0.55},
    ])
    odds = pd.DataFrame([
        {"match_id": "m1", "market_key": "team_yellow_cards", "market": "team_yellow_cards", "scope": "team", "subject_team": "Spain", "line": 1.5, "side": "over", "bookmaker_odds": 1.80},
    ])
    by_market, summary = audit_historical_odds_coverage(model, odds)
    assert summary["model_lines"] == 2
    assert summary["priced_lines"] == 1
    assert round(summary["coverage_pct"], 2) == 0.50
    assert by_market["coverage_pct"].iloc[0] == 0.5
