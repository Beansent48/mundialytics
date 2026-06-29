from __future__ import annotations

import pandas as pd

from mundialytics.evaluation.hierarchical_prop_calibration import run_hierarchical_calibration_search
from mundialytics.evaluation.player_prop_policy import build_player_prop_policy, choose_market_calibration_policy
from mundialytics.inference.safe_props import predict_props_for_lineups
from mundialytics.data.identity import player_global_id


def _synthetic_competition_rows() -> pd.DataFrame:
    rows = []
    # La Liga and Ligue 1 have different calibration needs. We create a stable
    # temporal split: first half calibration, second half test.
    for i in range(80):
        for comp, threshold in [("La Liga", 0.70), ("Ligue 1", 0.30)]:
            rows.append({
                "match_id": f"m{i}_{comp}",
                "date": f"2024-01-{(i % 28) + 1:02d}",
                "competition": comp,
                "team_type": "club",
                "team_scope": "club",
                "competition_context": "domestic_league",
                "gender": "men",
                "player": f"P{i}",
                "market_type": "player_shots",
                "probability": 0.50,
                "actual": int((i % 100) < threshold * 100),
                "expected_minutes": 75,
                "sample_size": 900,
                "expected_count": 1.0,
            })
    return pd.DataFrame(rows)


def test_v17_adaptive_hierarchical_reports_competition_diagnostics_and_selection_mode():
    _, calibrated, report = run_hierarchical_calibration_search(
        _synthetic_competition_rows(),
        calibration_fraction=0.5,
        min_group_rows=20,
        min_market_rows=20,
        selection_mode="adaptive",
    )
    assert report["selection_mode"] == "adaptive"
    assert not calibrated.empty
    assert "competition_diagnostics" in report
    assert any("La Liga" in k or "Ligue 1" in k for k in report["competition_diagnostics"])
    assert calibrated["calibration_level"].isin(["competition", "domain_context", "team_type_gender", "market_global"]).all()


def test_v17_policy_prefers_hierarchical_when_bias_improves_without_big_logloss_penalty():
    policy = choose_market_calibration_policy(
        market="player_shots_on_target",
        simple_metrics={"log_loss": 0.50, "brier": 0.16, "probability_bias": -0.08},
        hierarchical_metrics={"log_loss": 0.515, "brier": 0.165, "probability_bias": -0.01},
        simple_method="platt_logit_extra",
        hierarchical_level_counts={"competition": 1000},
    )
    assert policy["use_hierarchical"] is True
    assert policy["recommended_source"] == "hierarchical"
    assert "safe_probability_caps" in policy


def test_v17_safe_lineup_can_follow_policy_to_use_simple_market_calibrator():
    player = "Policy Player"
    gid = player_global_id(player)
    hist = []
    for i in range(10):
        hist.append({
            "match_id": f"h{i}", "date": f"2024-01-{i+1:02d}", "competition": "La Liga",
            "team": "Club A", "opponent": "Club B", "player": player, "player_id_global": gid,
            "position": "FW", "started": 1, "minutes": 80,
            "shots": 2, "shots_on_target": 1, "fouls_committed": 1, "fouls_drawn": 1, "yellow_cards": 0,
        })
    cal = []
    for i in range(70):
        cal.append({
            "match_id": f"c{i}", "date": f"2024-02-{(i % 28) + 1:02d}", "competition": "La Liga",
            "team_type": "club", "team_scope": "club", "competition_context": "domestic_league", "gender": "men",
            "market_type": "player_shots", "probability": 0.45, "actual": int(i % 2 == 0),
            "expected_minutes": 70, "sample_size": 900, "expected_count": 1.0,
        })
    lineups = pd.DataFrame([{
        "match_id": "future_1", "date": "2024-07-01", "competition": "La Liga",
        "team": "Club A", "opponent": "Club B", "player": player, "player_id_global": gid,
        "position": "FW", "started": 1, "expected_minutes": 75,
    }])
    policy = {"markets": {"player_shots": {"recommended_source": "simple_market", "readiness_status": "yellow", "reason": "unit_test"}}}
    out = predict_props_for_lineups(
        pd.DataFrame(hist), lineups, markets=("player_shots",), calibration_predictions=pd.DataFrame(cal),
        min_calibration_rows=20, min_hierarchical_group_rows=20, market_calibration_policy=policy,
    )
    assert out["calibration_policy_source"].iloc[0] == "simple_market"
    assert out["calibration_level"].iloc[0] == "market_global_simple"
    assert out["calibration_policy_status"].iloc[0] == "yellow"
