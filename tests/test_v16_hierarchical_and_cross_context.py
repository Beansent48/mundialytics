from __future__ import annotations

import pandas as pd

from mundialytics.evaluation.hierarchical_prop_calibration import run_hierarchical_calibration_search
from mundialytics.evaluation.player_props import PlayerPropBacktestConfig, backtest_player_props
from mundialytics.data.identity import player_global_id, player_context_id
from mundialytics.data.competition_taxonomy import enrich_competition_metadata
from mundialytics.inference.safe_props import predict_props_for_lineups


def test_hierarchical_calibration_prefers_competition_when_sample_sufficient():
    rows = []
    for i in range(80):
        for comp, actual_rate in [("La Liga", 0.65), ("Ligue 1", 0.25)]:
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
                "probability": 0.45,
                "actual": int((i % 100) < actual_rate * 100),
                "expected_minutes": 75,
                "sample_size": 900,
                "expected_count": 1.0,
            })
    df = pd.DataFrame(rows)
    results, calibrated, report = run_hierarchical_calibration_search(
        df, calibration_fraction=0.5, min_group_rows=20, min_market_rows=20
    )
    assert not results.empty
    assert not calibrated.empty
    assert "competition" in set(calibrated["calibration_level"])
    assert report["selection_counts"].get("competition", 0) > 0


def test_national_backtest_can_use_pre_cutoff_club_history_without_future_leakage():
    club_rows = []
    national_rows = []
    player = "Test Player"
    gid = player_global_id(player)
    for i in range(10):
        club_rows.append({
            "match_id": f"club_{i}", "date": f"2023-01-{i+1:02d}", "competition": "La Liga",
            "team": "Club A", "opponent": "Club B", "player": player,
            "player_id_global": gid, "player_context_id": player_context_id(player, "Club A", "club", "La Liga"),
            "position": "FW", "started": 1, "minutes": 80,
            "shots": 3, "shots_on_target": 1, "fouls_committed": 1, "fouls_drawn": 1, "yellow_cards": 0,
        })
    # 4 train national matches + 3 test matches.
    for i in range(7):
        national_rows.append({
            "match_id": f"nat_{i}", "date": f"2024-06-{i+1:02d}", "competition": "UEFA Euro",
            "team": "Spain", "opponent": "France", "player": player,
            "player_id_global": gid, "player_context_id": player_context_id(player, "Spain", "national", "UEFA Euro"),
            "position": "FW", "started": 1, "minutes": 75,
            "shots": 1, "shots_on_target": 0, "fouls_committed": 1, "fouls_drawn": 1, "yellow_cards": 0,
        })
    target = enrich_competition_metadata(pd.DataFrame(national_rows), overwrite=True)
    features = enrich_competition_metadata(pd.DataFrame(club_rows + national_rows), overwrite=True)
    pred, summary = backtest_player_props(
        target,
        PlayerPropBacktestConfig(min_train_matches=4, test_matches=3, markets=("player_shots",), line="1+"),
        feature_events=features,
    )
    assert not pred.empty
    assert pred["cross_context_feature_used"].any()
    assert pred["club_minutes_sample"].max() > 0
    assert summary["feature_training"]["used_feature_events"] is True
    assert summary["feature_training"]["feature_max_date"] < "2024-06-05"


def test_safe_lineup_outputs_hierarchical_and_cross_context_columns():
    player = "Safe Player"
    gid = player_global_id(player)
    hist = []
    for i in range(12):
        hist.append({
            "match_id": f"h{i}", "date": f"2024-01-{(i % 28) + 1:02d}", "competition": "La Liga",
            "team": "Club A", "opponent": "Club B", "player": player, "player_id_global": gid,
            "position": "FW", "started": 1, "minutes": 80, "shots": 2, "shots_on_target": 1,
            "fouls_committed": 1, "fouls_drawn": 1, "yellow_cards": 0,
        })
    cal_rows = []
    for i in range(60):
        cal_rows.append({
            "match_id": f"c{i}", "date": f"2024-02-{(i % 28) + 1:02d}", "competition": "UEFA Euro",
            "team_type": "national_team", "team_scope": "national", "competition_context": "international_national_tournament", "gender": "men",
            "market_type": "player_shots", "probability": 0.45, "actual": int(i % 2 == 0),
            "expected_minutes": 70, "sample_size": 900, "expected_count": 1.0,
        })
    lineups = pd.DataFrame([{
        "match_id": "future_1", "date": "2024-07-01", "competition": "UEFA Euro",
        "team": "Spain", "opponent": "France", "player": player, "player_id_global": gid,
        "position": "FW", "started": 1, "expected_minutes": 75,
    }])
    out = predict_props_for_lineups(
        pd.DataFrame(hist), lineups, markets=("player_shots",), calibration_predictions=pd.DataFrame(cal_rows),
        min_calibration_rows=20, min_hierarchical_group_rows=20,
    )
    assert {"calibration_level", "club_minutes_sample", "cross_context_feature_used"}.issubset(out.columns)
    assert out["club_minutes_sample"].iloc[0] > 0
    assert bool(out["cross_context_feature_used"].iloc[0]) is True
