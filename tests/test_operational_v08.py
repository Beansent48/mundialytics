import pandas as pd

from mundialytics.evaluation.backtest_runner import BacktestConfig, walk_forward_backtest
from mundialytics.evaluation.player_props import PlayerPropBacktestConfig, backtest_player_props
from mundialytics.models.goal_model import GoalLambdaModel, GoalModelConfig


def test_goal_model_drops_all_null_event_features_without_warning():
    frame = pd.DataFrame({
        "goals_for": [1, 0, 2, 1],
        "team_elo": [1500, 1510, 1490, 1520],
        "opponent_elo": [1490, 1500, 1510, 1480],
        "elo_diff": [10, 10, -20, 40],
        "is_home_non_neutral": [1, 0, 1, 0],
        "goals_for_last5": [1.0, 0.8, 1.2, 1.1],
        "shots_for_last5": [float("nan")] * 4,
        "competition": ["a", "a", "b", "b"],
        "stage": ["league", "league", "cup", "cup"],
    })
    model = GoalLambdaModel(GoalModelConfig(model_type="poisson")).fit(frame)
    nums, _ = model._available_features(frame.drop(columns=["goals_for"]))
    assert "team_elo" in nums
    assert "shots_for_last5" not in nums


def test_chunked_backtest_returns_predictions_quickly():
    rows = []
    teams = ["A", "B", "C", "D"]
    for i in range(60):
        h = teams[i % 4]
        a = teams[(i + 1) % 4]
        rows.append({
            "match_id": f"m{i}",
            "date": f"2023-01-{(i%28)+1:02d}",
            "home_team": h,
            "away_team": a,
            "home_goals": i % 3,
            "away_goals": (i + 1) % 2,
            "neutral": 0,
            "competition": "sample",
            "stage": "league",
            "team_scope": "club",
        })
    matches = pd.DataFrame(rows)
    pred, summary = walk_forward_backtest(matches, BacktestConfig(min_train_matches=30, retrain_every=10, max_test_matches=20))
    assert len(pred) == 20
    assert summary["n_predictions"] == 20


def test_player_props_backtest_synthetic_events():
    rows = []
    for m in range(30):
        for player in ["winger", "midfielder"]:
            is_winger = player == "winger"
            rows.append({
                "match_id": f"m{m}",
                "date": f"2024-01-{(m%28)+1:02d}",
                "team": "team_a",
                "opponent": "team_b",
                "team_scope": "club",
                "competition": "sample",
                "player": player,
                "position": "Right Wing" if is_winger else "Center Midfield",
                "minutes": 80,
                "shots": 2 if is_winger else 0,
                "shots_on_target": 1 if is_winger and m % 2 == 0 else 0,
                "fouls_committed": 0 if is_winger else 1,
                "fouls_drawn": 1 if is_winger else 0,
                "yellow_cards": 0 if is_winger else (1 if m % 5 == 0 else 0),
                "goals": 0,
                "assists": 0,
            })
    pred, summary = backtest_player_props(pd.DataFrame(rows), PlayerPropBacktestConfig(min_train_matches=10, test_matches=10))
    assert not pred.empty
    assert summary["n_predictions"] > 0
    assert "player_shots" in summary["markets"]
