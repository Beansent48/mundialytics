import pandas as pd

from mundialytics.betting.pick_policy import standardize_settled_line_signals, evaluate_policy_grid


def test_standardize_settled_line_signals_supports_typical_over_under_markets():
    df = pd.DataFrame({
        "match_id": ["m1", "m1", "m1", "m1", "m1"],
        "market": ["corners", "team shots", "team shots on target", "yellow_cards", "goalkeeper saves"],
        "scope": ["match", "team", "team", "match", "player"],
        "selection": ["under", "over", "under", "over", "under"],
        "line": [9.5, 10.5, 3.5, 4.5, 2.5],
        "model_probability": [0.64, 0.61, 0.58, 0.57, 0.55],
        "settled_stat": [8, 12, 2, 5, 1],
    })
    out = standardize_settled_line_signals(df)
    assert set(out["signal_group"]) == {
        "corners_under",
        "team_shots_over",
        "team_shots_on_target_under",
        "yellow_cards_over",
        "goalkeeper_saves_under",
    }
    assert out["actual_win"].eq(1).all()


def test_policy_grid_can_select_event_market_sides_when_present():
    rows = []
    for i in range(80):
        rows.append({
            "match_id": f"m{i}",
            "date": f"2024-01-{(i%28)+1:02d}",
            "market": "team_shots_on_target",
            "scope": "team",
            "selection": "under",
            "line": 4.5,
            "model_probability": 0.70,
            "settled_stat": 3 if i % 3 else 6,
        })
    signals = standardize_settled_line_signals(pd.DataFrame(rows))
    leaderboard, best = evaluate_policy_grid(signals, min_picks=5, require_odds=False)
    assert not leaderboard.empty
    assert "team_shots_on_target_under" in set(leaderboard["allowed_signal_group"])
