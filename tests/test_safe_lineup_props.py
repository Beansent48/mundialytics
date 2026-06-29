from __future__ import annotations

import pandas as pd

from mundialytics.inference.safe_props import predict_props_for_lineups, safe_cap_probability


def test_safe_cap_limits_extreme_props() -> None:
    p, warnings = safe_cap_probability("player_shots", 0.999999, sample_size=1000)
    assert p == 0.95
    assert warnings
    p2, warnings2 = safe_cap_probability("player_yellow_card", 0.999999, sample_size=1000)
    assert p2 == 0.45
    assert warnings2


def test_lineup_inference_outputs_only_supplied_players() -> None:
    events = pd.DataFrame({
        "match_id": ["m1", "m1", "m2", "m2"],
        "date": ["2024-01-01", "2024-01-01", "2024-01-08", "2024-01-08"],
        "team": ["old fc", "old fc", "old fc", "new fc"],
        "opponent": ["new fc", "new fc", "new fc", "old fc"],
        "player": ["retired star", "current player", "retired star", "other player"],
        "position": ["ST", "RW", "ST", "CB"],
        "minutes": [90, 90, 90, 90],
        "shots": [5, 1, 4, 0],
        "shots_on_target": [2, 0, 1, 0],
        "fouls_committed": [1, 0, 2, 1],
        "yellow_cards": [0, 0, 1, 0],
    })
    lineups = pd.DataFrame({
        "match_id": ["future1"],
        "date": ["2026-06-26"],
        "team": ["New FC"],
        "opponent": ["Old FC"],
        "player": ["Current Player"],
        "position": ["RW"],
        "expected_minutes": [80],
        "started": [1],
    })
    out = predict_props_for_lineups(events, lineups, markets=["player_shots", "player_yellow_card"])
    assert set(out["player"]) == {"current player"}
    assert set(out["market_type"]) == {"player_shots", "player_yellow_card"}
    assert out["safe_probability"].between(0, 1).all()
