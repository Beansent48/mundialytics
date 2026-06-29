from __future__ import annotations

import pandas as pd
import pytest

from mundialytics.evaluation.player_props import PlayerPropBacktestConfig, backtest_player_props
from mundialytics.inference.safe_props import predict_props_for_lineups


def _events() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", periods=8, freq="7D")
    for i, d in enumerate(dates):
        for team, opp, player, pos in [
            ("Spain", "Uruguay", "Player A", "FW"),
            ("Uruguay", "Spain", "Player B", "CM"),
        ]:
            rows.append({
                "match_id": f"m{i}",
                "date": d.date().isoformat(),
                "competition": "FIFA World Cup",
                "team_scope": "national",
                "team": team,
                "opponent": opp,
                "player": player,
                "player_id_global": f"pid_{player[-1].lower()}",
                "player_context_id": f"ctx_{player[-1].lower()}",
                "position": pos,
                "started": 1,
                "minutes": 90 if i < 6 else 15,  # latest actual minutes must not leak into expected_minutes
                "shots": 2 if pos == "FW" else 0,
                "shots_on_target": 1 if pos == "FW" else 0,
                "fouls_committed": 1,
                "fouls_drawn": 0,
                "yellow_cards": 0,
                "goals": 0,
                "assists": 0,
            })
    return pd.DataFrame(rows)


def test_backtest_preserves_metadata_and_avoids_observed_minutes_leakage():
    pred, summary = backtest_player_props(
        _events(),
        PlayerPropBacktestConfig(min_train_matches=4, test_matches=2, markets=("player_shots",), use_observed_test_minutes=False),
    )
    required = {"competition", "team_scope", "player_id_global", "player_context_id", "position", "started", "expected_minutes_source", "actual_minutes"}
    assert required.issubset(pred.columns)
    assert pred["date"].isna().mean() == 0
    assert not pred["expected_minutes_source"].str.contains("LEAKY|observed", case=False, regex=True).any()
    assert (pred["expected_minutes"] >= 55).all()  # starter floor/history, not actual 15-minute outcome
    assert (pred["actual_minutes"] == 15).any()


def test_safe_props_only_outputs_supplied_lineup_players():
    hist = _events()
    lineups = pd.DataFrame([
        {
            "match_id": "future_1", "date": "2026-06-26", "competition": "FIFA World Cup", "team_scope": "national",
            "team": "Spain", "opponent": "Uruguay", "player": "Current Only", "position": "FW", "expected_minutes": 80, "started": 1,
        }
    ])
    out = predict_props_for_lineups(hist, lineups, markets=("player_shots",), strict_lineup_contract=True)
    assert set(out["player"]) == {"current only"}
    assert "player a" not in set(out["player"])
    assert {"safe_probability", "competition", "team_scope", "player_id_global", "player_context_id"}.issubset(out.columns)
