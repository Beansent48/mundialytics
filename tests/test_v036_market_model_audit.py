from __future__ import annotations

import pandas as pd

from mundialytics.betting.pick_policy import (
    add_chronological_split,
    build_match_pick_signals,
    evaluate_model_performance_by_market,
    build_market_takeaways,
)


def test_v036_market_model_performance_outputs_market_and_line_metrics():
    df = pd.DataFrame([
        {"match_id": i, "date": f"2026-01-{i+1:02d}", "home_team": "a", "away_team": "b", "actual_home_goals": 2, "actual_away_goals": 1, "p_home_win": 0.55, "p_draw": 0.25, "p_away_win": 0.20, "p_over_05": 0.9, "p_over_15": 0.75, "p_over_25": 0.55, "p_over_35": 0.35, "p_btts": 0.6}
        for i in range(1, 12)
    ])
    signals = add_chronological_split(build_match_pick_signals(df), train_frac=0.5, validation_frac=0.25)
    market_summary, line_summary = evaluate_model_performance_by_market(signals)
    assert not market_summary.empty
    assert not line_summary.empty
    assert {"1x2", "goals", "btts"}.issubset(set(market_summary["market"]))
    assert {"hit_rate", "avg_model_probability", "calibration_gap", "brier", "log_loss"}.issubset(market_summary.columns)
    takeaways = build_market_takeaways(market_summary, min_test_n=1)
    assert takeaways["status"] == "completed"
    assert any(row["market"] == "goals" for row in takeaways["markets"])
