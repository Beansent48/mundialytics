from __future__ import annotations

import pandas as pd

from mundialytics.statistical_core.market_distribution_lab import (
    prepare_line_signals,
    unique_stat_predictions,
    summarize_exact_error,
    summarize_range_coverage,
    build_range_frame,
    summarize_fair_odds_buckets,
    build_decision_matrix,
)


def test_v039_distribution_lab_outputs_market_side_metrics():
    rows = []
    for i in range(30):
        actual = 5 + (i % 3)
        expected = 5.8
        for line in [4.5, 5.5, 6.5]:
            p_over = 0.72 if line == 4.5 else (0.58 if line == 5.5 else 0.38)
            for sel, p in [("over", p_over), ("under", 1 - p_over)]:
                rows.append({
                    "match_id": f"m{i}",
                    "date": f"2024-01-{(i % 28) + 1:02d}",
                    "market": "team_corners",
                    "scope": "team",
                    "selection": sel,
                    "signal_group": f"team_corners_{sel}",
                    "target_quality": "real_target",
                    "line": line,
                    "model_probability": p,
                    "fair_odds": 1 / p,
                    "settled_stat": actual,
                    "expected_stat": expected,
                    "actual_win": int(actual > line) if sel == "over" else int(actual < line),
                    "model_family": "smoke",
                })
    signals = prepare_line_signals(pd.DataFrame(rows), min_model_probability=0.0)
    assert not signals.empty
    assert {"fair_odds_bucket", "line_margin", "line_margin_bucket", "split"}.issubset(signals.columns)

    preds = unique_stat_predictions(signals)
    assert len(preds) == 30
    exact = summarize_exact_error(preds)
    assert "mae" in exact.columns
    range_summary = summarize_range_coverage(build_range_frame(preds))
    assert "empirical_coverage" in range_summary.columns
    fair = summarize_fair_odds_buckets(signals)
    assert set(fair["selection"].unique()) >= {"over", "under"}
    matrix = build_decision_matrix(signals, min_sample=1)
    assert not matrix.empty
    assert "decision" in matrix.columns
