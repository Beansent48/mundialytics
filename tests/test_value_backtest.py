import pandas as pd

from mundialytics.evaluation.value_backtest import run_match_value_backtest


def test_value_backtest_settles_only_value_flags():
    preds = pd.DataFrame({
        "match_id": ["m1"],
        "date": ["2026-01-01"],
        "competition": ["Test"],
        "team_scope": ["club"],
        "home_team": ["barcelona"],
        "away_team": ["valencia"],
        "actual_outcome": ["H"],
        "p_home_win": [0.70],
        "p_draw": [0.18],
        "p_away_win": [0.12],
        "lambda_home": [2.0],
        "lambda_away": [0.8],
        "most_likely_score": ["2-0"],
    })
    odds = pd.DataFrame({
        "match_id": ["m1", "m1", "m1"],
        "bookmaker": ["demo", "demo", "demo"],
        "market_type": ["match_winner", "match_winner", "match_winner"],
        "selection": ["home", "draw", "away"],
        "odds": [1.80, 4.00, 8.00],
    })
    settled, summary = run_match_value_backtest(preds, odds, min_edge=0.0, min_ev=0.0)
    assert summary["bets"] >= 1
    assert "profit" in settled.columns
    home = settled[settled["selection_type"] == "home"].iloc[0]
    assert bool(home["won"]) is True

from mundialytics.evaluation.backtest_runner import walk_forward_backtest, BacktestConfig
from mundialytics.data.loaders import load_matches
from pathlib import Path


def test_backtest_summary_includes_reliability_bins():
    matches = load_matches(Path("data/sample/sample_matches.csv"))
    pred, summary = walk_forward_backtest(matches, BacktestConfig(min_train_matches=10, model_type="poisson", retrain_every=5))
    assert "picked_probability" in pred.columns
    assert "reliability_by_confidence_bin" in summary
