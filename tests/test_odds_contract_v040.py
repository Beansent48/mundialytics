from __future__ import annotations

import math
import pandas as pd

from mundialytics.betting.odds_contract import (
    fair_odds_from_probability,
    min_acceptable_odds_from_probability,
    standard_model_line_frame,
    standard_odds_input_frame,
    merge_model_lines_with_odds,
)


def test_fair_and_min_acceptable_odds():
    assert round(fair_odds_from_probability(0.70), 3) == 1.429
    required = min_acceptable_odds_from_probability(0.70, min_ev=0.03, min_edge=0.02)
    assert required > 1.429
    assert required >= (1.03 / 0.70)


def test_standard_model_line_schema():
    df = pd.DataFrame([
        {"match_id": "m1", "market": "team_yellow_cards", "scope": "team", "team": "Spain", "selection": "over", "line": 1.5, "model_probability": 0.70},
    ])
    out = standard_model_line_frame(df)
    assert out.loc[0, "market_key"] == "team_yellow_cards"
    assert out.loc[0, "subject_team"] == "Spain"
    assert round(float(out.loc[0, "fair_odds"]), 3) == 1.429


def test_merge_model_lines_with_odds_and_ev():
    model = pd.DataFrame([
        {"match_id": "m1", "market": "team_yellow_cards", "scope": "team", "subject_team": "Spain", "selection": "over", "side": "over", "line": 1.5, "model_probability": 0.70, "actual_win": 1},
    ])
    odds = pd.DataFrame([
        {"match_id": "m1", "market": "team_yellow_cards", "scope": "team", "subject_team": "Spain", "side": "over", "line": 1.5, "bookmaker_odds": 1.80, "bookmaker": "demo"},
    ])
    edges = merge_model_lines_with_odds(model, odds)
    assert len(edges) == 1
    assert round(float(edges.loc[0, "ev"]), 3) == 0.260
    assert float(edges.loc[0, "edge"]) > 0
    assert edges.loc[0, "value_label"] in {"value", "high_value"}
    assert round(float(edges.loc[0, "profit_1u"]), 3) == 0.800
