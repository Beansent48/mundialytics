import pandas as pd
import pytest

from mundialytics.reports.match_value import build_match_value_picks


def _predictions():
    return pd.DataFrame({
        "fixture_id": ["fx1"],
        "date": ["2026-06-01"],
        "competition": ["World Cup"],
        "team_scope": ["national"],
        "home_team": ["spain"],
        "away_team": ["uruguay"],
        "p_home_win": [0.55],
        "p_draw": [0.25],
        "p_away_win": [0.20],
        "lambda_home": [1.6],
        "lambda_away": [0.9],
        "most_likely_score": ["1-0"],
    })


def test_build_match_value_picks_maps_home_draw_away_and_normalizes_overround():
    odds = pd.DataFrame({
        "fixture_id": ["fx1", "fx1", "fx1"],
        "bookmaker": ["demo", "demo", "demo"],
        "market_type": ["match_winner", "match_winner", "match_winner"],
        "selection": ["home", "draw", "away"],
        "odds": [2.10, 3.50, 4.50],
    })
    picks = build_match_value_picks(_predictions(), odds, min_edge=0.0, min_ev=0.0)
    assert len(picks) == 3
    assert abs(picks["implied_probability"].sum() - 1.0) < 1e-9
    home = picks[picks["selection_type"] == "home"].iloc[0]
    assert home["model_probability"] == 0.55
    assert home["value_flag"] is True or bool(home["value_flag"]) is True


def test_build_match_value_picks_maps_team_names():
    odds = pd.DataFrame({
        "fixture_id": ["fx1", "fx1", "fx1"],
        "bookmaker": ["demo", "demo", "demo"],
        "market_type": ["match_winner", "match_winner", "match_winner"],
        "selection": ["Spain", "X", "Uruguay"],
        "odds": [1.80, 3.80, 5.10],
    })
    picks = build_match_value_picks(_predictions(), odds)
    assert set(picks["selection_type"]) == {"home", "draw", "away"}


def test_build_match_value_rejects_unknown_selection():
    odds = pd.DataFrame({
        "fixture_id": ["fx1"],
        "bookmaker": ["demo"],
        "market_type": ["match_winner"],
        "selection": ["Brazil"],
        "odds": [2.0],
    })
    with pytest.raises(ValueError):
        build_match_value_picks(_predictions(), odds)

from mundialytics.betting.value import shrink_probability


def test_shrink_probability_handles_pd_na_sample_size():
    assert shrink_probability(0.62, pd.NA, strength=180) == 0.62
    assert shrink_probability(0.62, sample_size=10, strength=0) == 0.62

from mundialytics.data.adapters.football_data_uk import football_data_uk_to_match_odds


def test_football_data_uk_to_match_odds_extracts_1x2(tmp_path):
    src = tmp_path / "E0.csv"
    pd.DataFrame({
        "Date": ["15/08/25"],
        "HomeTeam": ["Liverpool"],
        "AwayTeam": ["Bournemouth"],
        "FTHG": [4],
        "FTAG": [2],
        "B365H": [1.35],
        "B365D": [5.25],
        "B365A": [8.50],
    }).to_csv(src, index=False)
    odds = football_data_uk_to_match_odds(src)
    assert len(odds) == 3
    assert set(odds["selection"]) == {"home", "draw", "away"}
    assert odds["match_id"].iloc[0] == "fduk_E0_00000"


def test_build_match_value_picks_allows_missing_optional_team_scope():
    pred = _predictions().drop(columns=["team_scope"])
    odds = pd.DataFrame({
        "fixture_id": ["fx1", "fx1", "fx1"],
        "bookmaker": ["demo", "demo", "demo"],
        "market_type": ["match_winner", "match_winner", "match_winner"],
        "selection": ["home", "draw", "away"],
        "odds": [1.8, 3.8, 5.0],
    })
    picks = build_match_value_picks(pred, odds)
    assert picks["team_scope"].eq("unknown").all()


def test_build_match_value_picks_uses_predictions_as_descriptive_source_when_odds_overlap():
    pred = _predictions()
    odds = pd.DataFrame({
        "fixture_id": ["fx1", "fx1", "fx1"],
        "date": ["2099-01-01", "2099-01-01", "2099-01-01"],
        "competition": ["Wrong League", "Wrong League", "Wrong League"],
        "home_team": ["wrong home", "wrong home", "wrong home"],
        "away_team": ["wrong away", "wrong away", "wrong away"],
        "bookmaker": ["demo", "demo", "demo"],
        "market_type": ["match_winner", "match_winner", "match_winner"],
        "selection": ["home", "draw", "away"],
        "odds": [2.10, 3.50, 4.50],
    })
    picks = build_match_value_picks(pred, odds, min_edge=0.0, min_ev=0.0)
    assert picks["home_team"].eq("spain").all()
    assert picks["away_team"].eq("uruguay").all()
    assert picks["competition"].eq("World Cup").all()


def test_build_match_value_picks_filters_full_season_odds_to_prediction_window():
    pred = _predictions()
    odds = pd.DataFrame({
        "fixture_id": ["older_fx", "older_fx", "older_fx", "fx1", "fx1", "fx1"],
        "bookmaker": ["demo"] * 6,
        "market_type": ["match_winner"] * 6,
        "selection": ["home", "draw", "away", "home", "draw", "away"],
        "odds": [2.0, 3.0, 4.0, 2.10, 3.50, 4.50],
    })
    picks = build_match_value_picks(pred, odds, min_edge=0.0, min_ev=0.0)
    assert set(picks["fixture_id"]) == {"fx1"}
