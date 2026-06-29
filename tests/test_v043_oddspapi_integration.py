from __future__ import annotations

import pandas as pd

from mundialytics.data.adapters.oddspapi import (
    OddsPapiClient,
    flatten_oddspapi_odds_response,
    markets_to_frame,
)
from scripts.build_training_odds_features import build_match_1x2_features, build_line_features


def test_rapidapi_client_defaults_to_rapidapi_proxy():
    client = OddsPapiClient(mode="rapidapi", rapidapi_key="k")
    assert client.base_url == "https://odds-api1.p.rapidapi.com"
    assert client.rapidapi_host == "odds-api1.p.rapidapi.com"
    headers = client._headers()
    assert headers["X-RapidAPI-Key"] == "k"
    assert headers["X-RapidAPI-Host"] == "odds-api1.p.rapidapi.com"


def test_flatten_v5_bookmakers_tree_historical_shape():
    markets = markets_to_frame([
        {
            "marketId": 1010,
            "sportId": 10,
            "playerProp": False,
            "handicap": 2.5,
            "period": "fulltime",
            "marketType": "totals",
            "marketName": "Over Under Full Time",
            "outcomes": [
                {"outcomeId": 1010, "outcomeName": "Over"},
                {"outcomeId": 1011, "outcomeName": "Under"},
            ],
        }
    ])
    payload = {
        "fixtureId": "id100001234",
        "startTime": 1767290400,
        "participants": {"participant1Name": "Spain", "participant2Name": "Germany"},
        "bookmakers": {
            "pinnacle": {
                "markets": {
                    "1010": {
                        "outcomes": {
                            "1010": {
                                "players": {
                                    "0": [
                                        {"price": 1.90, "createdAt": 1767280000000},
                                        {"price": 1.85, "createdAt": 1767290300000},
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        },
    }
    out = flatten_oddspapi_odds_response(payload, markets_df=markets, snapshot_policy="closing")
    assert len(out) == 1
    assert out.iloc[0]["market_key"] == "goals"
    assert out.iloc[0]["side"] == "over"
    assert float(out.iloc[0]["bookmaker_odds"]) == 1.85


def test_build_training_features_devigs_1x2_and_totals():
    odds = pd.DataFrame([
        {"match_id": "m1", "bookmaker": "pin", "market_key": "1x2", "side": "home", "bookmaker_odds": 2.0},
        {"match_id": "m1", "bookmaker": "pin", "market_key": "1x2", "side": "draw", "bookmaker_odds": 3.5},
        {"match_id": "m1", "bookmaker": "pin", "market_key": "1x2", "side": "away", "bookmaker_odds": 4.0},
        {"match_id": "m1", "bookmaker": "pin", "market_key": "goals", "scope": "match", "line": 2.5, "side": "over", "bookmaker_odds": 1.90},
        {"match_id": "m1", "bookmaker": "pin", "market_key": "goals", "scope": "match", "line": 2.5, "side": "under", "bookmaker_odds": 1.95},
    ])
    odds["implied_probability_raw"] = 1.0 / odds["bookmaker_odds"]
    one = build_match_1x2_features(odds)
    lines = build_line_features(odds)
    assert len(one) == 1
    assert round(float(one.iloc[0]["odds_p_home_devig"] + one.iloc[0]["odds_p_draw_devig"] + one.iloc[0]["odds_p_away_devig"]), 6) == 1.0
    assert len(lines) == 1
    assert round(float(lines.iloc[0]["odds_p_over_devig"] + lines.iloc[0]["odds_p_under_devig"]), 6) == 1.0
