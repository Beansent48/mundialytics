from __future__ import annotations

import pandas as pd

from mundialytics.data.adapters.oddspapi import (
    build_fixture_windows_from_model_lines,
    build_market_mapping_frame,
    fixtures_to_frame,
    flatten_oddspapi_odds_response,
    markets_to_frame,
    match_model_lines_to_provider_fixtures,
)


def test_markets_mapping_basic_soccer_1x2_and_totals():
    payload = [
        {
            "marketId": 101,
            "sportId": 10,
            "playerProp": False,
            "handicap": 0,
            "period": "fulltime",
            "marketType": "1x2",
            "marketName": "Full Time Result",
            "outcomes": [
                {"outcomeId": 101, "outcomeName": "1"},
                {"outcomeId": 102, "outcomeName": "X"},
                {"outcomeId": 103, "outcomeName": "2"},
            ],
        },
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
        },
    ]
    mapped = build_market_mapping_frame(markets_to_frame(payload))
    one = mapped[mapped["outcomeId"].eq(101)].iloc[0]
    over = mapped[mapped["outcomeId"].eq(1010)].iloc[0]
    assert one["internal_market_key"] == "1x2"
    assert one["internal_side"] == "home"
    assert over["internal_market_key"] == "goals"
    assert over["internal_side"] == "over"
    assert float(over["internal_line"]) == 2.5


def test_flatten_historical_odds_selects_closing_snapshot():
    markets = markets_to_frame([
        {
            "marketId": 1010,
            "sportId": 10,
            "playerProp": False,
            "handicap": 2.5,
            "period": "fulltime",
            "marketType": "totals",
            "marketName": "Over Under Full Time",
            "outcomes": [{"outcomeId": 1010, "outcomeName": "Over"}],
        }
    ])
    # kickoff = 2026-01-01T18:00:00Z = 1767290400 sec
    payload = {
        "fixtureId": "id100001234",
        "startTime": 1767290400,
        "status": {"live": False},
        "participants": {"participant1Name": "Spain", "participant2Name": "Germany"},
        "odds": {
            "pinnacle": {
                "id100001234:pinnacle:1010:0": {
                    "1767280000000": {"bookmaker": "pinnacle", "outcomeId": 1010, "playerId": 0, "price": 1.90, "changedAt": 1767280000000},
                    "1767290300000": {"bookmaker": "pinnacle", "outcomeId": 1010, "playerId": 0, "price": 1.85, "changedAt": 1767290300000},
                    "1767290500000": {"bookmaker": "pinnacle", "outcomeId": 1010, "playerId": 0, "price": 1.80, "changedAt": 1767290500000},
                }
            }
        },
    }
    out = flatten_oddspapi_odds_response(payload, markets_df=markets, snapshot_policy="closing")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["provider_event_id"] == "id100001234"
    assert row["market_key"] == "goals"
    assert row["side"] == "over"
    assert float(row["line"]) == 2.5
    assert float(row["bookmaker_odds"]) == 1.85


def test_fixture_windows_are_chunked_not_per_match():
    model_lines = pd.DataFrame({
        "match_id": ["m1", "m1", "m2", "m3"],
        "date": ["2026-06-01", "2026-06-01", "2026-06-03", "2026-06-10"],
        "home_team": ["A", "A", "C", "E"],
        "away_team": ["B", "B", "D", "F"],
    })
    windows = build_fixture_windows_from_model_lines(model_lines, chunk_days=7, pad_hours=0)
    assert len(windows) == 2
    assert int(windows["expected_matches"].sum()) == 3


def test_fuzzy_match_model_to_provider_fixture():
    model_lines = pd.DataFrame({
        "match_id": ["esp_ger"],
        "date": ["2026-06-01"],
        "home_team": ["Spain"],
        "away_team": ["Germany"],
    })
    fixtures = fixtures_to_frame([
        {
            "fixtureId": "id100001",
            "startTime": 1780272000,
            "sport": {"sportId": 10, "sportName": "Soccer"},
            "tournament": {"tournamentName": "World Cup"},
            "participants": {"participant1Name": "Spain", "participant2Name": "Germany"},
            "status": {"statusName": "Finished"},
        }
    ])
    candidates = match_model_lines_to_provider_fixtures(model_lines, fixtures)
    assert not candidates.empty
    assert candidates.iloc[0]["provider_fixture_id"] == "id100001"
    assert bool(candidates.iloc[0]["auto_match"]) is True
