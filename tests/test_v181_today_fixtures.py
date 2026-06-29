from __future__ import annotations

import pandas as pd

from scripts.fetch_api_football_fixtures import _format_fixture_table
from mundialytics.data.adapters.api_football import fixtures_response_to_df


def test_api_football_fixtures_parser_keeps_fixture_id_and_timezone():
    payload = {
        "response": [
            {
                "fixture": {
                    "id": 123456,
                    "date": "2026-06-26T20:00:00-04:00",
                    "timezone": "America/New_York",
                    "timestamp": 1782518400,
                    "venue": {"id": 7, "name": "MetLife Stadium"},
                    "status": {"short": "NS", "long": "Not Started"},
                },
                "league": {"id": 1, "name": "FIFA World Cup", "season": 2026, "round": "Group H"},
                "teams": {"home": {"id": 9, "name": "Uruguay"}, "away": {"id": 10, "name": "Spain"}},
                "goals": {"home": None, "away": None},
                "score": {"penalty": {"home": None, "away": None}},
            }
        ]
    }
    df = fixtures_response_to_df(payload)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["fixture_id"] == 123456
    assert row["provider_match_id"] == 123456
    assert row["match_id"] == "api_football:123456"
    assert row["fixture_timezone"] == "America/New_York"
    assert row["competition"] == "FIFA World Cup"
    assert row["team_type"] == "national_team"


def test_format_fixture_table_includes_ids_and_teams():
    df = pd.DataFrame({
        "fixture_id": [123456],
        "date": ["2026-06-26T20:00:00-04:00"],
        "competition": ["FIFA World Cup"],
        "home_team": ["uruguay"],
        "away_team": ["spain"],
        "status_short": ["NS"],
        "status_long": ["Not Started"],
    })
    text = _format_fixture_table(df)
    assert "123456" in text
    assert "uruguay" in text
    assert "spain" in text
    assert "FIFA World Cup" in text
