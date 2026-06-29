from __future__ import annotations

import pandas as pd

from scripts.fetch_api_football_fixtures import (
    API_FOOTBALL_WORLD_CUP_LEAGUE_ID,
    API_FOOTBALL_WORLD_CUP_SEASON,
    _filter_df_by_local_date,
)
from mundialytics.data.adapters.api_football import fixtures_response_to_df


def test_world_cup_constants_match_api_football_2026_shortcut():
    assert API_FOOTBALL_WORLD_CUP_LEAGUE_ID == "1"
    assert API_FOOTBALL_WORLD_CUP_SEASON == "2026"


def test_local_date_post_filter_keeps_only_requested_american_day():
    df = pd.DataFrame({
        "fixture_id": [1, 2],
        "timestamp": [1782518400, 1782604800],  # 2026-06-26 20:00 ET, 2026-06-27 20:00 ET approximately
        "date": ["2026-06-26T20:00:00-04:00", "2026-06-27T20:00:00-04:00"],
        "competition": ["FIFA World Cup", "FIFA World Cup"],
    })
    out, warning = _filter_df_by_local_date(df, local_date="2026-06-26", timezone="America/New_York")
    assert len(out) == 1
    assert int(out.iloc[0]["fixture_id"]) == 1
    assert warning == "post_filtered_by_local_date_removed_rows=1"
    assert out.iloc[0]["kickoff_local_date"] == "2026-06-26"


def test_world_cup_fixture_parser_keeps_league_id_and_competition_context():
    payload = {
        "response": [
            {
                "fixture": {
                    "id": 66457002,
                    "date": "2026-06-26T20:00:00-04:00",
                    "timezone": "America/New_York",
                    "timestamp": 1782518400,
                    "venue": {"id": 1, "name": "Example"},
                    "status": {"short": "NS", "long": "Not Started"},
                },
                "league": {"id": 1, "name": "FIFA World Cup", "season": 2026, "round": "Group H"},
                "teams": {"home": {"id": 1, "name": "Uruguay"}, "away": {"id": 2, "name": "Spain"}},
                "goals": {"home": None, "away": None},
                "score": {"penalty": {"home": None, "away": None}},
            }
        ]
    }
    df = fixtures_response_to_df(payload)
    assert df.iloc[0]["league_id"] == 1
    assert df.iloc[0]["competition"] == "FIFA World Cup"
    assert df.iloc[0]["team_type"] == "national_team"
    assert df.iloc[0]["competition_context"] == "international_national_tournament"
