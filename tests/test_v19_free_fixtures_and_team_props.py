from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from mundialytics.data.adapters.espn import scoreboard_response_to_df
from mundialytics.data.adapters.sofascore import lineups_response_to_df
from mundialytics.data.free_fixtures import filter_by_local_date, world_cup_filters, filter_contains
from mundialytics.features.team_match_stats import build_team_match_stats_from_player_events, predict_team_props_simple


def test_espn_world_cup_scoreboard_parser_contract():
    payload = {
        "leagues": [{"name": "FIFA World Cup"}],
        "events": [
            {
                "id": "401",
                "date": "2026-06-17T20:00Z",
                "season": {"year": 2026},
                "status": {"type": {"name": "STATUS_SCHEDULED", "description": "Scheduled"}},
                "competitions": [
                    {
                        "id": "401",
                        "competitors": [
                            {"homeAway": "home", "score": "0", "team": {"id": "1", "displayName": "Spain", "abbreviation": "ESP"}},
                            {"homeAway": "away", "score": "0", "team": {"id": "2", "displayName": "Uruguay", "abbreviation": "URU"}},
                        ],
                    }
                ],
            }
        ],
    }
    df = scoreboard_response_to_df(payload)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["provider"] == "espn"
    assert row["fixture_id"] == "401"
    assert row["home_team"] == "spain"
    assert row["away_team"] == "uruguay"
    assert "competition_context" in df.columns


def test_free_fixture_filters_world_cup_and_timezone():
    payload = {
        "leagues": [{"name": "FIFA World Cup"}],
        "events": [
            {"id": "1", "date": "2026-06-18T02:00Z", "status": {"type": {"name": "scheduled"}}, "competitions": [{"competitors": [{"homeAway":"home","team":{"displayName":"Mexico"}}, {"homeAway":"away","team":{"displayName":"South Korea"}}]}]},
            {"id": "2", "date": "2026-06-18T02:00Z", "status": {"type": {"name": "scheduled"}}, "competitions": [{"competitors": [{"homeAway":"home","team":{"displayName":"Club A"}}, {"homeAway":"away","team":{"displayName":"Club B"}}]}]},
        ],
    }
    df = scoreboard_response_to_df(payload)
    # Make second event look like club by changing competition text.
    df.loc[df["fixture_id"].astype(str).eq("2"), "competition"] = "Club World Cup"
    include, exclude = world_cup_filters()
    filtered, _ = filter_contains(df, include=include, exclude=exclude)
    out, warning = filter_by_local_date(filtered, local_date="2026-06-17", timezone="America/New_York")
    assert warning is None
    assert len(out) == 1
    assert out.iloc[0]["kickoff_local_date"] == "2026-06-17"


def test_sofascore_lineups_parser_preserves_provider_player_ids():
    payload = {
        "confirmed": True,
        "home": {"formation": "4-3-3", "players": [{"player": {"id": 10, "name": "Lamine Yamal"}, "position": "RW", "shirtNumber": 19, "substitute": False}]},
        "away": {"formation": "4-4-2", "players": [{"player": {"id": 8, "name": "Federico Valverde"}, "position": "CM", "substitute": False}]},
    }
    df = lineups_response_to_df(payload, fixture_row={"fixture_id": 999, "match_id": "sofascore:999", "home_team": "Spain", "away_team": "Uruguay", "competition": "FIFA World Cup"})
    assert len(df) == 2
    assert set(df["provider_player_id"].astype(str)) == {"10", "8"}
    assert set(df["team"]) == {"spain", "uruguay"}
    assert set(df["lineup_status"]) == {"official"}


def test_build_and_predict_team_match_stats_without_inventing_corners():
    pe = pd.DataFrame([
        {"match_id": "m1", "date": "2024-01-01", "competition": "FIFA World Cup", "team": "Spain", "opponent": "Uruguay", "player": "A", "shots": 2, "shots_on_target": 1, "fouls_committed": 1, "yellow_cards": 0, "goals": 1},
        {"match_id": "m1", "date": "2024-01-01", "competition": "FIFA World Cup", "team": "Uruguay", "opponent": "Spain", "player": "B", "shots": 1, "shots_on_target": 0, "fouls_committed": 2, "yellow_cards": 1, "goals": 0},
        {"match_id": "m2", "date": "2024-01-10", "competition": "FIFA World Cup", "team": "Spain", "opponent": "France", "player": "A", "shots": 3, "shots_on_target": 2, "fouls_committed": 0, "yellow_cards": 0, "goals": 0},
        {"match_id": "m2", "date": "2024-01-10", "competition": "FIFA World Cup", "team": "France", "opponent": "Spain", "player": "C", "shots": 4, "shots_on_target": 2, "fouls_committed": 1, "yellow_cards": 0, "goals": 2},
    ])
    stats = build_team_match_stats_from_player_events(pe)
    assert "shots_for" in stats.columns
    assert "corners_for" not in stats.columns
    fixtures = pd.DataFrame([{"fixture_id": "f1", "match_id": "f1", "date": "2024-02-01", "competition": "FIFA World Cup", "team_scope":"national", "home_team": "Spain", "away_team": "Uruguay", "neutral": 1}])
    preds = predict_team_props_simple(stats, fixtures)
    assert len(preds) == 2
    assert "expected_shots_for" in preds.columns
    assert preds["expected_shots_for"].notna().any()
    assert "expected_corners_for" in preds.columns
    assert preds["expected_corners_for"].isna().all()
