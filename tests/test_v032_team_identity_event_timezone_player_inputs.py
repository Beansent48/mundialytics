from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from mundialytics.data.adapters.espn import scoreboard_response_to_df, summary_response_to_lineups_df, team_payload_to_squad_df
from mundialytics.data.adapters.sofascore import scheduled_events_response_to_df, lineups_response_to_df
from mundialytics.data.free_fixtures import filter_by_matchday_date
from mundialytics.matchday.today_builder import build_matchday_fixtures, write_matchday_inputs


def _ts(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def test_v032_ivory_coast_maps_to_cote_d_ivoire_for_historical_identity() -> None:
    payload = {
        "events": [
            {
                "id": 760448,
                "date": "2026-06-20T20:00Z",
                "season": {"year": 2026},
                "status": {"type": {"name": "STATUS_SCHEDULED", "description": "Scheduled"}},
                "competitions": [
                    {
                        "id": "760448",
                        "venue": {"fullName": "MetLife Stadium", "address": {"city": "East Rutherford", "country": "United States"}},
                        "competitors": [
                            {"homeAway": "home", "team": {"id": "481", "displayName": "Germany", "shortDisplayName": "GER"}},
                            {"homeAway": "away", "team": {"id": "214", "displayName": "Ivory Coast", "shortDisplayName": "CIV"}},
                        ],
                    }
                ],
                "leagueName": "FIFA World Cup",
            }
        ],
        "leagues": [{"name": "FIFA World Cup"}],
    }
    provider = scoreboard_response_to_df(payload)
    fixtures, report = build_matchday_fixtures(provider, local_date="2026-06-20", timezone="Europe/Madrid", date_mode="event_or_user")
    assert report["rows_final"] == 1
    row = fixtures.iloc[0]
    assert row["home_team"] == "germany"
    assert row["away_team"] == "cote d ivoire"
    assert row["event_timezone"] == "America/New_York"


def test_v032_event_or_user_date_includes_us_evening_game_that_is_next_day_in_spain() -> None:
    payload = {
        "events": [
            {
                "id": 123,
                "date": "2026-06-21T01:00Z",  # 21 June in Madrid, 20 June event-local New York.
                "status": {"type": {"name": "STATUS_SCHEDULED", "description": "Scheduled"}},
                "competitions": [
                    {
                        "id": "123",
                        "venue": {"fullName": "MetLife Stadium", "address": {"city": "East Rutherford", "country": "United States"}},
                        "competitors": [
                            {"homeAway": "home", "team": {"id": "1", "displayName": "USA"}},
                            {"homeAway": "away", "team": {"id": "2", "displayName": "Mexico"}},
                        ],
                    }
                ],
            }
        ],
        "leagues": [{"name": "FIFA World Cup"}],
    }
    provider = scoreboard_response_to_df(payload)
    user_only, _ = filter_by_matchday_date(provider, local_date="2026-06-20", timezone="Europe/Madrid", date_mode="user")
    event_or_user, _ = filter_by_matchday_date(provider, local_date="2026-06-20", timezone="Europe/Madrid", date_mode="event_or_user")
    assert user_only.empty
    assert len(event_or_user) == 1
    assert event_or_user.iloc[0]["kickoff_event_date"] == "2026-06-20"
    assert event_or_user.iloc[0]["kickoff_user_date"] == "2026-06-21"


def test_v032_sofascore_lineups_parser_writes_current_lineups(tmp_path) -> None:
    event = {
        "events": [
            {
                "id": 111,
                "startTimestamp": _ts(datetime(2026, 6, 20, 20, 0)),
                "tournament": {"name": "World Cup", "category": {"name": "World"}},
                "uniqueTournament": {"name": "FIFA World Cup", "category": {"name": "World"}},
                "homeTeam": {"id": 1, "name": "Spain"},
                "awayTeam": {"id": 2, "name": "Côte d'Ivoire"},
                "status": {"type": "notstarted", "description": "Not started"},
            }
        ]
    }
    provider = scheduled_events_response_to_df(event)
    fixtures, _ = build_matchday_fixtures(provider, local_date="2026-06-20", timezone="UTC", date_mode="event_or_user")
    lineup_payload = {
        "confirmed": True,
        "home": {"formation": "4-3-3", "players": [{"player": {"id": 10, "name": "Lamine Yamal"}, "position": "RW", "shirtNumber": 19, "substitute": False}]},
        "away": {"formation": "4-2-3-1", "players": [{"player": {"id": 20, "name": "Simon Adingra"}, "position": "LW", "shirtNumber": 11, "substitute": False}]},
    }
    lineups = lineups_response_to_df(lineup_payload, fixture_row=fixtures.iloc[0].to_dict(), provider_match_id=111)
    result = write_matchday_inputs(provider, out_dir=tmp_path, local_date="2026-06-20", timezone="UTC", lineups_df=lineups)
    saved = pd.read_csv(result["lineups_csv"])
    assert len(saved) == 2
    assert set(saved["team"]) == {"spain", "cote d ivoire"}
    assert set(saved["started"]) == {1}
    assert result["player_input_status"] == "lineups_or_squads_written"


def test_v032_espn_summary_and_roster_parsers_are_defensive() -> None:
    fixture_row = {"match_id": "espn:1", "provider_match_id": "1", "home_team": "Germany", "away_team": "Ivory Coast"}
    summary_payload = {
        "boxscore": {
            "players": [
                {
                    "homeAway": "home",
                    "team": {"displayName": "Germany"},
                    "statistics": [{"athletes": [{"athlete": {"id": "9", "displayName": "Kai Havertz", "position": {"abbreviation": "ST"}}, "starter": True}]}],
                }
            ]
        }
    }
    lineups = summary_response_to_lineups_df(summary_payload, fixture_row=fixture_row, provider_match_id="1")
    assert len(lineups) == 1
    assert lineups.iloc[0]["team"] == "germany"
    assert lineups.iloc[0]["player"] == "kai havertz"

    roster_payload = {"team": {"athletes": [{"id": "10", "displayName": "Serge Gnabry", "position": {"abbreviation": "RW"}}]}}
    squads = team_payload_to_squad_df(roster_payload, team_name="Germany", fixture_row=fixture_row, provider_match_id="1")
    assert len(squads) == 1
    assert squads.iloc[0]["status"] == "current_squad_unconfirmed"


def test_v0321_espn_status_mapping_excludes_completed_but_keeps_live() -> None:
    payload = {
        "events": [
            {
                "id": 1,
                "date": "2026-06-20T00:30Z",
                "status": {"type": {"name": "STATUS_FULL_TIME", "description": "Full Time"}},
                "competitions": [{"id": "1", "venue": {"fullName": "MetLife Stadium", "address": {"city": "East Rutherford", "country": "United States"}}, "competitors": [
                    {"homeAway": "home", "team": {"id": "1", "displayName": "Brazil"}},
                    {"homeAway": "away", "team": {"id": "2", "displayName": "Haiti"}},
                ]}],
            },
            {
                "id": 2,
                "date": "2026-06-20T17:00Z",
                "status": {"type": {"name": "STATUS_SECOND_HALF", "description": "Second Half"}},
                "competitions": [{"id": "2", "venue": {"fullName": "NRG Stadium", "address": {"city": "Houston", "country": "United States"}}, "competitors": [
                    {"homeAway": "home", "team": {"id": "3", "displayName": "Netherlands"}},
                    {"homeAway": "away", "team": {"id": "4", "displayName": "Sweden"}},
                ]}],
            },
            {
                "id": 3,
                "date": "2026-06-20T20:00Z",
                "status": {"type": {"name": "STATUS_SCHEDULED", "description": "Scheduled"}},
                "competitions": [{"id": "3", "venue": {"fullName": "BMO Field", "address": {"city": "Toronto", "country": "Canada"}}, "competitors": [
                    {"homeAway": "home", "team": {"id": "5", "displayName": "Germany"}},
                    {"homeAway": "away", "team": {"id": "6", "displayName": "Ivory Coast"}},
                ]}],
            },
        ],
        "leagues": [{"name": "FIFA World Cup"}],
    }
    provider = scoreboard_response_to_df(payload)
    fixtures, report = build_matchday_fixtures(
        provider,
        local_date="2026-06-20",
        timezone="Europe/Madrid",
        date_mode="event_or_user",
        include_live=True,
        include_completed=False,
        include_unknown_status=True,
    )
    assert report["rows_removed_by_status"] == 1
    assert report["status_counts"] == {"live": 1, "scheduled": 1}
    assert set(fixtures["status_bucket"]) == {"live", "scheduled"}
    assert set(fixtures["match_id"].astype(str)) == {"espn:2", "espn:3"}


def test_v0322_espn_roster_positions_are_compact_strings() -> None:
    fixture_row = {"match_id": "espn:1", "provider_match_id": "1", "home_team": "Netherlands", "away_team": "Sweden"}
    roster_payload = {
        "team": {
            "athletes": [
                {"id": "1", "displayName": "Bart Verbruggen", "position": {"id": "1", "name": "Goalkeeper", "displayName": "Goalkeeper", "abbreviation": "G"}},
                {"id": "2", "displayName": "Cody Gakpo", "position": {"id": "3", "name": "Forward", "displayName": "Forward", "abbreviation": "F"}},
            ]
        }
    }
    squads = team_payload_to_squad_df(roster_payload, team_name="Netherlands", fixture_row=fixture_row, provider_match_id="1")
    assert set(squads["position"].astype(str)) == {"G", "F"}


def test_v0322_position_key_handles_provider_dict_strings() -> None:
    from mundialytics.statistical_core.player_event_model import _position_key, _position_group

    provider_position = "{'id': '1', 'name': 'Goalkeeper', 'displayName': 'Goalkeeper', 'abbreviation': 'G', 'leaf': False}"
    assert _position_key(provider_position) == "gk"
    assert _position_group(provider_position) == "goalkeeper"
    assert _position_group("F") == "forward"
    assert _position_group("M") == "central_midfield"


def test_v0322_dynamic_player_lines_hide_unresolved_zero_sample_props() -> None:
    from mundialytics.statistical_core.dynamic_lines import build_dynamic_market_lines
    from mundialytics.statistical_core.match_model import MatchOutcomeModel
    from mundialytics.statistical_core.team_stats_model import TeamStatsModel

    fixtures = pd.DataFrame([{"match_id":"m1", "date":"2026-06-20", "home_team":"Netherlands", "away_team":"Sweden", "competition":"FIFA World Cup", "team_type":"national_team", "competition_context":"international_national_tournament", "gender":"men"}])
    events = pd.DataFrame([
        {"match_id":"h1", "date":"2024-01-01", "team":"Netherlands", "opponent":"Sweden", "player":"Known Forward", "position":"Forward", "minutes":90, "shots":2, "shots_on_target":1, "fouls_committed":1, "yellow_cards":0, "goals":1},
        {"match_id":"h1", "date":"2024-01-01", "team":"Sweden", "opponent":"Netherlands", "player":"Swede", "position":"Forward", "minutes":90, "shots":1, "shots_on_target":0, "fouls_committed":1, "yellow_cards":0, "goals":1},
    ])
    mp, scores = MatchOutcomeModel().fit(events).predict_fixtures(fixtures)
    ts = TeamStatsModel().fit(events).predict_fixtures(fixtures, mp)
    props = pd.DataFrame([
        {"match_id":"m1", "team":"Netherlands", "opponent":"Sweden", "player":"Unknown Squad GK", "market":"player_shots", "expected_count":0.4, "position_group":"goalkeeper", "expected_minutes":35, "sample_size_minutes":0, "identity_status":"unresolved", "identity_match_level":"unresolved", "candidate_source":"squads", "warnings":"sample_size_zero_no_player_pick"},
    ])
    lines = build_dynamic_market_lines(fixtures, mp, scores, ts, props, events)
    rows = lines[(lines["scope"].eq("player")) & (lines["market"].eq("player_shots"))]
    assert not rows.empty
    assert set(rows["availability"].unique()) == {"not_available"}
    assert rows["reason_code"].astype(str).str.contains("identity_unresolved|sample_size_zero|role_guardrail").any()
