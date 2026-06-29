from __future__ import annotations

from mundialytics.data.adapters.creativesdev import normalize_events, normalize_fixtures, normalize_lineups, normalize_match_statistics


def test_creativesdev_normalize_fixtures_nested_payload():
    payload = {
        "response": [
            {
                "fixture": {"id": 123, "date": "2026-06-23T20:00:00+00:00", "status": {"long": "Scheduled"}},
                "teams": {"home": {"name": "Portugal"}, "away": {"name": "Uzbekistan"}},
                "league": {"name": "FIFA World Cup"},
            }
        ]
    }
    df = normalize_fixtures(payload)
    assert len(df) == 1
    assert df.loc[0, "provider_fixture_id"] == 123
    assert df.loc[0, "home_team"] == "Portugal"
    assert df.loc[0, "away_team"] == "Uzbekistan"


def test_creativesdev_normalizers_are_defensive():
    lineups = normalize_lineups({"response": [{"team": {"name": "Portugal"}, "players": [{"player": {"id": 7, "name": "Cristiano"}, "position": "FW"}]}]})
    stats = normalize_match_statistics({"response": [{"team": {"name": "Portugal"}, "statistics": [{"type": "Shots on Goal", "value": 5}]}]})
    events = normalize_events({"response": [{"time": {"elapsed": 22}, "team": {"name": "Portugal"}, "player": {"name": "Player"}, "type": "Goal"}]})
    assert lineups.loc[0, "player"] == "Cristiano"
    assert stats.loc[0, "stat_name"] == "Shots on Goal"
    assert events.loc[0, "event_type"] == "Goal"


def test_creativesdev_fixtures_by_date_reads_status_utctime_and_score():
    payload = {
        "data": [
            {
                "id": 4621624,
                "home": {"name": "Feyenoord"},
                "away": {"name": "Salzburg"},
                "leagueId": 42,
                "status": {
                    "utcTime": "2024-11-06T20:00:00.000Z",
                    "scoreStr": "1 - 3",
                    "reason": {"short": "FT", "long": "Full-Time"},
                },
                "tournamentStage": "League Stage",
            }
        ]
    }
    df = normalize_fixtures(payload)
    assert len(df) == 1
    assert df.loc[0, "kickoff_utc"] == "2024-11-06T20:00:00Z"
    assert df.loc[0, "home_score"] == "1"
    assert df.loc[0, "away_score"] == "3"
    assert df.loc[0, "status"] == "Full-Time"
    assert df.loc[0, "status_short"] == "FT"


def test_creativesdev_fixture_payload_does_not_create_fake_events_or_empty_csv_columns():
    payload = {"data": [{"id": 1, "home": {"name": "A"}, "away": {"name": "B"}, "status": {"utcTime": "2026-01-01T12:00:00.000Z"}}]}
    events = normalize_events(payload)
    lineups = normalize_lineups(payload)
    stats = normalize_match_statistics(payload)
    assert len(events) == 0
    assert list(events.columns) == ["match_id", "provider", "minute", "team", "player", "event_type", "detail"]
    assert len(lineups) == 0
    assert "player" in lineups.columns
    assert len(stats) == 0
    assert "stat_name" in stats.columns
