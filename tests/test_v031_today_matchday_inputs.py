from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from mundialytics.data.adapters.sofascore import scheduled_events_response_to_df
from mundialytics.matchday.today_builder import build_matchday_fixtures, write_matchday_inputs


def ts(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def test_build_matchday_fixtures_filters_completed_and_writes_run_ready_schema(tmp_path):
    payload = {
        "events": [
            {
                "id": 111,
                "startTimestamp": ts(datetime(2026, 6, 21, 10, 0)),
                "tournament": {"id": 1, "name": "World Cup", "category": {"name": "World"}},
                "uniqueTournament": {"id": 16, "name": "FIFA World Cup", "category": {"name": "World"}},
                "season": {"name": "World Cup 2026", "year": "2026"},
                "roundInfo": {"name": "Group H"},
                "homeTeam": {"id": 100, "name": "Spain"},
                "awayTeam": {"id": 200, "name": "Saudi Arabia"},
                "status": {"type": "notstarted", "description": "Not started"},
            },
            {
                "id": 222,
                "startTimestamp": ts(datetime(2026, 6, 21, 12, 0)),
                "tournament": {"id": 1, "name": "World Cup", "category": {"name": "World"}},
                "uniqueTournament": {"id": 16, "name": "FIFA World Cup", "category": {"name": "World"}},
                "homeTeam": {"id": 101, "name": "Uruguay"},
                "awayTeam": {"id": 201, "name": "Cape Verde"},
                "status": {"type": "finished", "description": "Finished"},
            },
        ]
    }
    provider = scheduled_events_response_to_df(payload)
    fixtures, report = build_matchday_fixtures(provider, local_date="2026-06-21", timezone="UTC")
    assert report["rows_final"] == 1
    assert len(fixtures) == 1
    row = fixtures.iloc[0]
    assert row["match_id"] == "sofascore:111"
    assert row["home_team"] == "spain"
    assert row["away_team"] == "saudi arabia"
    assert row["team_type"] == "national_team"
    assert row["competition_context"] == "international_national_tournament"
    assert row["stage"] == "Group"
    assert row["group"] == "H"

    result = write_matchday_inputs(provider, out_dir=tmp_path, local_date="2026-06-21", timezone="UTC")
    assert result["fixtures_rows"] == 1
    assert (tmp_path / "today_fixtures.csv").exists()
    assert (tmp_path / "today_matchday_audit.json").exists()
    assert (tmp_path / "today_current_lineups.csv").exists()


def test_dynamic_lines_accept_timezone_aware_fixture_dates() -> None:
    import pandas as pd
    from mundialytics.statistical_core.dynamic_lines import build_dynamic_market_lines

    fixtures = pd.DataFrame([
        {"match_id": "TZ_FIX", "date": "2026-06-20T20:00:00+00:00", "home_team": "Spain", "away_team": "Saudi Arabia"}
    ])
    match_predictions = pd.DataFrame([{"match_id": "TZ_FIX", "lambda_home": 2.0, "lambda_away": 1.0}])
    team_stats = pd.DataFrame([
        {"match_id": "TZ_FIX", "team": "Spain", "market": "shots", "expected_count": 12.0, "availability": "available"},
        {"match_id": "TZ_FIX", "team": "Saudi Arabia", "market": "shots", "expected_count": 8.0, "availability": "available"},
        {"match_id": "TZ_FIX", "team": "match_total", "market": "total_shots", "expected_count": 20.0, "availability": "available"},
    ])
    historical_events = pd.DataFrame([
        {"match_id": "H1", "date": "2024-01-01", "team": "Spain", "opponent": "Italy", "player": "A", "type_name": "Shot"},
        {"match_id": "H2", "date": "2024-01-02", "team": "Saudi Arabia", "opponent": "Japan", "player": "B", "type_name": "Shot"},
    ])

    lines = build_dynamic_market_lines(
        fixtures,
        match_predictions,
        pd.DataFrame(),
        team_stats,
        pd.DataFrame(),
        historical_events,
    )

    assert not lines.empty
    assert "goals" in set(lines["market"])
