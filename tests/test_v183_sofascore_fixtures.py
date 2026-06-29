from __future__ import annotations

from datetime import datetime, timezone

from mundialytics.data.adapters.sofascore import scheduled_events_response_to_df
from scripts.fetch_sofascore_fixtures import _filter_contains, _filter_df_by_local_date


def ts(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def sample_payload():
    return {
        "events": [
            {
                "id": 111,
                "slug": "spain-uruguay",
                "startTimestamp": ts(datetime(2026, 6, 17, 20, 0)),
                "tournament": {"id": 1, "name": "World Cup", "category": {"name": "World"}},
                "uniqueTournament": {"id": 16, "name": "FIFA World Cup", "category": {"name": "World"}},
                "season": {"name": "World Cup 2026", "year": "2026"},
                "roundInfo": {"name": "Group H"},
                "homeTeam": {"id": 100, "name": "Spain", "slug": "spain"},
                "awayTeam": {"id": 200, "name": "Uruguay", "slug": "uruguay"},
                "status": {"type": "notstarted", "description": "Not started"},
            },
            {
                "id": 222,
                "startTimestamp": ts(datetime(2026, 6, 17, 21, 0)),
                "tournament": {"id": 2, "name": "Club World Cup", "category": {"name": "World"}},
                "uniqueTournament": {"id": 22, "name": "Club World Cup"},
                "homeTeam": {"id": 300, "name": "Club A"},
                "awayTeam": {"id": 400, "name": "Club B"},
                "status": {"type": "notstarted", "description": "Not started"},
            },
        ]
    }


def test_sofascore_scheduled_events_to_df_contract():
    df = scheduled_events_response_to_df(sample_payload())
    assert len(df) == 2
    row = df[df["fixture_id"].astype(str).eq("111")].iloc[0]
    assert row["provider"] == "sofascore"
    assert row["match_id"] == "sofascore:111"
    assert row["home_team"] == "spain"
    assert row["away_team"] == "uruguay"
    assert "competition_context" in df.columns


def test_world_cup_filter_excludes_club_world_cup():
    df = scheduled_events_response_to_df(sample_payload())
    out, report = _filter_contains(
        df,
        include=["world cup", "world championship"],
        exclude=["club", "women", "u20", "qualification"],
    )
    assert len(out) == 1
    assert out.iloc[0]["fixture_id"] == 111
    assert report["exclude_removed_rows"] == 1


def test_local_date_filter_uses_requested_timezone():
    df = scheduled_events_response_to_df(sample_payload())
    out, warning = _filter_df_by_local_date(df, local_date="2026-06-17", timezone="America/New_York")
    assert len(out) == 2
    assert warning is None
    assert "kickoff_local_time" in out.columns
