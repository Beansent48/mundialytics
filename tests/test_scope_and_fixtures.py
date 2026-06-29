import pandas as pd

from mundialytics.data.schema import normalize_fixtures, normalize_matches


def test_normalize_matches_accepts_club_scope():
    df = pd.DataFrame([
        {
            "match_id": "m1",
            "date": "2025-01-01",
            "home_team": "Barcelona",
            "away_team": "Real Madrid",
            "home_goals": 2,
            "away_goals": 1,
            "neutral": 0,
            "team_scope": "club",
        }
    ])
    out = normalize_matches(df)
    assert out.loc[0, "team_scope"] == "club"


def test_normalize_fixtures_accepts_national_scope():
    df = pd.DataFrame([
        {
            "fixture_id": "f1",
            "date": "2026-06-21",
            "home_team": "Spain",
            "away_team": "Saudi Arabia",
            "neutral": 1,
            "team_scope": "national",
        }
    ])
    out = normalize_fixtures(df)
    assert out.loc[0, "team_scope"] == "national"
