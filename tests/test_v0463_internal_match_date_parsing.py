import pandas as pd

from mundialytics.betting.historical_odds_backfill import normalize_internal_matches, parse_datetime_series_utc


def test_parse_datetime_series_accepts_yyyymmdd_ints():
    parsed = parse_datetime_series_utc(pd.Series([20260623, "2026-06-24", None]))
    assert parsed.dt.strftime("%Y-%m-%d").tolist()[:2] == ["2026-06-23", "2026-06-24"]


def test_normalize_internal_matches_handles_empty_kickoff_and_date_column():
    raw = pd.DataFrame({
        "match_id": ["m1", "m1", "m2"],
        "date": [20260623, 20260623, "2026-06-24"],
        "kickoff_utc": ["", None, ""],
        "home_team": ["Portugal", "Portugal", "England"],
        "away_team": ["Uzbekistan", "Uzbekistan", "Ghana"],
    })
    out = normalize_internal_matches(raw, min_date="2026-01-01")
    assert len(out) == 2
    assert out.loc[out["match_id"].eq("m1"), "date"].iloc[0] == "2026-06-23"
    assert out["kickoff_utc"].str.endswith("Z").all()


def test_normalize_internal_matches_raises_helpful_message_for_missing_columns():
    raw = pd.DataFrame({"foo": [1], "bar": [2]})
    try:
        normalize_internal_matches(raw)
    except ValueError as exc:
        assert "Detected columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
