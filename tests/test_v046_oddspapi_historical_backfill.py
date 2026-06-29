from __future__ import annotations

import pandas as pd

from mundialytics.betting.historical_odds_backfill import (
    normalize_internal_matches,
    build_fixture_request_windows,
    match_internal_to_provider,
    select_best_fixture_matches,
    build_snapshot_rows,
    filter_target_market_mapping,
)


def test_v046_build_fixture_windows_from_internal_matches():
    raw = pd.DataFrame({
        "match_id": ["m1", "m2"],
        "date": ["2026-01-05", "2026-01-06"],
        "home_team": ["Argentina", "France"],
        "away_team": ["Austria", "Iraq"],
    })
    matches = normalize_internal_matches(raw, min_date="2026-01-01")
    windows = build_fixture_request_windows(matches, chunk_hours=24, pad_hours=4)
    assert len(matches) == 2
    assert set(["from", "to", "startTimeFrom", "startTimeTo", "expected_matches"]).issubset(windows.columns)
    assert len(windows) >= 1


def test_v046_match_internal_to_provider_selects_direct_match():
    internal = pd.DataFrame({
        "match_id": ["m1"],
        "date": ["2026-01-05"],
        "kickoff_utc": ["2026-01-05T20:00:00Z"],
        "home_team": ["Argentina"],
        "away_team": ["Austria"],
    })
    provider = pd.DataFrame({
        "provider": ["oddspapi"],
        "provider_fixture_id": ["id100"],
        "kickoff_utc": ["2026-01-05T20:05:00Z"],
        "home_team": ["Argentina"],
        "away_team": ["Austria"],
    })
    candidates = match_internal_to_provider(internal, provider, auto_threshold=0.86)
    selected = select_best_fixture_matches(candidates, auto_threshold=0.86)
    assert len(selected) == 1
    assert selected.iloc[0]["provider_fixture_id"] == "id100"
    assert selected.iloc[0]["auto_match"] in [True, 1]


def test_v046_snapshot_rows_are_pre_kickoff_only():
    ticks = pd.DataFrame({
        "snapshot_time_utc": [
            "2026-01-05T18:50:00Z",
            "2026-01-05T19:05:00Z",
            "2026-01-05T20:01:00Z",  # post kickoff: must not be selected
        ],
        "bookmaker": ["pinnacle"] * 3,
        "provider": ["oddspapi"] * 3,
        "provider_event_id": ["id100"] * 3,
        "match_id": ["m1"] * 3,
        "market_key": ["goals"] * 3,
        "scope": ["match"] * 3,
        "subject_team": [""] * 3,
        "subject_player": [""] * 3,
        "line": [2.5] * 3,
        "side": ["over"] * 3,
        "bookmaker_odds": [1.90, 1.80, 1.50],
    })
    mapping = pd.DataFrame({
        "match_id": ["m1"],
        "provider_fixture_id": ["id100"],
        "provider_kickoff_utc": ["2026-01-05T20:00:00Z"],
    })
    out = build_snapshot_rows(ticks, mapping, snapshot_offsets={"t1h": 3600, "closing": 0})
    assert set(out["snapshot_label"]) == {"t1h", "closing"}
    closing = out[out["snapshot_label"].eq("closing")].iloc[0]
    assert closing["bookmaker_odds"] == 1.80
    assert closing["snapshot_time_utc"] == "2026-01-05T19:05:00Z"


def test_v046_market_mapping_target_filter():
    mapping = pd.DataFrame({
        "internal_market_key": ["goals", "player_shots", "random_market"],
        "mapping_confidence": ["high", "medium", "needs_manual_review"],
    })
    out = filter_target_market_mapping(mapping)
    assert set(out["internal_market_key"]) == {"goals", "player_shots"}
