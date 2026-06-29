import json
from pathlib import Path

import pandas as pd

from mundialytics.data.adapters import (
    statsbomb_events_to_player_events,
    statsbomb_events_to_lineups,
    statsbomb_open_data_match_metadata,
)
from mundialytics.data.event_quality import EventReadinessThresholds, diagnose_player_event_dataset
from mundialytics.data.events import merge_player_events_with_lineups

ROOT = Path(__file__).resolve().parents[1]


def test_event_quality_fails_without_real_prop_columns():
    df = pd.DataFrame({"match_id": ["m1"], "team": ["a"], "player": ["p"], "minutes": [90]})
    report = diagnose_player_event_dataset(
        df,
        thresholds=EventReadinessThresholds(min_matches=1, min_player_rows=1, min_total_events_per_market=1, min_minutes_coverage=0.5),
    )
    assert report["status"] == "EVENT_DATA_NOT_READY"
    assert any(not m["passed"] for m in report["market_checks"])


def test_event_quality_passes_on_real_statsbomb_sample_with_low_thresholds():
    fp = ROOT / "data/sample/event_json/statsbomb_sample_events.json"
    pe = statsbomb_events_to_player_events(fp, match_id="sb_1", team_scope="national", competition="sample")
    lu = statsbomb_events_to_lineups(fp, match_id="sb_1", team_scope="national", competition="sample")
    # Only require markets present in this tiny sample.
    report = diagnose_player_event_dataset(
        merge_player_events_with_lineups(pe, lu),
        lineups=lu,
        required_markets=["player_shots", "player_shots_on_target", "player_fouls_committed"],
        thresholds=EventReadinessThresholds(min_matches=1, min_player_rows=2, min_total_events_per_market=1, min_minutes_coverage=0.5),
    )
    assert report["status"] == "EVENT_DATA_READY"


def test_statsbomb_metadata_scanner(tmp_path):
    data = tmp_path / "data"
    (data / "matches" / "1").mkdir(parents=True)
    (data / "events").mkdir()
    (data / "competitions.json").write_text(json.dumps([{"competition_id": 1, "season_id": 99, "competition_name": "World Cup", "season_name": "2026"}]), encoding="utf-8")
    (data / "matches" / "1" / "99.json").write_text(json.dumps([{"match_id": 123, "match_date": "2026-06-15", "home_team": {"home_team_name": "Spain"}, "away_team": {"away_team_name": "Uruguay"}}]), encoding="utf-8")
    meta = statsbomb_open_data_match_metadata(data)
    assert meta["123"]["date"] == "2026-06-15"
    assert meta["123"]["competition"] == "World Cup"
