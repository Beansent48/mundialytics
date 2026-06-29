from pathlib import Path

import pandas as pd

from mundialytics.data.extra_match_stats import (
    parse_statsbomb_goalkeeper_match_json,
    parse_api_football_fixture_player_stats_json,
    parse_football_data_csvs,
)
from mundialytics.statistical_core.event_line_backtest import (
    build_settled_event_line_signals,
    build_goalkeeper_save_line_signals,
)
from mundialytics.betting.pick_policy import standardize_settled_line_signals

ROOT = Path(__file__).resolve().parents[1]


def test_statsbomb_goalkeeper_parser_keeps_zero_save_rows():
    df = parse_statsbomb_goalkeeper_match_json([ROOT / "data/sample/extra_stats/statsbomb_goalkeeper_events_sample.json"])
    assert not df.empty
    assert set(df["goalkeeper"]) >= {"alpha keeper", "beta keeper"}
    alpha = df[df["goalkeeper"].eq("alpha keeper")].iloc[0]
    beta = df[df["goalkeeper"].eq("beta keeper")].iloc[0]
    assert alpha["saves"] == 1
    assert beta["saves"] == 0
    assert alpha["saves_data_quality_flag"] == "raw_event_goalkeeper_saves"


def test_api_football_player_stats_parser_gets_goalkeeper_saves():
    df = parse_api_football_fixture_player_stats_json([ROOT / "data/sample/extra_stats/api_football_fixture_players_sample.json"])
    assert not df.empty
    assert set(df["goalkeeper"]) == {"alpha keeper", "beta keeper"}
    assert float(df[df["goalkeeper"].eq("alpha keeper")]["saves"].iloc[0]) == 3.0
    assert "provider_player_goalkeeper_saves" in df["data_quality_flag"].iloc[0]


def test_relational_event_line_backtest_generates_corners_and_team_saves():
    stats = parse_football_data_csvs([ROOT / "data/sample/extra_stats/football_data_sample.csv"], derive_saves_from_sot=True)
    signals = build_settled_event_line_signals(stats, min_history=1)
    std = standardize_settled_line_signals(signals)
    assert not std.empty
    assert "team_corners_over" in set(std["signal_group"])
    assert "corners_under" in set(std["signal_group"])
    assert "team_goalkeeper_saves_over" in set(std["signal_group"])
    assert "expected_stat" in std.columns
    assert "expected_components" in std.columns


def test_player_goalkeeper_save_lines_are_player_scope():
    gk = parse_api_football_fixture_player_stats_json([ROOT / "data/sample/extra_stats/api_football_fixture_players_sample.json"])
    # duplicate dates to allow some rolling fallback still with min_history=1
    gk2 = pd.concat([gk, gk.assign(match_id="1002", saves=[2.0, 4.0])], ignore_index=True)
    signals = build_goalkeeper_save_line_signals(gk2, min_history=1)
    std = standardize_settled_line_signals(signals)
    assert not std.empty
    assert set(std["scope"]) == {"player"}
    assert "goalkeeper_saves_over" in set(std["signal_group"])
    assert "goalkeeper_saves_under" in set(std["signal_group"])
