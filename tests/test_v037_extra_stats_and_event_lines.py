from pathlib import Path
import json

import pandas as pd

from mundialytics.data.extra_match_stats import (
    parse_football_data_csvs,
    parse_api_football_fixture_stats_json,
    parse_statsbomb_event_json,
)
from mundialytics.statistical_core.event_line_backtest import build_settled_event_line_signals
from mundialytics.betting.pick_policy import standardize_settled_line_signals, evaluate_policy_grid

ROOT = Path(__file__).resolve().parents[1]


def test_football_data_import_creates_corner_targets():
    df = parse_football_data_csvs([ROOT / "data/sample/extra_stats/football_data_sample.csv"])
    assert not df.empty
    assert {"corners_for", "corners_against", "shots_for", "shots_on_target_for"}.issubset(df.columns)
    assert df["corners_for"].notna().sum() == len(df)
    alpha = df[(df["team"].eq("alpha fc")) & (df["opponent"].eq("beta united"))].iloc[0]
    assert alpha["corners_for"] == 7
    assert alpha["corners_against"] == 4


def test_provider_fixture_stats_import_creates_corners_and_saves():
    df = parse_api_football_fixture_stats_json([ROOT / "data/sample/extra_stats/api_football_fixture_stats_sample.json"])
    assert len(df) == 2
    assert df["corners_for"].notna().all()
    assert df["saves_for"].notna().all()
    alpha = df[df["team"].eq("alpha fc")].iloc[0]
    assert alpha["corners_for"] == 8
    assert alpha["saves_for"] == 2
    assert alpha["saves_against"] == 5


def test_statsbomb_raw_extra_stats_counts_event_based_corners_and_saves():
    df = parse_statsbomb_event_json([ROOT / "data/sample/extra_stats/statsbomb_extra_events_sample.json"])
    assert len(df) == 2
    alpha = df[df["team"].eq("alpha fc")].iloc[0]
    beta = df[df["team"].eq("beta united")].iloc[0]
    assert alpha["corners_for"] == 2
    assert beta["corners_for"] == 1
    assert alpha["saves_for"] == 1


def test_event_line_backtest_generates_over_and_under_for_corners_shots_cards():
    stats = parse_football_data_csvs([ROOT / "data/sample/extra_stats/football_data_sample.csv"])
    signals = build_settled_event_line_signals(stats, min_history=1)
    standardized = standardize_settled_line_signals(signals)
    assert not standardized.empty
    groups = set(standardized["signal_group"])
    assert "team_corners_over" in groups
    assert "team_corners_under" in groups
    assert "corners_over" in groups
    assert "corners_under" in groups
    assert "team_shots_over" in groups
    assert "yellow_cards_under" in groups
    assert standardized["actual_win"].isin([0, 1]).all()


def test_pick_policy_can_learn_corner_side_when_line_signals_exist():
    rows = []
    for i in range(60):
        rows.append({
            "match_id": f"m{i}",
            "date": f"2024-03-{(i % 28) + 1:02d}",
            "market": "corners",
            "scope": "match",
            "selection": "under",
            "line": 9.5,
            "model_probability": 0.72,
            "settled_stat": 8 if i % 4 else 12,
        })
    signals = standardize_settled_line_signals(pd.DataFrame(rows))
    leaderboard, best = evaluate_policy_grid(signals, min_picks=5, require_odds=False)
    assert not leaderboard.empty
    assert "corners_under" in set(leaderboard["allowed_signal_group"])


def test_football_data_can_explicitly_derive_saves_from_sot_minus_goals():
    df = parse_football_data_csvs([ROOT / "data/sample/extra_stats/football_data_sample.csv"], derive_saves_from_sot=True)
    assert df["saves_for"].notna().all()
    alpha = df[(df["team"].eq("alpha fc")) & (df["opponent"].eq("beta united"))].iloc[0]
    # Beta had 3 shots on target and scored 1, so Alpha keeper saves ~= 2.
    assert alpha["saves_for"] == 2
    assert alpha["saves_data_quality_flag"] == "derived_saves_from_sot_minus_goals"
    assert "goalkeeper_saves_derived_sot_minus_goals" in alpha["data_quality_flag"]


def test_event_line_backtest_includes_goalkeeper_saves_when_derived_saves_exist():
    stats = parse_football_data_csvs([ROOT / "data/sample/extra_stats/football_data_sample.csv"], derive_saves_from_sot=True)
    signals = build_settled_event_line_signals(stats, min_history=1)
    standardized = standardize_settled_line_signals(signals)
    groups = set(standardized["signal_group"])
    assert "team_goalkeeper_saves_over" in groups
    assert "team_goalkeeper_saves_under" in groups
    save_rows = standardized[standardized["market"].eq("goalkeeper_saves")]
    assert not save_rows.empty
    assert "saves_data_quality_flag" in save_rows.columns
