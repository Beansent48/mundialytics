from __future__ import annotations

import pandas as pd

from mundialytics.data_quality.model_ready_snapshots import build_model_ready_match_snapshots


def _sample_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2024-08-01",
                "competition": "LaLiga",
                "season": "2024-2025",
                "stage": "league",
                "team_scope": "club",
                "home_team": "Alpha",
                "away_team": "Beta",
                "neutral": 0,
                "home_goals": 2,
                "away_goals": 1,
                "home_xg": 1.8,
                "away_xg": 0.7,
                "home_shots": 10,
                "away_shots": 6,
                "home_sot": 5,
                "away_sot": 2,
                "home_corners": 7,
                "away_corners": 3,
                "home_fouls": 8,
                "away_fouls": 11,
                "home_yellow_cards": 1,
                "away_yellow_cards": 2,
            },
            {
                "match_id": "m2",
                "date": "2024-08-08",
                "competition": "LaLiga",
                "season": "2024-2025",
                "stage": "league",
                "team_scope": "club",
                "home_team": "Gamma",
                "away_team": "Alpha",
                "neutral": 0,
                "home_goals": 0,
                "away_goals": 3,
                "home_xg": 0.5,
                "away_xg": 2.2,
                "home_shots": 4,
                "away_shots": 14,
                "home_sot": 1,
                "away_sot": 7,
                "home_corners": 2,
                "away_corners": 8,
                "home_fouls": 12,
                "away_fouls": 9,
                "home_yellow_cards": 3,
                "away_yellow_cards": 1,
            },
            {
                "match_id": "m3",
                "date": "2024-08-15",
                "competition": "LaLiga",
                "season": "2024-2025",
                "stage": "league",
                "team_scope": "club",
                "home_team": "Beta",
                "away_team": "Gamma",
                "neutral": 0,
                "home_goals": 1,
                "away_goals": 1,
                "home_xg": 1.1,
                "away_xg": 1.0,
                "home_shots": 9,
                "away_shots": 8,
                "home_sot": 3,
                "away_sot": 3,
                "home_corners": 4,
                "away_corners": 5,
                "home_fouls": 10,
                "away_fouls": 10,
                "home_yellow_cards": 2,
                "away_yellow_cards": 2,
            },
        ]
    )


def test_model_ready_snapshots_separate_pre_match_features_from_targets() -> None:
    outputs = build_model_ready_match_snapshots(_sample_matches(), dataset_name="test_big5_hybrid")
    snapshots = outputs.snapshots
    contract = outputs.feature_contract

    assert outputs.summary["version"] == "v0.49.6_enriched_hybrid_model_ready_snapshots"
    assert outputs.summary["hybrid_model_policy"] == "global_big5_model_with_league_features_and_league_level_calibration"
    assert outputs.summary["xg_features_available"] is True
    assert len(snapshots) == 3

    target_rows = contract[contract["role"] == "target"]
    assert "target_home_goals" in set(target_rows["column"])
    assert set(target_rows["leakage_status"]) == {"post_match_target_not_feature"}

    feature_rows = contract[contract["role"] == "feature"]
    assert "league_goal_rate_pre" in set(feature_rows["column"])
    assert "home_xg_for_last5" in set(feature_rows["column"])
    assert set(feature_rows["leakage_status"]) == {"pre_match_feature"}


def test_model_ready_snapshots_are_leakage_safe_for_rolling_and_league_rates() -> None:
    outputs = build_model_ready_match_snapshots(_sample_matches(), dataset_name="test_big5_hybrid")
    snapshots = outputs.snapshots.set_index("match_id")

    # m1 is the first league match, so no prior league goal-rate exists.
    assert pd.isna(snapshots.loc["m1", "league_goal_rate_pre"])

    # Before m2, the only prior league match was m1 with 3 total goals.
    assert float(snapshots.loc["m2", "league_goal_rate_pre"]) == 3.0

    # Alpha plays away in m2. Its prior rolling goals are from m1 only,
    # not from the current m2 result.
    assert float(snapshots.loc["m2", "away_goals_for_last5"]) == 2.0
    assert float(snapshots.loc["m2", "away_xg_for_last5"]) == 1.8

    # Beta plays home in m3. Its prior rolling goals are from m1 only.
    assert float(snapshots.loc["m3", "home_goals_for_last5"]) == 1.0
    assert float(snapshots.loc["m3", "home_xg_for_last5"]) == 0.7

    # Targets remain available for training/evaluation, but they are explicitly
    # marked as targets rather than pre-match features by the contract.
    assert float(snapshots.loc["m2", "target_away_goals"]) == 3.0
