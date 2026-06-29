from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from mundialytics.data_quality.match_dataset_foundation import prepare_match_dataset


def test_match_dataset_foundation_cleans_profiles_and_flags_anomalies() -> None:
    df = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2024-08-01",
                "home_team": "FC Barcelona",
                "away_team": "Valencia CF",
                "home_goals": 2,
                "away_goals": 1,
                "neutral": 0,
                "competition": "LaLiga",
                "season": "2024-2025",
                "team_scope": "club",
                "home_shots": 8,
                "away_shots": 7,
                "home_sot": 3,
                "away_sot": 2,
                "home_corners": 5,
                "away_corners": 4,
                "home_yellow_cards": 1,
                "away_yellow_cards": 2,
            },
            {
                "match_id": "m1",
                "date": "2024-08-01",
                "home_team": "FC Barcelona",
                "away_team": "Valencia CF",
                "home_goals": 2,
                "away_goals": 1,
                "neutral": 0,
                "competition": "LaLiga",
                "season": "2024-2025",
                "team_scope": "club",
            },
            {
                "match_id": "m2",
                "date": "2024-08-08",
                "home_team": "Real Madrid CF",
                "away_team": "Atl Madrid",
                "home_goals": 1,
                "away_goals": 1,
                "neutral": 0,
                "competition": "LaLiga",
                "season": "2024-2025",
                "team_scope": "club",
                "home_shots": 4,
                "away_shots": 10,
                "home_sot": 6,
                "away_sot": 3,
                "home_corners": 30,
                "away_corners": 2,
                "home_yellow_cards": 11,
                "away_yellow_cards": 1,
            },
            {
                "match_id": "m3",
                "date": "not-a-date",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_goals": 0,
                "away_goals": 0,
                "neutral": 0,
                "competition": "LaLiga",
                "season": "2024-2025",
                "team_scope": "club",
            },
        ]
    )

    outputs = prepare_match_dataset(df, dataset_name="unit_test", drop_incomplete_goals=True)

    assert len(outputs.cleaned_matches) == 2
    assert outputs.summary["input_rows"] == 4
    assert outputs.summary["output_rows"] == 2
    assert outputs.summary["dropped_rows"] == 2
    assert outputs.summary["status"] == "warning"
    assert "rows_dropped" in outputs.summary["warnings"]
    assert "anomalies_detected" in outputs.summary["warnings"]

    issues = set(outputs.anomalies["issue"])
    assert "shots_on_target_exceed_shots" in issues
    assert "extreme_corner_count" in issues
    assert "extreme_yellow_card_count" in issues

    coverage = outputs.feature_coverage.set_index("feature_group")
    assert coverage.loc["goals", "status"] == "available"
    assert coverage.loc["corners", "rows_with_full_group"] == 2


def test_build_match_dataset_script_writes_foundation_outputs(tmp_path: Path) -> None:
    raw = tmp_path / "2425_SP1.csv"
    raw.write_text(
        "\n".join(
            [
                "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HS,AS,HST,AST,HC,AC,HY,AY",
                "SP1,15/08/24,Barcelona,Valencia,2,1,10,7,5,2,6,3,1,2",
                "SP1,22/08/24,Real Madrid,Atl Madrid,1,1,12,8,4,3,5,4,2,3",
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "foundation"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_match_dataset.py",
            "--source",
            "football-data-uk",
            "--inputs",
            str(raw),
            "--out-dir",
            str(out_dir),
            "--drop-incomplete-goals",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    summary = json.loads(result.stdout)
    assert summary["status"] == "warning"
    assert summary["output_rows"] == 2
    assert "sparse_event_feature_coverage" in summary["warnings"]
    assert (out_dir / "canonical_matches.csv").exists()
    assert (out_dir / "match_dataset_feature_coverage.csv").exists()
    assert (out_dir / "match_dataset_quality_by_competition_season.csv").exists()
    assert (out_dir / "match_dataset_foundation_report.json").exists()
