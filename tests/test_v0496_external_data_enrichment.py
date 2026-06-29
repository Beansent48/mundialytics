from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from mundialytics.data_quality.model_ready_snapshots import build_model_ready_match_snapshots
from mundialytics.data_quality.team_registry import build_team_registry
from mundialytics.enrichment.clubelo import download_clubelo_team_histories, enrich_matches_with_clubelo
from mundialytics.enrichment.xg import enrich_matches_with_xg


def _sample_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2024-08-01",
                "competition": "Premier League",
                "season": "2024-2025",
                "stage": "league",
                "team_scope": "club",
                "home_team": "Man City",
                "away_team": "Arsenal",
                "neutral": 0,
                "home_goals": 2,
                "away_goals": 1,
                "home_shots": 12,
                "away_shots": 9,
                "home_sot": 6,
                "away_sot": 3,
                "home_corners": 7,
                "away_corners": 4,
                "home_fouls": 8,
                "away_fouls": 10,
                "home_yellow_cards": 1,
                "away_yellow_cards": 2,
            },
            {
                "match_id": "m2",
                "date": "2024-08-08",
                "competition": "Premier League",
                "season": "2024-2025",
                "stage": "league",
                "team_scope": "club",
                "home_team": "Arsenal",
                "away_team": "Man City",
                "neutral": 0,
                "home_goals": 0,
                "away_goals": 1,
                "home_shots": 8,
                "away_shots": 11,
                "home_sot": 2,
                "away_sot": 5,
                "home_corners": 3,
                "away_corners": 6,
                "home_fouls": 11,
                "away_fouls": 9,
                "home_yellow_cards": 2,
                "away_yellow_cards": 1,
            },
        ]
    )


def test_team_registry_clubelo_aliases_are_generated() -> None:
    outputs = build_team_registry(_sample_matches(), dataset_name="test_registry")
    registry = outputs.registry.set_index("football_data_name")

    assert outputs.summary["version"] == "v0.49.6_team_registry"
    assert registry.loc["Man City", "clubelo_name"] == "Man City"
    assert registry.loc["Man City", "understat_name"] == "Manchester City"
    assert registry.loc["Arsenal", "alias_status"] == "seeded"




def test_clubelo_enrichment_uses_cached_team_histories(tmp_path: Path) -> None:
    matches = _sample_matches()
    registry = build_team_registry(matches).registry
    team_dir = tmp_path / "clubelo" / "teams"
    team_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"Rank": 1, "Club": "Man City", "Country": "ENG", "Level": 1, "Elo": 2050.0, "From": "2024-01-01", "To": "2024-08-05"},
            {"Rank": 1, "Club": "Man City", "Country": "ENG", "Level": 1, "Elo": 2060.0, "From": "2024-08-06", "To": "2024-12-31"},
        ]
    ).to_csv(team_dir / "clubelo_team_man_city.csv", index=False)
    pd.DataFrame(
        [
            {"Rank": 2, "Club": "Arsenal", "Country": "ENG", "Level": 1, "Elo": 1980.0, "From": "2024-01-01", "To": "2024-08-04"},
            {"Rank": 2, "Club": "Arsenal", "Country": "ENG", "Level": 1, "Elo": 1990.0, "From": "2024-08-05", "To": "2024-12-31"},
        ]
    ).to_csv(team_dir / "clubelo_team_arsenal.csv", index=False)

    outputs = enrich_matches_with_clubelo(
        matches,
        registry,
        tmp_path / "clubelo",
        dataset_name="test_clubelo_history",
        source_mode="team-history",
    )
    enriched = outputs.enriched_matches.set_index("match_id")

    assert outputs.summary["status"] == "ok"
    assert outputs.summary["source_mode"] == "team-history"
    assert float(enriched.loc["m1", "home_clubelo"]) == 2050.0
    assert float(enriched.loc["m1", "away_clubelo"]) == 1980.0
    assert float(enriched.loc["m2", "home_clubelo"]) == 1990.0
    assert float(enriched.loc["m2", "away_clubelo"]) == 2060.0
    assert float(enriched.loc["m2", "clubelo_diff"]) == -70.0


def test_clubelo_enrichment_uses_cached_daily_snapshots(tmp_path: Path) -> None:
    matches = _sample_matches()
    registry = build_team_registry(matches).registry
    daily_dir = tmp_path / "clubelo" / "daily"
    daily_dir.mkdir(parents=True)
    snapshot = pd.DataFrame(
        [
            {"Rank": 1, "Club": "Man City", "Country": "ENG", "Elo": 2050.0},
            {"Rank": 2, "Club": "Arsenal", "Country": "ENG", "Elo": 1980.0},
        ]
    )
    snapshot.to_csv(daily_dir / "clubelo_2024-08-01.csv", index=False)
    snapshot.to_csv(daily_dir / "clubelo_2024-08-08.csv", index=False)

    outputs = enrich_matches_with_clubelo(matches, registry, tmp_path / "clubelo", dataset_name="test_clubelo")
    enriched = outputs.enriched_matches.set_index("match_id")

    assert outputs.summary["status"] == "ok"
    assert float(enriched.loc["m1", "home_clubelo"]) == 2050.0
    assert float(enriched.loc["m1", "away_clubelo"]) == 1980.0
    assert float(enriched.loc["m2", "clubelo_diff"]) == -70.0
    assert outputs.summary["coverage_rate"] == 1.0




def test_clubelo_download_team_histories_is_one_request_per_team(monkeypatch, tmp_path: Path) -> None:
    matches = _sample_matches()
    registry = build_team_registry(matches).registry
    calls: list[str] = []

    class _Response:
        text = "Rank,Club,Country,Level,Elo,From,To\n1,Man City,ENG,1,2050,2024-01-01,2024-12-31\n"

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, **kwargs: object) -> _Response:
        calls.append(url)
        return _Response()

    monkeypatch.setattr("mundialytics.enrichment.clubelo.requests.get", fake_get)
    outputs = download_clubelo_team_histories(
        matches,
        tmp_path / "clubelo",
        registry=registry,
        sleep_seconds=0,
    )

    assert outputs.report["mode"] == "team_histories"
    assert outputs.report["requested_team_aliases"] == 2
    assert len(calls) == 2
    assert outputs.report["downloaded_or_cached_files"] == 2


def test_xg_enrichment_and_model_ready_features_are_leakage_safe() -> None:
    matches = _sample_matches()
    registry = build_team_registry(matches).registry
    xg = pd.DataFrame(
        [
            {
                "provider": "understat",
                "provider_match_id": "u1",
                "date": "2024-08-01",
                "home_team": "Manchester City",
                "away_team": "Arsenal",
                "home_xg": 1.9,
                "away_xg": 0.8,
            },
            {
                "provider": "understat",
                "provider_match_id": "u2",
                "date": "2024-08-08",
                "home_team": "Arsenal",
                "away_team": "Manchester City",
                "home_xg": 0.6,
                "away_xg": 1.4,
            },
        ]
    )

    enriched = enrich_matches_with_xg(matches, xg, registry=registry, provider="understat").enriched_matches
    assert enriched["xg_available"].all()

    snapshots = build_model_ready_match_snapshots(enriched, dataset_name="test_enriched_snapshots").snapshots.set_index("match_id")
    # m2 can use Arsenal's prior xG from m1, but not m2's current xG.
    assert float(snapshots.loc["m2", "home_xg_for_last5"]) == 0.8
    assert float(snapshots.loc["m2", "target_home_xg"]) == 0.6
    assert "home_rest_days_pre" in snapshots.columns
    assert "home_goal_conversion_last5" in snapshots.columns


def test_external_enrichment_scripts_write_outputs(tmp_path: Path) -> None:
    matches = tmp_path / "matches.csv"
    _sample_matches().to_csv(matches, index=False)

    registry_dir = tmp_path / "entities"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_team_registry.py",
            "--matches",
            str(matches),
            "--out-dir",
            str(registry_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    summary = json.loads(result.stdout)
    assert summary["status"] == "ok"
    assert (registry_dir / "team_registry.csv").exists()

    xg = tmp_path / "xg.csv"
    pd.DataFrame(
        [
            {"date": "2024-08-01", "home_team": "Manchester City", "away_team": "Arsenal", "home_xg": 1.9, "away_xg": 0.8},
            {"date": "2024-08-08", "home_team": "Arsenal", "away_team": "Manchester City", "home_xg": 0.6, "away_xg": 1.4},
        ]
    ).to_csv(xg, index=False)
    out_dir = tmp_path / "xg_out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/enrich_matches_with_xg.py",
            "--matches",
            str(matches),
            "--xg",
            str(xg),
            "--registry",
            str(registry_dir / "team_registry.csv"),
            "--provider",
            "understat",
            "--out-dir",
            str(out_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    xg_summary = json.loads(result.stdout)
    assert xg_summary["coverage_rate"] == 1.0
    assert (out_dir / "canonical_matches_with_xg.csv").exists()
