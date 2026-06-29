
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from mundialytics.enrichment.advanced import (
    ADVANCED_DATA_VERSION,
    audit_advanced_data_coverage,
    enrich_matches_with_advanced_stats,
    import_advanced_provider_csv,
    import_statsbomb_open_advanced,
    merge_advanced_match_sources,
)


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _sample_statsbomb_open_data(tmp_path: Path) -> Path:
    data = tmp_path / "open-data" / "data"
    _write_json(
        data / "competitions.json",
        [{"competition_id": 11, "season_id": 90, "competition_name": "La Liga", "season_name": "2020/2021"}],
    )
    _write_json(
        data / "matches" / "11" / "90.json",
        [{
            "match_id": 1234,
            "match_date": "2021-01-02",
            "home_team": {"name": "Barcelona"},
            "away_team": {"name": "Real Madrid"},
            "home_score": 2,
            "away_score": 1,
        }],
    )
    _write_json(
        data / "events" / "1234.json",
        [
            {
                "id": "s1",
                "type": {"name": "Shot"},
                "team": {"name": "Barcelona"},
                "player": {"id": 10, "name": "Player One"},
                "minute": 12,
                "second": 5,
                "location": [102.0, 40.0],
                "shot": {
                    "statsbomb_xg": 0.25,
                    "type": {"name": "Open Play"},
                    "body_part": {"name": "Right Foot"},
                    "outcome": {"name": "Goal"},
                },
            },
            {
                "id": "p1",
                "type": {"name": "Pass"},
                "team": {"name": "Barcelona"},
                "player": {"id": 8, "name": "Midfielder"},
                "pass": {"length": 30.0},
            },
            {
                "id": "s2",
                "type": {"name": "Shot"},
                "team": {"name": "Real Madrid"},
                "player": {"id": 9, "name": "Player Two"},
                "minute": 55,
                "second": 12,
                "location": [108.0, 38.0],
                "shot": {
                    "statsbomb_xg": 0.76,
                    "type": {"name": "Penalty"},
                    "body_part": {"name": "Right Foot"},
                    "outcome": {"name": "Saved"},
                },
            },
            {
                "id": "pr1",
                "type": {"name": "Pressure"},
                "team": {"name": "Real Madrid"},
                "player": {"id": 6, "name": "Press Player"},
            },
        ],
    )
    return data


def test_statsbomb_advanced_import_outputs_match_player_and_shots(tmp_path: Path) -> None:
    data = _sample_statsbomb_open_data(tmp_path)

    outputs = import_statsbomb_open_advanced(data)

    assert outputs.report["version"] == ADVANCED_DATA_VERSION
    assert outputs.report["advanced_match_rows"] == 1
    assert outputs.report["shot_event_rows"] == 2
    row = outputs.advanced_matches.iloc[0]
    assert row["home_xg"] == 0.25
    assert row["away_xg"] == 0.76
    assert row["away_penalty_xg"] == 0.76
    assert outputs.player_matches["player"].nunique() >= 3


def test_advanced_csv_merge_enrich_and_audit(tmp_path: Path) -> None:
    provider_csv = tmp_path / "provider.csv"
    pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "competition": "Premier League",
                "season": "2023-2024",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_xg": 1.8,
                "away_xg": 0.7,
                "home_xa": 1.2,
                "away_xa": 0.4,
                "home_shots": 15,
                "away_shots": 8,
                "home_possession": 61,
                "away_possession": 39,
            }
        ]
    ).to_csv(provider_csv, index=False)

    imported = import_advanced_provider_csv(provider_csv, provider="fbref")
    merged = merge_advanced_match_sources([("fbref", imported.advanced_matches)])

    matches = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2024-01-01",
                "competition": "Premier League",
                "season": "2023-2024",
                "home_team": "arsenal",
                "away_team": "chelsea",
                "home_goals": 2,
                "away_goals": 1,
            }
        ]
    )
    enriched = enrich_matches_with_advanced_stats(matches, merged.canonical_advanced_matches, dataset_name="test")
    assert enriched.summary["matches_with_advanced_data"] == 1
    assert enriched.enriched_matches.loc[0, "home_xg"] == 1.8

    audit = audit_advanced_data_coverage(enriched.enriched_matches)
    coverage = audit["coverage"].set_index("feature_group")
    assert coverage.loc["xg", "status"] == "available"
    assert coverage.loc["possession", "status"] == "available"



def test_advanced_enrichment_uses_manual_aliases_and_preserves_base_stats() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_id": "m_psg",
                "date": "2021-10-29",
                "competition": "Ligue 1",
                "season": "2021-2022",
                "home_team": "paris sg",
                "away_team": "lille",
                "home_shots": 17,
                "away_shots": 11,
                "home_sot": 6,
                "away_sot": 4,
            }
        ]
    )
    advanced = pd.DataFrame(
        [
            {
                "provider": "statsbomb_open_data",
                "provider_match_id": "sb_1",
                "date": "2021-10-29",
                "competition": "Ligue 1",
                "season": "2021-2022",
                "home_team": "Paris Saint-Germain",
                "away_team": "Lille",
                "home_xg": 2.41,
                "away_xg": 1.73,
                "home_shots": pd.NA,
                "away_shots": pd.NA,
                "home_sot": pd.NA,
                "away_sot": pd.NA,
            }
        ]
    )
    manual_aliases = pd.DataFrame(
        [
            {
                "football_data_name": "paris sg",
                "provider_name": "Paris Saint-Germain",
                "canonical_name": "Paris Saint-Germain",
            }
        ]
    )

    merged_provider = merge_advanced_match_sources([("statsbomb_open_data", advanced)])
    enriched = enrich_matches_with_advanced_stats(
        matches,
        merged_provider.canonical_advanced_matches,
        manual_aliases=manual_aliases,
        dataset_name="manual_alias_test",
    )

    row = enriched.enriched_matches.iloc[0]
    assert enriched.summary["matches_with_advanced_data"] == 1
    assert row["provider"] == "statsbomb_open_data"
    assert row["home_xg"] == 2.41
    assert row["home_shots"] == 17
    assert row["away_sot"] == 4


def test_fbref_team_match_raw_exports_are_pivoted_and_mapped() -> None:
    raw = pd.DataFrame(
        [
            {
                "league": "ENG-Premier League",
                "season": 2024,
                "team": "Arsenal",
                "game": "Arsenal-Chelsea",
                "date": "2024-01-01",
                "venue": "Home",
                "opponent": "Chelsea",
                "Standard": 2,
                "Standard.1": 15,
                "Standard.2": 6,
                "Standard.3": 40.0,
                "Standard.4": 0.13,
                "Standard.5": 0.33,
                "Standard.6": 1,
                "Standard.7": 0,
                "match_report": "/en/matches/abc/Arsenal-Chelsea",
            },
            {
                "league": "ENG-Premier League",
                "season": 2024,
                "team": "Chelsea",
                "game": "Arsenal-Chelsea",
                "date": "2024-01-01",
                "venue": "Away",
                "opponent": "Arsenal",
                "Standard": 1,
                "Standard.1": 8,
                "Standard.2": 3,
                "Standard.3": 37.5,
                "Standard.4": 0.12,
                "Standard.5": 0.33,
                "Standard.6": 0,
                "Standard.7": 0,
                "match_report": "/en/matches/abc/Arsenal-Chelsea",
            },
        ]
    )

    # Exercise the public CSV import path because this is how the local
    # pipeline normalizes soccerdata FBref raw files.
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8", delete=False) as f:
        raw.to_csv(f.name, index=False)
        outputs = import_advanced_provider_csv(f.name, provider="fbref_shooting")

    assert len(outputs.advanced_matches) == 1
    row = outputs.advanced_matches.iloc[0]
    assert row["home_team"] == "Arsenal"
    assert row["away_team"] == "Chelsea"
    assert row["home_shots"] == 15
    assert row["away_shots"] == 8
    assert row["home_sot"] == 6
    assert row["away_sot"] == 3
    assert pd.isna(row["home_xg"])


def test_fbref_keeper_and_misc_raw_exports_are_mapped() -> None:
    keeper = pd.DataFrame(
        [
            {
                "league": "ENG-Premier League",
                "season": 2024,
                "team": "Arsenal",
                "game": "Arsenal-Chelsea",
                "date": "2024-01-01",
                "venue": "Home",
                "opponent": "Chelsea",
                "Performance": 3,
                "Performance.1": 1,
                "Performance.2": 2,
                "Performance.3": 66.7,
                "Performance.4": 0,
            },
            {
                "league": "ENG-Premier League",
                "season": 2024,
                "team": "Chelsea",
                "game": "Arsenal-Chelsea",
                "date": "2024-01-01",
                "venue": "Away",
                "opponent": "Arsenal",
                "Performance": 6,
                "Performance.1": 2,
                "Performance.2": 4,
                "Performance.3": 66.7,
                "Performance.4": 0,
            },
        ]
    )
    misc = pd.DataFrame(
        [
            {
                "league": "ENG-Premier League",
                "season": 2024,
                "team": "Arsenal",
                "game": "Arsenal-Chelsea",
                "date": "2024-01-01",
                "venue": "Home",
                "opponent": "Chelsea",
                "Performance": 2,
                "Performance.1": 0,
                "Performance.3": 11,
                "Performance.7": 10,
                "Performance.8": 5,
            },
            {
                "league": "ENG-Premier League",
                "season": 2024,
                "team": "Chelsea",
                "game": "Arsenal-Chelsea",
                "date": "2024-01-01",
                "venue": "Away",
                "opponent": "Arsenal",
                "Performance": 3,
                "Performance.1": 1,
                "Performance.3": 14,
                "Performance.7": 8,
                "Performance.8": 6,
            },
        ]
    )

    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8", delete=False) as f:
        keeper.to_csv(f.name, index=False)
        keeper_outputs = import_advanced_provider_csv(f.name, provider="fbref_keeper")

    with NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8", delete=False) as f:
        misc.to_csv(f.name, index=False)
        misc_outputs = import_advanced_provider_csv(f.name, provider="fbref_misc")

    keeper_row = keeper_outputs.advanced_matches.iloc[0]
    assert keeper_row["home_keeper_saves"] == 2
    assert keeper_row["away_keeper_saves"] == 4
    assert keeper_row["home_keeper_goals_against"] == 1
    assert keeper_row["away_keeper_save_pct"] == 66.7

    misc_row = misc_outputs.advanced_matches.iloc[0]
    assert misc_row["home_yellow_cards"] == 2
    assert misc_row["away_red_cards"] == 1
    assert misc_row["home_fouls"] == 11
    assert misc_row["away_interceptions"] == 8

def test_import_statsbomb_open_advanced_script_smoke(tmp_path: Path) -> None:
    data = _sample_statsbomb_open_data(tmp_path)
    out = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts" / "import_statsbomb_open_advanced.py"
    result = subprocess.run(
        [sys.executable, str(script), "--data-dir", str(data), "--out-dir", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "ok"
    assert (out / "statsbomb_advanced_match_stats.csv").exists()
    assert (out / "statsbomb_player_match_stats.csv").exists()
    assert (out / "statsbomb_shot_events.csv").exists()
