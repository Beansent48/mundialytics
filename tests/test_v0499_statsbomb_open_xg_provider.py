from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from mundialytics.enrichment.statsbomb_open import import_statsbomb_open_xg
def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _sample_statsbomb_open_data(tmp_path: Path) -> Path:
    data = tmp_path / "open-data" / "data"
    _write_json(
        data / "competitions.json",
        [
            {
                "competition_id": 11,
                "season_id": 90,
                "competition_name": "La Liga",
                "season_name": "2020/2021",
            }
        ],
    )
    _write_json(
        data / "matches" / "11" / "90.json",
        [
            {
                "match_id": 1234,
                "match_date": "2021-01-02",
                "home_team": {"home_team_id": 1, "home_team_name": "Barcelona", "name": "Barcelona"},
                "away_team": {"away_team_id": 2, "away_team_name": "Real Madrid", "name": "Real Madrid"},
                "home_score": 2,
                "away_score": 1,
            }
        ],
    )
    _write_json(
        data / "events" / "1234.json",
        [
            {
                "id": "s1",
                "type": {"id": 16, "name": "Shot"},
                "team": {"id": 1, "name": "Barcelona"},
                "player": {"id": 10, "name": "Player One"},
                "minute": 12,
                "second": 5,
                "location": [102.0, 40.0],
                "shot": {
                    "statsbomb_xg": 0.25,
                    "type": {"name": "Open Play"},
                    "body_part": {"name": "Right Foot"},
                    "outcome": {"name": "Goal"},
                    "technique": {"name": "Normal"},
                },
            },
            {
                "id": "s2",
                "type": {"id": 16, "name": "Shot"},
                "team": {"id": 2, "name": "Real Madrid"},
                "player": {"id": 9, "name": "Player Two"},
                "minute": 55,
                "second": 12,
                "location": [108.0, 38.0],
                "shot": {
                    "statsbomb_xg": 0.76,
                    "type": {"name": "Penalty"},
                    "body_part": {"name": "Right Foot"},
                    "outcome": {"name": "Saved"},
                    "technique": {"name": "Normal"},
                },
            },
        ],
    )
    return data


def test_statsbomb_open_import_aggregates_match_and_shot_xg(tmp_path: Path) -> None:
    data = _sample_statsbomb_open_data(tmp_path)

    outputs = import_statsbomb_open_xg(data)

    assert outputs.report["version"] == "v0.49.9_statsbomb_open_xg_import"
    assert outputs.report["xg_match_rows"] == 1
    assert outputs.report["shot_rows"] == 2

    row = outputs.xg_matches.iloc[0]
    assert row["provider"] == "statsbomb_open_data"
    assert row["home_team"] == "Barcelona"
    assert row["away_team"] == "Real Madrid"
    assert abs(float(row["home_xg"]) - 0.25) < 1e-9
    assert abs(float(row["away_xg"]) - 0.76) < 1e-9
    assert abs(float(row["away_npxg"]) - 0.0) < 1e-9

    shots = outputs.xg_shots.sort_values("team")
    assert set(shots["team"]) == {"Barcelona", "Real Madrid"}
    assert "x" in shots.columns
    assert "y" in shots.columns


def test_import_statsbomb_open_xg_script_writes_outputs(tmp_path: Path) -> None:
    data = _sample_statsbomb_open_data(tmp_path)
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_statsbomb_open_xg.py",
            "--data-dir",
            str(data),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["status"] == "ok"
    assert (out_dir / "statsbomb_xg_matches.csv").exists()
    assert (out_dir / "statsbomb_xg_shots.csv").exists()
    assert pd.read_csv(out_dir / "statsbomb_xg_matches.csv").shape[0] == 1
