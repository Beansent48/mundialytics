from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from mundialytics.enrichment.understat import download_understat_xg, normalize_provider_xg_csv


def test_understat_blocked_download_writes_empty_combined_csv(monkeypatch, tmp_path: Path) -> None:
    def fake_fetch(*args: object, **kwargs: object) -> pd.DataFrame:
        raise ValueError("Could not find Understat datesData JSON in page.")

    monkeypatch.setattr("mundialytics.enrichment.understat.fetch_understat_league_season", fake_fetch)

    outputs = download_understat_xg([("EPL", 2024)], tmp_path, sleep_seconds=0)

    combined = tmp_path / "understat_xg_matches.csv"
    assert outputs.report["status"] == "blocked"
    assert outputs.report["direct_scrape_blocked"] is True
    assert combined.exists()
    assert pd.read_csv(combined).empty


def test_import_xg_csv_normalizes_provider_columns(tmp_path: Path) -> None:
    source = tmp_path / "provider.csv"
    pd.DataFrame(
        [
            {
                "match_date": "2024-08-17",
                "league": "EPL",
                "season": "2024-2025",
                "home": "Arsenal",
                "away": "Wolves",
                "xg_home": 1.7,
                "xg_away": 0.6,
            }
        ]
    ).to_csv(source, index=False)

    outputs = normalize_provider_xg_csv(source, tmp_path / "xg", provider="manual_test")

    out = pd.read_csv(tmp_path / "xg" / "understat_xg_matches.csv")
    assert outputs.report["status"] == "ok"
    assert out.loc[0, "provider"] == "manual_test"
    assert float(out.loc[0, "home_xg"]) == 1.7
    assert float(out.loc[0, "away_xg"]) == 0.6


def test_enrich_matches_with_xg_allow_missing_xg_does_not_crash(tmp_path: Path) -> None:
    matches = tmp_path / "matches.csv"
    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2024-08-17",
                "competition": "Premier League",
                "season": "2024-2025",
                "team_scope": "club",
                "home_team": "Arsenal",
                "away_team": "Wolves",
                "neutral": 0,
                "home_goals": 2,
                "away_goals": 0,
            }
        ]
    ).to_csv(matches, index=False)

    out_dir = tmp_path / "enriched"
    cmd = [
        sys.executable,
        "scripts/enrich_matches_with_xg.py",
        "--matches",
        str(matches),
        "--xg",
        str(tmp_path / "missing_xg.csv"),
        "--out-dir",
        str(out_dir),
        "--allow-missing-xg",
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)

    report = json.loads((out_dir / "xg_enrichment_report.json").read_text(encoding="utf-8"))
    enriched = pd.read_csv(out_dir / "canonical_matches_with_xg.csv")
    assert report["status"] == "warning"
    assert report["warning"] == "xg_file_missing_empty_xg_used"
    assert int(report["matches_with_xg"]) == 0
    assert bool(enriched.loc[0, "xg_available"]) is False
    assert "xg_file_missing_empty_xg_used" in result.stdout
