#!/usr/bin/env python3
"""Rebuild data/processed/foundation_big5_multi_season.csv with extended
history (2000/01-2025/26 by default) from football-data.co.uk CSVs.

Why a new script instead of the existing football_data_uk_to_matches()
adapter (src/mundialytics/data/adapters/football_data_uk.py): that adapter
(and international_results_to_matches / openfootball_json_to_matches)
imports `normalize_matches` from mundialytics.data.schema -- confirmed
(2026-07-01) this function has never existed in that module since the
initial commit (verified via `git show <initial-commit>:.../schema.py`), so
all three adapters have always raised ImportError at call time. This was
masked locally by a stale bytecode cache. Rather than reverse-engineer what
normalize_matches was supposed to do, this script parses the CSVs directly
and matches foundation_big5_multi_season.csv's existing column schema
exactly (verified by inspecting the current file's dtypes/sample rows).

The existing file (8907 rows, 2021-2026, all 5 Big5 leagues) was entirely
sourced from football-data.co.uk already (source column confirms it) but
only had 5 seasons downloaded (data/raw/football_data/ only had E0/SP1 CSVs
for 2122-2526 before this run). scripts/download_football_data_stats.py was
run first to fetch E0/D1/I1/SP1/F1 for season codes 0001-2526 (26 seasons x
5 leagues = 130 files) -- see that script for the download step.

This is a full REBUILD (not a merge/append) since it's the exact same
source for old and new rows alike -- simpler and avoids any schema drift
between an old and new half of the file.

Run:
    python scripts/build_foundation_big5_historical.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", message="Could not infer format")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.identity.normalization import canonical_team_name

RAW_DIR = ROOT / "data/raw/football_data"
OUT_PATH = ROOT / "data/processed/foundation_big5_multi_season.csv"

_FD_MAP = {
    "Date": "date",
    "HomeTeam": "home_team_raw",
    "AwayTeam": "away_team_raw",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
}

_DIV_LABELS = {
    "E0": "Premier League",
    "SP1": "LaLiga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}


def _season_label(season_code: str) -> str:
    yy1, yy2 = season_code[:2], season_code[2:]
    century1 = "19" if yy1 in ("93", "94", "95", "96", "97", "98", "99") else "20"
    century2 = "19" if yy2 in ("94", "95", "96", "97", "98", "99") else "20"
    return f"{century1}{yy1}-{century2}{yy2}"


def _parse_one(path: Path) -> pd.DataFrame:
    season_code, league = path.stem.split("_", 1)
    # A handful of older football-data CSVs have malformed trailing columns
    # on odd rows (extra commas in bookmaker-odds fields) -- skip those rows
    # rather than fail the whole file.
    raw = pd.read_csv(path, encoding="latin1", on_bad_lines="skip", engine="python")
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(raw.columns)
    if missing:
        return pd.DataFrame()
    keep = [c for c in _FD_MAP if c in raw.columns]
    out = raw[keep].rename(columns=_FD_MAP).copy()
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
    out = out.dropna(subset=["date", "home_goals", "away_goals"]).reset_index(drop=True)
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)
    out["competition"] = _DIV_LABELS.get(league, league)
    out["season"] = _season_label(season_code)
    out["match_id"] = [f"fduk_{season_code}_{league}_{i:05d}" for i in range(len(out))]
    out["neutral"] = 0
    out["team_scope"] = "club"
    out["source"] = "football-data.co.uk"
    out["stage"] = "Regular Season"
    out["home_team"] = out["home_team_raw"].map(canonical_team_name)
    out["away_team"] = out["away_team_raw"].map(canonical_team_name)
    out["home_team_id"] = "club_" + out["home_team"].str.replace(" ", "_", regex=False)
    out["away_team_id"] = "club_" + out["away_team"].str.replace(" ", "_", regex=False)
    out["source_file"] = str(path)
    return out


def main() -> None:
    # Only root-level CSVs from the csv-mode downloader (named
    # "{season}_{league}.csv", e.g. "0001_E0.csv"), restricted to Big5 league
    # codes. data/raw/football_data/ also has older per-season SUBDIRECTORIES
    # (2122/, 2223/, ...) from an earlier --mode zip download that pulled
    # every league in DEFAULT_LEAGUES (Championship, Scottish leagues, etc.)
    # -- rglob would double-count 2021-2026 and pull in non-Big5 leagues.
    files = sorted(
        f for f in RAW_DIR.glob("*.csv")
        if f.stem.split("_", 1)[-1] in _DIV_LABELS
    )
    frames = []
    skipped = []
    for f in files:
        df = _parse_one(f)
        if df.empty:
            skipped.append(f.name)
            continue
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    col_order = [
        "date", "home_team", "away_team", "home_goals", "away_goals", "competition",
        "home_shots", "away_shots", "home_sot", "away_sot", "home_corners", "away_corners",
        "home_fouls", "away_fouls", "home_yellow_cards", "away_yellow_cards",
        "match_id", "neutral", "team_scope", "source", "stage", "season",
        "home_team_raw", "home_team_id", "away_team_raw", "away_team_id", "source_file",
    ]
    combined = combined[col_order].sort_values("date").reset_index(drop=True)

    combined.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(combined)} matches -> {OUT_PATH}")
    print(f"Skipped {len(skipped)} files (missing required columns): {skipped[:10]}")
    print(f"Date range: {combined['date'].min()} - {combined['date'].max()}")
    print(combined["season"].value_counts().sort_index())
    print()
    print(combined["competition"].value_counts())
    print()
    for col in ["home_shots", "home_sot", "home_corners", "home_fouls", "home_yellow_cards"]:
        rate = combined[col].notna().mean()
        print(f"{col} coverage: {rate:.1%}")


if __name__ == "__main__":
    main()
