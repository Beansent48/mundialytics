#!/usr/bin/env python3
"""Targeted re-download of Understat shot events for the (league, season) pairs
missing from the local understat_shots.csv, then MERGE (append + dedupe by
shot_id) into the existing file. Append-only + dedupe so good data is never lost
if a scrape call fails.

Missing pairs identified 2026-07-20 (see [[project_xg_pipeline]]):
  Serie A 2015/16-2020/21 (6 seasons), Bundesliga 2024/25.

Run with the project venv (has soccerdata; Understat needs a one-time tls-client dll):
    .venv/Scripts/python.exe scripts/download_understat_missing_seasons.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from soccerdata import Understat

ROOT = Path(__file__).resolve().parents[1]
OUT_SHOTS = ROOT / "data/external/advanced/understat/understat_shots.csv"

# (league, season) pairs to (re)fetch.
MISSING = (
    [("ITA-Serie A", sn) for sn in ["1516", "1617", "1718", "1819", "1920", "2021"]]
    + [("GER-Bundesliga", "2425")]
)


def main() -> None:
    existing = pd.read_csv(OUT_SHOTS) if OUT_SHOTS.exists() else pd.DataFrame()
    print(f"existing shots: {len(existing)} rows", flush=True)

    new_frames: list[pd.DataFrame] = []
    for lg, sn in MISSING:
        try:
            u = Understat(leagues=lg, seasons=sn)
            sh = u.read_shot_events().reset_index()
            new_frames.append(sh)
            print(f"OK   shots {lg} {sn}: {len(sh)} rows", flush=True)
        except Exception as exc:
            print(f"FAIL shots {lg} {sn}: {str(exc)[:160]}", flush=True)

    if not new_frames:
        print("no new data fetched; leaving file unchanged", flush=True)
        return

    combined = pd.concat([existing, *new_frames], ignore_index=True)
    before = len(combined)
    if "shot_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["shot_id"], keep="first")
    print(f"merged: {before} -> {len(combined)} after dedupe (was {len(existing)})", flush=True)
    combined.to_csv(OUT_SHOTS, index=False)
    print(f"WROTE {OUT_SHOTS}", flush=True)

    # Quick per (league, season) game counts for the affected leagues.
    for lg in sorted({lg for lg, _ in MISSING}):
        sub = combined[combined["league"] == lg]
        counts = sub.groupby("season")["game_id"].nunique().to_dict()
        print(f"{lg} games/season now: {counts}", flush=True)


if __name__ == "__main__":
    main()
