#!/usr/bin/env python3
"""Download Understat per-season player stats + shot events for the Big-5
leagues, 2014/15-2025/26 -- the rich MODERN-player source (xG, np_xG, xA,
key_passes, xg_chain, xg_buildup at season level; per-shot xG for finishing).

Fills the modern gap where StatsBomb open data stops (e.g. Kanté only had his
2015/16 Leicester StatsBomb season; this brings his Chelsea years, Ronaldo's
Madrid xG, Modrić's peak, etc.). See [[project_data_state]] for the full
source inventory. Understat needs a one-time tls-client dll download.

Run with the project venv (has soccerdata):
    .venv/Scripts/python.exe scripts/download_understat_players.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from soccerdata import Understat

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]
SEASONS = ["1415", "1516", "1617", "1718", "1819", "1920", "2021",
           "2122", "2223", "2324", "2425", "2526"]
OUT_SEASON = ROOT / "data/external/advanced/understat/understat_player_season.csv"
OUT_SHOTS = ROOT / "data/external/advanced/understat/understat_shots.csv"


def main() -> None:
    OUT_SEASON.parent.mkdir(parents=True, exist_ok=True)
    season_frames, shot_frames = [], []
    for lg in LEAGUES:
        # Season stats (small, essential) -- one call per league across all seasons.
        try:
            u = Understat(leagues=lg, seasons=SEASONS)
            s = u.read_player_season_stats().reset_index()
            season_frames.append(s)
            print(f"season {lg}: {len(s)} rows", flush=True)
        except Exception as exc:
            print(f"season {lg} FAILED: {str(exc)[:120]}", flush=True)
        # Shot events (big, for per-shot finishing) -- per season to isolate failures.
        for sn in SEASONS:
            try:
                u = Understat(leagues=lg, seasons=sn)
                sh = u.read_shot_events().reset_index()
                shot_frames.append(sh)
                print(f"shots {lg} {sn}: {len(sh)} rows", flush=True)
            except Exception as exc:
                print(f"shots {lg} {sn} FAILED: {str(exc)[:120]}", flush=True)

    if season_frames:
        out = pd.concat(season_frames, ignore_index=True)
        out.to_csv(OUT_SEASON, index=False)
        print(f"WROTE {OUT_SEASON} ({len(out)} rows, {out['player'].nunique()} players)", flush=True)
    if shot_frames:
        out = pd.concat(shot_frames, ignore_index=True)
        out.to_csv(OUT_SHOTS, index=False)
        print(f"WROTE {OUT_SHOTS} ({len(out)} shots)", flush=True)


if __name__ == "__main__":
    main()
