#!/usr/bin/env python3
"""Bulk-download Understat per-match player stats (= historical LINEUPS + per-player
per-match xG/xA/xg_chain) for the Big5 covered window.

Each match yields ~28-32 player rows with position (GK/DL/DMC/.../FW, and 'Sub' for
substitutes -> starters are position != 'Sub'), minutes, goals, xG, xA, xg_chain,
xg_buildup, key_passes, cards. This is BOTH the lineup history (who started, who was
missing) and the player-quality source for the XI-strength feature — one dataset,
no browser, proven tls-client path (~0.6s/match, cached by soccerdata, resumable).

Per league-season isolation, append + dedupe on (game_id, player_id), safe to re-run.

Run:
    .venv/Scripts/python.exe scripts/download_understat_player_match.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from soccerdata import Understat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external/advanced/understat/understat_player_match.csv"
LEAGUES = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]
SEASONS = ["1415", "1516", "1617", "1718", "1819", "1920", "2021",
           "2122", "2223", "2324", "2425", "2526"]


def main() -> None:
    existing = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    done_games: set = set(existing["game_id"].unique()) if len(existing) else set()
    print(f"existing rows: {len(existing)} ({len(done_games)} games)", flush=True)

    total_new = 0
    for lg in LEAGUES:
        for sn in SEASONS:
            t0 = time.time()
            try:
                u = Understat(leagues=lg, seasons=sn)
                pm = u.read_player_match_stats().reset_index()
                if len(pm) == 0:
                    print(f"EMPTY {lg} {sn}", flush=True)
                    continue
                new = pm[~pm["game_id"].isin(done_games)]
                if len(new):
                    combined = pd.concat([existing, new], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["game_id", "player_id"], keep="first")
                    combined.to_csv(OUT, index=False)
                    existing = combined
                    done_games.update(new["game_id"].unique())
                    total_new += len(new)
                print(f"OK   {lg} {sn}: {pm['game_id'].nunique()} games, +{len(new)} new rows "
                      f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as exc:
                print(f"FAIL {lg} {sn}: {str(exc)[:140]}", flush=True)

    print(f"\nDONE: total {len(existing)} rows, {existing['game_id'].nunique() if len(existing) else 0} games "
          f"(+{total_new} this run) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
