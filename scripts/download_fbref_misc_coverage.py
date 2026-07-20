#!/usr/bin/env python3
"""Close the FBref-misc coverage gap for the modern profiles: Serie A (all
seasons) + the early 2014/15-2016/17 seasons of all Big-5 leagues (the existing
cache only had ENG/ESP/GER/FRA for 2017/18-2025/26). Just triggers soccerdata
to fetch+cache each 'misc' page; build_player_profiles_modern.py then parses the
cached HTML offline. Serie A previously failed via soccerdata -- best effort.

Run with the project venv:
    .venv/Scripts/python.exe scripts/download_fbref_misc_coverage.py
"""
from __future__ import annotations

from soccerdata import FBref

EARLY = ["1415", "1516", "1617"]
ALL = EARLY + ["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
TARGETS = (
    [("ITA-Serie A", s) for s in ALL]
    + [(lg, s) for lg in ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "FRA-Ligue 1"]
       for s in EARLY]
)


def main() -> None:
    for lg, sn in TARGETS:
        try:
            fb = FBref(leagues=lg, seasons=sn)
            df = fb.read_player_season_stats(stat_type="misc")
            print(f"OK misc {lg} {sn}: {len(df)} rows", flush=True)
        except Exception as exc:
            print(f"FAIL misc {lg} {sn}: {str(exc)[:110]}", flush=True)
    print("DONE fbref misc coverage", flush=True)


if __name__ == "__main__":
    main()
