#!/usr/bin/env python3
"""Seed the GitHub Actions cache with local data that cannot be re-downloaded.

Run this ONCE from your local machine after the first successful workflow run.
It creates a tarball of the essential data that the workflow uploads as a cache
entry, so subsequent runs have the full xG history.

Usage (from the repo root):

    python scripts/seed_actions_cache.py

This prints the paths and sizes of what it would pack. To actually upload,
push the tarball as a release asset and add a download step to the workflow —
OR simply trigger the workflow_dispatch twice:

  1. First run: builds the foundation without xG (cache bootstrapped).
  2. Meanwhile, copy your local xG data into the cache paths:
       git stash -u  # clean state
       # The next workflow run picks up whatever is in the cache.

The simplest seeding method: after the first automatic run succeeds (the cache
now exists), run the workflow again via workflow_dispatch. The cache from run 1
already has the foundation + football-data CSVs. The pipeline's step 3 (xG
aggregation) reads whatever Understat data exists and step 3c (ESPN text xG)
fills the current season. The only gap is the Understat HISTORY (2016-2026),
which affects historical xG coverage (~97% -> ~85% for seasons before 2026/27).

If you want FULL historical xG in the cloud pipeline:
  1. Create a GitHub release with a tarball of data/external/advanced/understat/
  2. Add a download step to the workflow that fetches it on cache-miss.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEED_PATHS = [
    ("data/raw/football_data", "Foundation source (26 seasons × 5 leagues)"),
    ("data/external/advanced/understat", "Historical xG shots + player-match (dead source)"),
    ("data/external/xg", "xG match aggregation"),
    ("data/external/clubelo", "ClubElo seed (local roll-forward)"),
    ("data/external/uefa", "UEFA fixtures"),
    ("data/processed/sofascore_team_match_xg.csv", "Sofascore current-season xG"),
    ("data/processed/espn_text_team_match_xg.csv", "ESPN text xG (current season)"),
    ("data/processed/understat_team_match_xg.csv", "Understat xG aggregation"),
]


def main() -> None:
    total = 0
    print("Files that the Actions cache needs (seed from local):\n")
    for rel, desc in SEED_PATHS:
        p = ROOT / rel
        if not p.exists():
            print(f"  MISSING  {rel}")
            continue
        if p.is_file():
            sz = p.stat().st_size
        else:
            sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        total += sz
        mb = sz / 1024 / 1024
        print(f"  {mb:8.1f} MB  {rel:50s}  {desc}")
    print(f"\n  {'TOTAL':>8s}    {total / 1024 / 1024:.1f} MB")
    print("\nAfter the first workflow_dispatch run succeeds, the cache holds the")
    print("football-data CSVs and the processed foundation. The Understat history")
    print("is the only piece that cannot be re-downloaded — see the docstring for")
    print("how to seed it via a GitHub release asset.")


if __name__ == "__main__":
    main()
