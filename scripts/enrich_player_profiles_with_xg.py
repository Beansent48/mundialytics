#!/usr/bin/env python3
"""Enrich player_profiles_with_positions.csv with real per-shot xG.

Aggregates data/external/xg/statsbomb/statsbomb_xg_shots.csv (one row per shot,
with StatsBomb's shot xG) into per-player rates and merges them onto the
existing profiles CSV used by PlayerStrengthModel:

  - xg_per_match:                 sum(shot xG) / matches
  - npxg_per_match:                sum(non-penalty shot xG) / matches
  - big_chances_missed_per_match:  shots with xg >= BIG_CHANCE_XG_THRESHOLD
                                    that did not result in a goal, / matches

"matches" is taken from the existing profiles CSV (not recomputed from the
shots file) so the rate stays consistent with every other *_per_match column
already in that file.

Run after the profiles CSV exists:
    python scripts/enrich_player_profiles_with_xg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROFILES_PATH = ROOT / "data/processed/player_profiles_with_positions.csv"
SHOTS_PATH = ROOT / "data/external/xg/statsbomb/statsbomb_xg_shots.csv"

# A shot with xG above this is a clear scoring opportunity ("big chance").
# 0.3 is the common analytics convention (~roughly a 1-in-3 finish or better).
BIG_CHANCE_XG_THRESHOLD = 0.3


def main() -> None:
    if not PROFILES_PATH.exists():
        raise FileNotFoundError(f"Profiles CSV not found: {PROFILES_PATH}")
    if not SHOTS_PATH.exists():
        raise FileNotFoundError(f"Shots CSV not found: {SHOTS_PATH}")

    profiles = pd.read_csv(PROFILES_PATH)
    shots = pd.read_csv(SHOTS_PATH, low_memory=False)

    shots["xg"] = pd.to_numeric(shots["xg"], errors="coerce").fillna(0.0)
    shots["npxg"] = pd.to_numeric(shots["npxg"], errors="coerce").fillna(0.0)
    shots["is_goal"] = shots["outcome"] == "Goal"
    shots["is_big_chance"] = shots["xg"] >= BIG_CHANCE_XG_THRESHOLD
    shots["is_big_chance_missed"] = shots["is_big_chance"] & ~shots["is_goal"]

    agg = (
        shots.groupby("player")
        .agg(
            total_xg=("xg", "sum"),
            total_npxg=("npxg", "sum"),
            total_shots_xg_sample=("xg", "size"),
            big_chances_missed=("is_big_chance_missed", "sum"),
        )
        .reset_index()
    )

    before_cols = set(profiles.columns)
    profiles = profiles.merge(agg, on="player", how="left")
    for c in ("total_xg", "total_npxg", "total_shots_xg_sample", "big_chances_missed"):
        profiles[c] = profiles[c].fillna(0.0)

    matches = profiles["matches"].clip(lower=1)
    profiles["xg_per_match"] = profiles["total_xg"] / matches
    profiles["npxg_per_match"] = profiles["total_npxg"] / matches
    profiles["big_chances_missed_per_match"] = profiles["big_chances_missed"] / matches
    profiles["xg_sample"] = profiles["total_shots_xg_sample"].astype(int)

    profiles = profiles.drop(columns=["total_xg", "total_npxg", "total_shots_xg_sample", "big_chances_missed"])

    new_cols = set(profiles.columns) - before_cols
    matched = int((profiles["xg_sample"] > 0).sum())
    print(f"Matched {matched}/{len(profiles)} players to shot-level xG data")
    print(f"New columns: {sorted(new_cols)}")

    profiles.to_csv(PROFILES_PATH, index=False)
    print(f"Wrote {PROFILES_PATH}")


if __name__ == "__main__":
    main()
