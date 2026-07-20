#!/usr/bin/env python3
"""Merge FBref recent-season players (2017/18-2025/26, see
scripts/build_player_profiles_fbref_recent.py) into
player_profiles_with_positions.csv -- the career file PlayerStrengthModel
actually loads.

Scope decision: only ADD players not already present, never touch/blend
existing rows. player_profiles_with_positions.csv is StatsBomb-based and
already has real defensive-quality/creation stats (duel_win_rate,
dribbled_past_per_match, etc. -- see enrich_player_profiles_with_defense_creation.py)
for players it covers; blending in FBref's thinner stat set for players who
already have richer StatsBomb data would only dilute quality. This only
fills the gap where a player has ZERO existing coverage (mostly recent
signings/breakouts StatsBomb's open data never reached).

New players get defense_creation_matches=0 (and neutral defaults for every
defense-quality/creation stat) even though FBref does supply real
interceptions_per_match/tklw_per_match for them -- deliberately not used.
Reason: the shared defense_creation_matches column drives credibility for
ALL of duel_win_rate/dribbled_past/clearances/blocks/interceptions/
key_passes/pass_completion at once (see player_strength.py). Giving it a
nonzero value to unlock real interceptions credibility would ALSO give fake
non-neutral credibility to the other 6 stats' placeholder defaults (e.g.
dribbled_past_per_match=0.0 would read as an elite anti-dribbling record,
not "no data"). Since FBref only gives 1 of 7 defense/creation stats, the
safe choice is neutral-everything: these players get a fully real
off_score, and a neutral (50) def_score/creation_score -- same situation as
any other player currently missing Big5 event coverage.

Run after player_profiles_fbref_recent.csv exists:
    python scripts/merge_fbref_recent_players.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CAREER_PATH = ROOT / "data/processed/player_profiles_with_positions.csv"
FBREF_PATH = ROOT / "data/processed/player_profiles_fbref_recent.csv"

NEUTRAL_DEFAULTS = {
    "tackles_per_match": 0.0, "pressures_per_match": 0.0,
    "xg_per_match": 0.0, "npxg_per_match": 0.0,
    "big_chances_missed_per_match": 0.0, "big_chance_miss_rate": 0.0,
    "defense_creation_matches": 0, "duel_win_rate": 0.5,
    "dribbled_past_per_match": 0.0, "clearances_per_match": 0.0,
    "blocks_per_match": 0.0, "interceptions_per_match": 0.0,
    "key_passes_per_match": 0.0, "passes_per_match": 0.0, "pass_completion": 0.75,
}


def main() -> None:
    career = pd.read_csv(CAREER_PATH)
    fbref = pd.read_csv(FBREF_PATH)

    existing_players = set(career["player"])
    new_players = fbref[~fbref["player"].isin(existing_players)]
    print(f"FBref players: {fbref['player'].nunique()} unique, "
          f"{len(new_players['player'].unique())} not already in the career file")

    # Roll up season rows to one row per (player, most-recent team) --
    # matches-weighted average for rates, sum for matches. Using the most
    # recent season's team keeps the displayed club current rather than
    # picking an arbitrary one.
    new_players = new_players.sort_values(["player", "season"])
    latest_team = new_players.groupby("player").last()[["team", "competition", "position"]]

    def weighted(g: pd.DataFrame, col: str) -> float:
        return (g[col] * g["matches"]).sum() / g["matches"].sum()

    rows = []
    for player, g in new_players.groupby("player"):
        total_matches = int(g["matches"].sum())
        row = {
            "player": player, "team": latest_team.loc[player, "team"],
            "competition": latest_team.loc[player, "competition"],
            "position": latest_team.loc[player, "position"], "matches": total_matches,
        }
        for stat in ("goals_per_match", "assists_per_match", "shots_per_match", "sot_per_match",
                     "fouls_per_match", "yellow_cards_per_match"):
            row[stat] = weighted(g, stat)
        rows.append(row)

    new_rows = pd.DataFrame(rows)
    for col, default in NEUTRAL_DEFAULTS.items():
        new_rows[col] = default
    for col in ("shots_sample", "sot_sample", "goals_sample", "assists_sample",
                "fouls_sample", "yellow_cards_sample", "tackles_sample",
                "pressures_sample", "xg_sample"):
        new_rows[col] = new_rows["matches"]

    new_rows = new_rows[career.columns.tolist()]
    combined = pd.concat([career, new_rows], ignore_index=True)
    combined.to_csv(CAREER_PATH, index=False)
    print(f"Added {len(new_rows)} new players -> {CAREER_PATH} ({len(combined)} total rows)")


if __name__ == "__main__":
    main()
