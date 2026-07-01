#!/usr/bin/env python3
"""Enrich player_profiles_with_positions.csv with defensive-QUALITY stats
(duel win rate, times dribbled past, clearances, blocks, interceptions) and
creation stats (key passes, pass completion %), sourced from
data/processed/player_match_events.csv.

Why this exists: investigated (2026-07-01) why the defense/midfielder rating
formulas produce nonsensical rankings (legendary center-backs like Sergio
Ramos/Van Dijk/Piqué ranking in the hundreds; midfielders capped at ~66 with
destroyers outranking Xavi/Modric/Kroos). Root cause confirmed at both team
and individual level: tackles_per_match/pressures_per_match are workrate
(volume) stats that are NEGATIVELY correlated with real team quality --
teams/players under constant defensive pressure rack up more tackles/presses
out of necessity, while dominant teams/players barely need to. Confirmed via
a 99-team-season regression against real AttackDefenseModel parameters (see
scripts/fit_squad_lambda_calibration_season_scoped.py's diagnostics).

Fix: replace volume with QUALITY signals newly extracted from raw StatsBomb
events (see statsbomb_events_to_player_events in
src/mundialytics/data/adapters/statsbomb.py) -- duel win/loss outcome,
"Dribbled Past" (opponent beat this player), Clearance, Block. These don't
have the same possession-context confound: a good defender wins a high % of
their duels and rarely gets dribbled past regardless of how much of the ball
their team's dominance affords them.

Also merges key_passes_per_match/passes_per_match/complete_passes_per_match
for a new "creation" score axis (see player_strength.py) -- these already
existed in the season-split rebuild but were never in the career file.

IMPORTANT -- coverage caveat: player_match_events.csv only covers the 5 Big5
domestic leagues (see build_player_profiles_by_season.py), while the career
file's own "matches" column may include Cup/Champions League/international
matches from broader StatsBomb open data. So the Big5-only match count used
here (bigfive_matches) is generally <= the existing "matches" column, and for
~4900/9885 players (mostly international-only appearances) it's zero. A
dedicated defense_creation_matches column is written so player_strength.py
can apply its OWN credibility shrinkage for these new stats (using the true
Big5-scoped sample size) instead of reusing the general "matches" column,
which would overstate confidence for players whose new-stat coverage is
much thinner than their overall sample.

Run after player_profiles_by_season.csv / player_match_events.csv exist:
    python scripts/enrich_player_profiles_with_defense_creation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROFILES_PATH = ROOT / "data/processed/player_profiles_with_positions.csv"
PER_MATCH_PATH = ROOT / "data/processed/player_match_events.csv"


def main() -> None:
    if not PROFILES_PATH.exists():
        raise FileNotFoundError(f"Profiles CSV not found: {PROFILES_PATH}")
    if not PER_MATCH_PATH.exists():
        raise FileNotFoundError(f"Per-match events CSV not found: {PER_MATCH_PATH}")

    profiles = pd.read_csv(PROFILES_PATH)
    pm = pd.read_csv(PER_MATCH_PATH)

    agg = pm.groupby("player").agg(
        defense_creation_matches=("match_id", "nunique"),
        duels_won=("duels_won", "sum"), duels_lost=("duels_lost", "sum"),
        dribbled_past=("dribbled_past", "sum"), clearances=("clearances", "sum"),
        blocks=("blocks", "sum"), interceptions=("interceptions", "sum"),
        key_passes=("key_passes", "sum"), passes=("passes", "sum"),
        complete_passes=("complete_passes", "sum"),
    ).reset_index()

    matches_safe = agg["defense_creation_matches"].clip(lower=1)
    known_outcome_duels = agg["duels_won"] + agg["duels_lost"]
    # Outcome is untagged for a large share of duels (off-ball challenges) --
    # fall back to a neutral 0.5 for players with zero known-outcome duels
    # rather than a misleading 0.
    agg["duel_win_rate"] = (agg["duels_won"] / known_outcome_duels.clip(lower=1)).where(
        known_outcome_duels > 0, 0.5
    )
    agg["dribbled_past_per_match"] = agg["dribbled_past"] / matches_safe
    agg["clearances_per_match"] = agg["clearances"] / matches_safe
    agg["blocks_per_match"] = agg["blocks"] / matches_safe
    agg["interceptions_per_match"] = agg["interceptions"] / matches_safe
    agg["key_passes_per_match"] = agg["key_passes"] / matches_safe
    agg["passes_per_match"] = agg["passes"] / matches_safe
    agg["pass_completion"] = agg["complete_passes"] / agg["passes"].clip(lower=1)

    keep = agg[[
        "player", "defense_creation_matches", "duel_win_rate", "dribbled_past_per_match",
        "clearances_per_match", "blocks_per_match", "interceptions_per_match",
        "key_passes_per_match", "passes_per_match", "pass_completion",
    ]]

    before_cols = set(profiles.columns)
    profiles = profiles.merge(keep, on="player", how="left")
    profiles["defense_creation_matches"] = profiles["defense_creation_matches"].fillna(0).astype(int)
    profiles["duel_win_rate"] = profiles["duel_win_rate"].fillna(0.5)
    for c in ("dribbled_past_per_match", "clearances_per_match", "blocks_per_match",
              "interceptions_per_match", "key_passes_per_match", "passes_per_match"):
        profiles[c] = profiles[c].fillna(0.0)
    profiles["pass_completion"] = profiles["pass_completion"].fillna(0.75)

    new_cols = sorted(set(profiles.columns) - before_cols)
    matched = int((profiles["defense_creation_matches"] > 0).sum())
    print(f"Matched {matched}/{len(profiles)} players to Big5 per-match defense/creation data")
    print(f"New columns: {new_cols}")

    profiles.to_csv(PROFILES_PATH, index=False)
    print(f"Wrote {PROFILES_PATH}")


if __name__ == "__main__":
    main()
