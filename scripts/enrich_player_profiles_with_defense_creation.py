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
SHOT_EVENTS_PATH = ROOT / "data/external/advanced/statsbomb/statsbomb_shot_events.csv"


def main() -> None:
    if not PROFILES_PATH.exists():
        raise FileNotFoundError(f"Profiles CSV not found: {PROFILES_PATH}")
    if not PER_MATCH_PATH.exists():
        raise FileNotFoundError(f"Per-match events CSV not found: {PER_MATCH_PATH}")

    profiles = pd.read_csv(PROFILES_PATH)
    pm = pd.read_csv(PER_MATCH_PATH)

    # Ball-progression columns (added 2026-07-02) may be absent if
    # player_match_events.csv predates the statsbomb.py adapter change -- fill
    # zero so re-running against an older per-match file doesn't KeyError.
    PROGRESSION = ["progressive_passes", "progressive_carries", "passes_into_final_third",
                   "passes_into_box", "through_balls", "carries", "successful_dribbles",
                   "crosses", "cut_backs"]
    # Raw counts summed here, then turned into RATES below (not simple per-match).
    RAW_EXTRA = ["passes_under_pressure", "complete_passes_under_pressure",
                 "aerials_won", "aerials_lost"]
    for c in PROGRESSION + RAW_EXTRA:
        if c not in pm.columns:
            pm[c] = 0

    agg = pm.groupby("player").agg(
        defense_creation_matches=("match_id", "nunique"),
        duels_won=("duels_won", "sum"), duels_lost=("duels_lost", "sum"),
        dribbled_past=("dribbled_past", "sum"), clearances=("clearances", "sum"),
        blocks=("blocks", "sum"), interceptions=("interceptions", "sum"),
        key_passes=("key_passes", "sum"), passes=("passes", "sum"),
        complete_passes=("complete_passes", "sum"),
        **{c: (c, "sum") for c in PROGRESSION + RAW_EXTRA},
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
    for c in PROGRESSION:
        agg[f"{c}_per_match"] = agg[c] / matches_safe

    # Composure + aerial rates (fallbacks: neutral 0.75 / 0.5), plus aerial-win volume.
    pressured = agg["passes_under_pressure"]
    agg["pass_completion_under_pressure"] = (
        agg["complete_passes_under_pressure"] / pressured.clip(lower=1)).where(pressured > 0, 0.75)
    aerial_total = agg["aerials_won"] + agg["aerials_lost"]
    agg["aerial_win_rate"] = (
        agg["aerials_won"] / aerial_total.clip(lower=1)).where(aerial_total > 0, 0.5)
    agg["aerials_won_per_match"] = agg["aerials_won"] / matches_safe

    keep = agg[[
        "player", "defense_creation_matches", "duel_win_rate", "dribbled_past_per_match",
        "clearances_per_match", "blocks_per_match", "interceptions_per_match",
        "key_passes_per_match", "passes_per_match", "pass_completion",
        "pass_completion_under_pressure", "aerial_win_rate", "aerials_won_per_match",
    ] + [f"{c}_per_match" for c in PROGRESSION]]

    # Idempotent: drop any previously-enriched columns before re-merging, so a
    # second run refreshes values instead of creating _x/_y suffix collisions
    # (the career file already carries the earlier defense/creation columns).
    enrich_cols = [c for c in keep.columns if c != "player"]
    profiles = profiles.drop(columns=[c for c in enrich_cols if c in profiles.columns])
    before_cols = set(profiles.columns)
    profiles = profiles.merge(keep, on="player", how="left")
    profiles["defense_creation_matches"] = profiles["defense_creation_matches"].fillna(0).astype(int)
    profiles["duel_win_rate"] = profiles["duel_win_rate"].fillna(0.5)
    profiles["aerial_win_rate"] = profiles["aerial_win_rate"].fillna(0.5)
    for c in (["dribbled_past_per_match", "clearances_per_match", "blocks_per_match",
               "interceptions_per_match", "key_passes_per_match", "passes_per_match",
               "aerials_won_per_match"]
              + [f"{c}_per_match" for c in PROGRESSION]):
        profiles[c] = profiles[c].fillna(0.0)
    profiles["pass_completion"] = profiles["pass_completion"].fillna(0.75)
    profiles["pass_completion_under_pressure"] = profiles["pass_completion_under_pressure"].fillna(0.75)

    # Finishing skill = (goals - xG) per shot (overperformance vs expected).
    # Penalties ARE included (2026-07-02, user call: converting penalties is a
    # real skill and excluding them unfairly docked penalty-taking scorers like
    # Ronaldo). From the shot-event file (per-shot xG). finishing_shots is
    # carried so player_strength.py can shrink low-volume finishing toward
    # neutral (a 5-shot hot streak isn't real finishing skill).
    if SHOT_EVENTS_PATH.exists():
        sh = pd.read_csv(SHOT_EVENTS_PATH, low_memory=False)
        sh["xg"] = pd.to_numeric(sh["xg"], errors="coerce").fillna(0.0)
        sh["is_goal"] = sh["is_goal"].astype(str).str.lower().isin(["true", "1", "1.0"])
        fin = sh.groupby("player").agg(
            np_goals=("is_goal", "sum"), np_xg=("xg", "sum"), finishing_shots=("xg", "size"),
        ).reset_index()
        fin["finishing_per_shot"] = (fin["np_goals"] - fin["np_xg"]) / fin["finishing_shots"].clip(lower=1)
        profiles = profiles.drop(columns=[c for c in ("finishing_per_shot", "finishing_shots")
                                          if c in profiles.columns])
        profiles = profiles.merge(fin[["player", "finishing_per_shot", "finishing_shots"]],
                                  on="player", how="left")
    for c, dflt in (("finishing_per_shot", 0.0), ("finishing_shots", 0.0)):
        if c not in profiles.columns:
            profiles[c] = dflt
        profiles[c] = profiles[c].fillna(dflt)

    # Granular position (StatsBomb-era only): most-common raw StatsBomb position
    # per player -> coarse bucket. Drives which roles are candidates in
    # player_strength.py. Empty for players with no per-match events (FBref
    # modern) -> they fall back to ALL roles of their 4-bucket position.
    GRANULAR_MAP = {
        "Goalkeeper": "Portero",
        "Left Center Back": "Central", "Right Center Back": "Central", "Center Back": "Central",
        "Left Back": "Lateral", "Right Back": "Lateral",
        "Left Wing Back": "Lateral", "Right Wing Back": "Lateral",
        "Center Defensive Midfield": "Pivote", "Left Defensive Midfield": "Pivote",
        "Right Defensive Midfield": "Pivote",
        "Center Midfield": "Mediocentro", "Left Center Midfield": "Mediocentro",
        "Right Center Midfield": "Mediocentro", "Left Midfield": "Mediocentro",
        "Right Midfield": "Mediocentro",
        "Center Attacking Midfield": "Mediapunta", "Left Attacking Midfield": "Mediapunta",
        "Right Attacking Midfield": "Mediapunta",
        "Left Wing": "Extremo", "Right Wing": "Extremo",
        "Center Forward": "Delantero", "Left Center Forward": "Delantero",
        "Right Center Forward": "Delantero", "Secondary Striker": "Delantero",
    }
    if "position" in pm.columns:
        gpos = pm[pm["position"] != "Substitute"].groupby("player")["position"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else None)
        gmap = gpos.map(GRANULAR_MAP).dropna().to_dict()
    else:
        gmap = {}
    profiles = profiles.drop(columns=[c for c in ("granular_position",) if c in profiles.columns])
    profiles["granular_position"] = profiles["player"].map(gmap).fillna("")

    new_cols = sorted(set(profiles.columns) - before_cols)
    matched = int((profiles["defense_creation_matches"] > 0).sum())
    print(f"Matched {matched}/{len(profiles)} players to Big5 per-match defense/creation data")
    print(f"New columns: {new_cols}")

    profiles.to_csv(PROFILES_PATH, index=False)
    print(f"Wrote {PROFILES_PATH}")


if __name__ == "__main__":
    main()
