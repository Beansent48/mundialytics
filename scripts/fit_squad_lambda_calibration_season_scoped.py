#!/usr/bin/env python3
"""Re-attempt the squad-lambda calibration as a precise per-club regression,
now that data/processed/player_profiles_by_season.csv exists (real
per-season player rosters, not career-aggregated).

Two earlier attempts at exactly this (see calibration_constants.py's
docstring) failed with R^2 < 0.1 because reconstructing "team X's current
best-11" from the OLD career-aggregated player file silently mixed players
from incompatible eras. This version reconstructs "team X's best-11 for
SEASON Y specifically" from the new season-split file, and regresses it
against an AttackDefenseModel fit on ONLY that season's matches (not the
whole multi-season live model) — so both sides of the regression are
genuinely from the same season.

If R^2 is now good, this REPLACES the range-based calibration in
calibration_constants.py with a precise per-club fit. If not, the
range-based approach (already in production) stays as-is — this script
only prints a report, it doesn't overwrite anything automatically.

Run:
    python scripts/fit_squad_lambda_calibration_season_scoped.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.attack_defense_model import AttackDefenseModel
from mundialytics.statistical_core.schemas import canonical_name
from mundialytics.statistical_core.player_strength import (
    ANCHOR_CURVES, DUEL_SHRINKAGE_MATCHES, POSITION_ATTACK_WEIGHT, POSITION_CREATION_WEIGHT,
    POSITION_DEFENSE_WEIGHT, SHRINKAGE_MATCHES, _build_gk_scores,
)

# Goalkeepers get near-floor scores from the generic tackles/pressures
# def_weights formula (they don't tackle/press) -- exactly the bug already
# fixed for the live rating system via _build_gk_scores. Reuse the same
# real save%/goals-conceded/clean-sheet score here instead, keyed by
# lowercased player name (goalkeeper_match_stats.csv's own convention).
GK_SCORES = _build_gk_scores()

BIG5_COMP_IDS = {9: "1. Bundesliga", 11: "La Liga", 7: "Ligue 1", 2: "Premier League", 12: "Serie A"}
POSITION_SLOTS = {"Goalkeeper": 1, "Defender": 4, "Midfielder": 3, "Forward": 3}
MIN_TEAM_MATCHES = 15  # minimum matches for a team within one season's AD fit to trust its params


def _discover_season_matches() -> pd.DataFrame:
    rows = []
    for comp_id, comp_name in BIG5_COMP_IDS.items():
        for f in (ROOT / f"data/raw/statsbomb/open-data/data/matches/{comp_id}").glob("*.json"):
            for m in json.loads(f.read_text(encoding="utf-8")):
                if m.get("home_score") is None or m.get("away_score") is None:
                    continue
                rows.append({
                    "match_id": m["match_id"], "date": m["match_date"], "competition": comp_name,
                    "season": m["season"]["season_name"],
                    "home_team": m["home_team"]["home_team_name"], "away_team": m["away_team"]["away_team_name"],
                    "home_goals": m["home_score"], "away_goals": m["away_score"],
                })
    return pd.DataFrame(rows)


def _off_def_creation_scores(row: pd.Series) -> tuple[float, float, float]:
    """Replicates PlayerStrengthModel.fit()'s off_score/def_score/creation_score
    exactly (same weights, same anchor curves, same shrinkage) for one
    season-row. The season file is Big5-only by construction, so its own
    "matches" column already is the correct credibility base for the
    defense-quality/creation stats too (no separate defense_creation_matches
    needed here, unlike the career file which mixes broader competitions)."""
    off_weights = {"goals_per_match": 0.30, "assists_per_match": 0.30,
                    "xg_per_match": 0.25, "big_chance_miss_rate": 0.15}
    def_weights = {"duel_win_rate": 0.50, "dribbled_past_per_match": 0.20,
                    "interceptions_per_match": 0.12,
                    "clearances_per_match": 0.08, "blocks_per_match": 0.05,
                    "fouls_per_match": 0.03, "yellow_cards_per_match": 0.02}
    creation_weights = {"key_passes_per_match": 0.65, "pass_completion": 0.35}
    credibility = row["matches"] / (row["matches"] + SHRINKAGE_MATCHES)
    credibility_duel = row["matches"] / (row["matches"] + DUEL_SHRINKAGE_MATCHES)

    def _scored(stat: str) -> float:
        xs, ys = zip(*ANCHOR_CURVES[stat])
        raw = float(np.interp(row[stat], xs, ys))
        cred = credibility_duel if stat == "duel_win_rate" else credibility
        return 50.0 + (raw - 50.0) * cred

    off_num = sum(_scored(s) * w for s, w in off_weights.items())
    def_num = sum(_scored(s) * w for s, w in def_weights.items())
    creation_num = sum(_scored(s) * w for s, w in creation_weights.items())
    return (off_num / sum(off_weights.values()), def_num / sum(def_weights.values()),
            creation_num / sum(creation_weights.values()))


def _team_strength(squad: pd.DataFrame) -> tuple[float, float]:
    total_atk = total_def = total_w_atk = total_w_def = 0.0
    for _, p in squad.iterrows():
        off_score, def_score, creation_score = _off_def_creation_scores(p)
        if p["position"] == "Goalkeeper":
            def_score = GK_SCORES.get(str(p["player"]).lower(), 50.0)
        atk_w = POSITION_ATTACK_WEIGHT.get(p["position"], 0.40)
        def_w = POSITION_DEFENSE_WEIGHT.get(p["position"], 0.35)
        cre_w = POSITION_CREATION_WEIGHT.get(p["position"], 0.25)
        total_atk += off_score * atk_w + creation_score * cre_w
        total_def += def_score * def_w
        total_w_atk += atk_w + cre_w
        total_w_def += def_w
    return total_atk / max(total_w_atk, 1e-6), total_def / max(total_w_def, 1e-6)


def main() -> None:
    print("Loading season-split player profiles...")
    profiles = pd.read_csv(ROOT / "data/processed/player_profiles_by_season.csv")
    profiles["team_c"] = profiles["team"].map(canonical_name)
    # Season file stores passes_per_match/complete_passes_per_match
    # separately (not a precomputed ratio) -- derive pass_completion here to
    # match the career-file enrichment column used at live rating time.
    profiles["pass_completion"] = (
        profiles["complete_passes_per_match"] / profiles["passes_per_match"].clip(lower=0.01)
    ).fillna(0.75).clip(0, 1)

    print("Loading real match results for AttackDefenseModel per-season fits...")
    season_matches = _discover_season_matches()

    rows = []
    n_groups_tried = 0
    for (comp, season), group_matches in season_matches.groupby(["competition", "season"]):
        if len(group_matches) < 20:
            continue
        ad = AttackDefenseModel()
        ad.fit(group_matches)
        params = ad.team_params().set_index("team")

        season_profiles = profiles[(profiles["competition"] == comp) & (profiles["season"] == season)]
        for team_c, team_group in season_profiles.groupby("team_c"):
            n_groups_tried += 1
            if team_c not in params.index or ad.match_counts_.get(team_c, 0) < MIN_TEAM_MATCHES:
                continue
            squad = []
            for pos, n in POSITION_SLOTS.items():
                cands = team_group[team_group["position"] == pos].sort_values("matches", ascending=False)
                squad.append(cands.head(n))
            squad_df = pd.concat(squad)
            if len(squad_df) < 8:
                continue
            attack_idx, defense_idx = _team_strength(squad_df)
            real = params.loc[team_c]
            rows.append({
                "team": team_c, "competition": comp, "season": season,
                "attack_idx": attack_idx, "defense_idx": defense_idx,
                "attack_param": real["attack"], "defense_param": real["defense"],
                "ad_matches": ad.match_counts_.get(team_c, 0),
            })

    result = pd.DataFrame(rows)
    print(f"\nMatched {len(result)} (team, season) rosters out of {n_groups_tried} candidates")

    if len(result) < 10:
        print("Too few matched rosters to fit a meaningful regression.")
        return

    slope_a, int_a = np.polyfit(result["attack_idx"], result["attack_param"], 1)
    pred_a = slope_a * result["attack_idx"] + int_a
    r2_a = 1 - ((result["attack_param"] - pred_a) ** 2).sum() / ((result["attack_param"] - result["attack_param"].mean()) ** 2).sum()

    slope_d, int_d = np.polyfit(result["defense_idx"], result["defense_param"], 1)
    pred_d = slope_d * result["defense_idx"] + int_d
    r2_d = 1 - ((result["defense_param"] - pred_d) ** 2).sum() / ((result["defense_param"] - result["defense_param"].mean()) ** 2).sum()

    print(f"\nattack:  slope={slope_a:.6f} intercept={int_a:.6f}  R^2={r2_a:.3f}")
    print(f"defense: slope={slope_d:.6f} intercept={int_d:.6f}  R^2={r2_d:.3f}")

    print("\n--- Sample of matched rosters (sorted by attack_idx) ---")
    print(result.sort_values("attack_idx", ascending=False).head(15).to_string())

    # Sanity check with min-sample filter, mirroring earlier session's diagnostics
    for thresh in (0, 20, 30):
        g = result[result["ad_matches"] >= thresh]
        if len(g) < 10:
            continue
        s, i = np.polyfit(g["attack_idx"], g["attack_param"], 1)
        p = s * g["attack_idx"] + i
        r2 = 1 - ((g["attack_param"] - p) ** 2).sum() / ((g["attack_param"] - g["attack_param"].mean()) ** 2).sum()
        print(f"ad_matches>={thresh:3d}: n={len(g):4d} attack R^2={r2:.3f}")


if __name__ == "__main__":
    main()
