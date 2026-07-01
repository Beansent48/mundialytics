#!/usr/bin/env python3
"""One-off calibration: map PlayerStrengthModel.team_strength() squad indices
(attack_index, defense_index, 0-100) onto AttackDefenseModel's log-scale
attack/defense parameters, plus per-market event-lambda scaling.

METHOD: range-based, not a per-club regression. See the module docstring in
src/mundialytics/statistical_core/squadlab/calibration_constants.py for why
the original per-club approach (reconstruct each real club's actual best-11,
regress its team_strength() against that same club's real
AttackDefenseModel.team_params() row) was abandoned: two attempts (against
the live 2021-2026 model, and against a temporary model fit on StatsBomb's
older, denser-coverage Big5 seasons) both produced R^2 < 0.1 because
player_profiles_with_positions.csv has no season column, so "reconstruct
team X's current best-11" silently mixes players from incompatible eras.

Instead: map the OBSERVED RANGE of achievable squad strength (weakest to
strongest XI draftable from the current player pool) onto the OBSERVED
RANGE of real AttackDefenseModel parameters (5th-95th percentile across all
fitted teams) via a two-point linear map. Guarantees correct ordering and a
realistic scale; does not claim per-club precision.

This is a one-off, human-reviewed fit -- NOT refit at runtime. Review the
printed values, then hand-paste into squadlab/calibration_constants.py.

Run:
    python scripts/fit_squad_lambda_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.attack_defense_model import AttackDefenseModel
from mundialytics.statistical_core.player_strength import PlayerStrengthModel
from mundialytics.statistical_core.schemas import canonical_name

POSITION_SLOTS = {"Goalkeeper": 1, "Defender": 4, "Midfielder": 3, "Forward": 3}
MARKETS = ["shots", "sot", "corners", "fouls", "yellow_cards"]
ATTACK_MARKETS = {"shots", "sot", "corners"}
MIN_MATCHES_FOR_DRAFT_POOL = 10  # exclude cameo-noise players from the best/worst XI


def two_point_linear(x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> tuple[float, float]:
    slope = (y_hi - y_lo) / (x_hi - x_lo)
    intercept = y_lo - slope * x_lo
    return slope, intercept


def extreme_squads(model: PlayerStrengthModel) -> tuple[list, list]:
    """Weakest and strongest XI draftable from the current player pool
    (top/bottom `overall` per position, same slot counts as Draft mode)."""
    best, worst = [], []
    for pos, n in POSITION_SLOTS.items():
        cands = [p for p in model.profiles_.values() if p.position == pos and p.matches >= MIN_MATCHES_FOR_DRAFT_POOL]
        cands_by_overall = sorted(cands, key=lambda p: -p.overall)
        best.extend(cands_by_overall[:n])
        worst.extend(sorted(cands, key=lambda p: p.overall)[:n])
    return best, worst


def main() -> None:
    print("Loading match-results data and fitting AttackDefenseModel...")
    df_clubs = pd.read_csv(ROOT / "data/processed/foundation_big5_multi_season.csv")
    ad_model = AttackDefenseModel()
    ad_model.fit(df_clubs)
    params = ad_model.team_params()
    print(f"  fitted {ad_model.n_teams_} teams across {ad_model.n_leagues_} leagues")

    print("Loading PlayerStrengthModel...")
    strength_model = PlayerStrengthModel()
    strength_model.fit()

    best_squad, worst_squad = extreme_squads(strength_model)
    best_str = strength_model.team_strength(best_squad)
    worst_str = strength_model.team_strength(worst_squad)
    ai_lo, ai_hi = worst_str["attack_index"], best_str["attack_index"]
    di_lo, di_hi = worst_str["defense_index"], best_str["defense_index"]
    print(f"\nAchievable attack_index range: {ai_lo:.2f} (weakest) - {ai_hi:.2f} (best all-star XI)")
    print(f"Achievable defense_index range: {di_lo:.2f} (weakest) - {di_hi:.2f} (best all-star XI)")

    a_lo, a_hi = params["attack"].quantile(0.05), params["attack"].quantile(0.95)
    d_lo, d_hi = params["defense"].quantile(0.05), params["defense"].quantile(0.95)
    a_min, a_max = params["attack"].min(), params["attack"].max()
    d_min, d_max = params["defense"].min(), params["defense"].max()

    goal_attack_slope, goal_attack_intercept = two_point_linear(ai_lo, ai_hi, a_lo, a_hi)
    goal_defense_slope, goal_defense_intercept = two_point_linear(di_lo, di_hi, d_lo, d_hi)

    print(f"\nReal attack_param 5th-95th percentile: [{a_lo:.4f}, {a_hi:.4f}] (full range [{a_min:.4f}, {a_max:.4f}])")
    print(f"Real defense_param 5th-95th percentile: [{d_lo:.4f}, {d_hi:.4f}] (full range [{d_min:.4f}, {d_max:.4f}])")

    # Per-market event rates: each real team's average "for" rate (home+away combined)
    df_clubs["home_c"] = df_clubs["home_team"].map(canonical_name)
    df_clubs["away_c"] = df_clubs["away_team"].map(canonical_name)
    teams = sorted(set(df_clubs["home_c"]) | set(df_clubs["away_c"]))
    event_rows = []
    for t in teams:
        rate = {}
        for mk in MARKETS:
            vals = pd.concat([
                df_clubs.loc[df_clubs["home_c"] == t, f"home_{mk}"],
                df_clubs.loc[df_clubs["away_c"] == t, f"away_{mk}"],
            ])
            rate[mk] = vals.mean()
        event_rows.append({"team": t, **rate})
    event_df = pd.DataFrame(event_rows)

    print("\n--- Event lambda calibration (range-based, attack_index or defense_index as basis) ---")
    event_constants: dict[str, dict] = {}
    for mk in MARKETS:
        lo, hi = event_df[mk].quantile(0.05), event_df[mk].quantile(0.95)
        mn, mx = event_df[mk].min(), event_df[mk].max()
        if mk in ATTACK_MARKETS:
            slope, intercept = two_point_linear(ai_lo, ai_hi, lo, hi)
            basis = "attack"
        else:
            slope, intercept = two_point_linear(di_lo, di_hi, lo, hi)
            basis = "defense"
        event_constants[mk] = {"basis": basis, "slope": slope, "intercept": intercept, "clip_min": mn, "clip_max": mx}
        print(f"  {mk:14s} basis={basis:8s} slope={slope:.4f} intercept={intercept:.4f} clip=[{mn:.2f},{mx:.2f}]")

    print("\n--- Paste into squadlab/calibration_constants.py ---")
    print(f"GOAL_ATTACK_SLOPE: float = {goal_attack_slope:.6f}")
    print(f"GOAL_ATTACK_INTERCEPT: float = {goal_attack_intercept:.6f}")
    print(f"GOAL_DEFENSE_SLOPE: float = {goal_defense_slope:.6f}")
    print(f"GOAL_DEFENSE_INTERCEPT: float = {goal_defense_intercept:.6f}")
    print(f"ATTACK_PARAM_CLIP: tuple[float, float] = ({a_min:.4f}, {a_max:.4f})")
    print(f"DEFENSE_PARAM_CLIP: tuple[float, float] = ({d_min:.4f}, {d_max:.4f})")
    print("EVENT_CALIBRATION: dict[str, dict[str, float | str]] = {")
    for mk, c in event_constants.items():
        print(f'    "{mk}": {{"basis": "{c["basis"]}", "slope": {c["slope"]:.4f}, '
              f'"intercept": {c["intercept"]:.4f}, "clip_min": {c["clip_min"]:.2f}, "clip_max": {c["clip_max"]:.2f}}},')
    print("}")


if __name__ == "__main__":
    main()
