from __future__ import annotations

"""Historical team catalog for SquadLab's "Champions histórica".

For every (club, season) in the season-split player file that can field a full
XI, reconstruct that XI, run it through SquadLab's OWN player->strength bridge,
and store the resulting attack/defense params plus an Elo-equivalent so the
team can be dropped straight into the validated European tournament simulator.

Why the bridge (and not each season's real AttackDefenseModel params): the
bridge is ONE global mapping from player quality to strength, so Barcelona
2010/11 and Milan 2015/16 land on a COMMON scale — exactly what a cross-era
tournament needs (per-league/era AD fits are not comparable to each other).

Output: data/processed/historical_teams.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mundialytics.statistical_core.squadlab import calibration_constants as CC  # noqa: E402
from fit_squad_lambda_calibration_season_scoped import (  # noqa: E402
    POSITION_SLOTS, _off_def_creation_scores, _team_strength)

OUT = ROOT / "data/processed/historical_teams.csv"
SEASON_FILE = ROOT / "data/processed/player_profiles_by_season.csv"
# Elo-equivalent: the tournament sim maps log-lambda = c + hfa + b*(dElo/400),
# while the squad bridge gives log-lambda = mu + ha + attack - defense_opp.
# Matching the two means dElo = 400/b * d(strength). Centre on a realistic Elo.
ELO_CENTRE, ELO_B = 1780.0, 0.7394


def main() -> None:
    df = pd.read_csv(SEASON_FILE)
    df["pass_completion"] = (df["complete_passes_per_match"]
                             / df["passes_per_match"].clip(lower=0.01)).fillna(0.75).clip(0, 1)
    df["finishing_per_shot"] = (df["goals_per_match"]
                                / df["shots_per_match"].clip(lower=0.01)).fillna(0.0).clip(0, 1)

    rows = []
    for (team, season), g in df.groupby(["team", "season"]):
        if not all(len(g[g["position"] == p]) >= n for p, n in POSITION_SLOTS.items()):
            continue
        xi = pd.concat([g[g["position"] == pos].sort_values("matches", ascending=False).head(n)
                        for pos, n in POSITION_SLOTS.items()])
        atk_idx, def_idx = _team_strength(xi)
        atk = float(np.clip(CC.GOAL_ATTACK_SLOPE * atk_idx + CC.GOAL_ATTACK_INTERCEPT,
                            *CC.ATTACK_PARAM_CLIP))
        dfn = float(np.clip(CC.GOAL_DEFENSE_SLOPE * def_idx + CC.GOAL_DEFENSE_INTERCEPT,
                            *CC.DEFENSE_PARAM_CLIP))
        # headline players (most-played of the XI) for the UI
        stars = ", ".join(xi.sort_values("matches", ascending=False)["player"].head(3))
        rows.append({
            "team": team, "season": season,
            "label": f"{team} {season.split('/')[0]}",
            "atk_idx": round(atk_idx, 2), "def_idx": round(def_idx, 2),
            "attack_param": round(atk, 4), "defense_param": round(dfn, 4),
            "strength": round((atk + dfn) / 2, 4),
            "n_players": len(g), "stars": stars,
        })

    cat = pd.DataFrame(rows)
    # Elo-equivalent on a common scale (differences are what the sim uses)
    cat["elo"] = (ELO_CENTRE + (400.0 / ELO_B) * (cat["strength"] - cat["strength"].mean())).round(0)
    cat = cat.sort_values("strength", ascending=False).reset_index(drop=True)
    cat.to_csv(OUT, index=False)

    print(f"WROTE {OUT}: {len(cat)} historical teams "
          f"({cat.team.nunique()} clubs, {cat.season.nunique()} seasons)")
    print(f"Elo range {cat.elo.min():.0f}-{cat.elo.max():.0f}\n")
    print("=== TOP 15 EQUIPOS HISTORICOS ===")
    print(cat.head(15)[["label", "strength", "elo", "stars"]].to_string(index=False))
    print("\n=== ejemplos concretos ===")
    for lbl in ["Barcelona 2010", "Barcelona 2014", "Real Madrid 2016", "Bayern Munich 2023"]:
        r = cat[cat.label == lbl]
        if len(r):
            r = r.iloc[0]
            print(f"  {lbl:22s} fuerza {r.strength:+.3f} elo {r.elo:.0f}  ({r.stars[:52]})")


if __name__ == "__main__":
    main()
