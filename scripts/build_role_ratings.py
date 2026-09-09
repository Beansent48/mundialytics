#!/usr/bin/env python3
"""Turn the per-player sub-stat profile into a role-based rating.

THE THREE IDEAS, in order:

1. ELITE ANCHOR (the "Malen problem"). A within-position percentile makes a
   merely-good season read as elite — 0.80 is the 80th percentile, which sounds
   top but is a normal starter. So each percentile is passed through an
   emphasis curve `p ** GAMMA` that only rewards the genuine top: 0.97 stays
   near 1, 0.80 falls to ~0.55, 0.50 to ~0.18. To be elite at a skill you must
   be near the very best AT IT, not merely above average.

2. VALUE BY ROLE. Not every skill makes a player great, and it depends on the
   role. Each role is an expert weight-vector over the sub-stats: what MATTERS
   for that role. The cheap, easily-farmed skills (crosses, interception
   volume, set-piece delivery, high regains) carry small weights everywhere, so
   nobody reaches 90 by maxing one of them — the user's rule, "no 90 por
   centros". The heavy weights sit on the skills that separate the elite (xG,
   xA, dribbling, duels).

3. BEST FIT, no dilution. A player is scored under EVERY role his position can
   play and keeps his BEST one. A poacher is judged as a poacher (finishing is
   everything, his weak creation is irrelevant); a false-9 as whichever of
   striker/playmaker flatters his real strengths. Versatility is rewarded, a
   specialist is never diluted by skills his role does not need.

The overall is calibrated (the "mixto" anchor) so only the genuine elite of each
position clears 90 and normal cards cap at 92; primes/icons live above and are
built elsewhere.

Run: .venv/Scripts/python.exe scripts/build_role_ratings.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MICRO = ROOT / "data/processed/player_micro_stats.csv"
OUT = ROOT / "data/processed/player_role_ratings.csv"
from build_player_attack_fbref import LEAGUE_W  # noqa: E402

# ── the role dictionary (weights /100, approved v3) ────────────────────────────
ROLE_WEIGHTS: dict[str, dict[str, float]] = {
    "Killer":             {"FIN": 42, "BOX": 22, "CONV": 12, "XA": 8, "SETP_FIN": 6, "AER": 5, "HIGHREG": 5},
    "Target man":         {"FIN": 28, "AER": 18, "BOX": 16, "SETP_FIN": 12, "RETEN": 10, "CONV": 8, "FOULDRAWN": 8},
    "Delantero completo": {"FIN": 30, "XA": 17, "DRIB": 14, "BOX": 13, "CONV": 11, "PROG": 10, "AER": 5},
    "Falso 9":            {"XA": 26, "FIN": 18, "THRU": 16, "DRIB": 12, "FOULDRAWN": 12, "RETEN": 10, "HIGHREG": 6},
    "Extremo":            {"DRIB": 28, "FIN": 30, "CROSS": 12, "PROG": 12, "BOX": 12, "FOULDRAWN": 6},
    "Extremo interior":   {"FIN": 28, "DRIB": 22, "XA": 18, "BOX": 14, "CONV": 10, "HIGHREG": 8},
    "Mediapunta":         {"XA": 26, "BIGC": 16, "THRU": 14, "FIN": 14, "DRIB": 12, "PROG": 10, "SETP_TAKE": 8},
    "Creador":            {"XA": 22, "THRU": 18, "PROG": 18, "RETEN": 14, "DRIB": 10, "LONG": 10, "SETP_TAKE": 8},
    "Box-to-box":         {"PROG": 16, "FIN": 12, "DUEL": 12, "HIGHREG": 12, "XA": 12, "RECOV": 10, "RETEN": 11, "GA_ON": 10, "AER": 5},
    "Pivote organizador": {"RETEN": 20, "PROG": 18, "LONG": 14, "DUEL": 12, "GA_ON": 12, "INT": 10, "RECOV": 8, "XA": 6},
    "Destructor":         {"DUEL": 26, "INT": 18, "RECOV": 16, "HIGHREG": 12, "GA_ON": 12, "AER": 10, "RETEN": 6},
    "Central stopper":    {"DUEL": 22, "AER": 22, "GA_ON": 20, "INT": 14, "RETEN": 8, "PROG": 8, "SETP_FIN": 6},
    "Central de salida":  {"RETEN": 18, "PROG": 16, "GA_ON": 16, "DUEL": 15, "AER": 12, "INT": 9, "LONG": 8, "THRU": 6},
    "Lateral ofensivo":   {"PROG": 16, "XA": 14, "DRIB": 14, "GA_ON": 14, "CROSS": 12, "DUEL": 12, "RETEN": 10, "AER": 8},
    "Lateral defensivo":  {"DUEL": 22, "GA_ON": 18, "INT": 14, "RETEN": 14, "RECOV": 12, "AER": 10, "PROG": 10},
    "Portero":            {"SHOTSTOP": 55, "AERIAL_GK": 18, "DISTRIB": 15, "SWEEP": 12},
}
# which roles a position may be scored under. Midfielder and Forward pool the
# outfield-attacking roles (the winger/attacking-mid boundary is fuzzy); the
# best-fit self-selects the right one. Defender and Goalkeeper stand apart.
# a forward is scored only under attacking roles — never a deep pivote; a
# midfielder can be advanced (Mediapunta/Extremo/Falso 9) or deep, but the
# best-fit self-selects. The boundary they share is the attacking-mid band.
FWD_ROLES = ["Killer", "Target man", "Delantero completo", "Falso 9",
             "Extremo", "Extremo interior", "Mediapunta"]
MID_ROLES = ["Mediapunta", "Creador", "Box-to-box", "Pivote organizador",
             "Destructor", "Extremo", "Extremo interior", "Falso 9"]
DEF_ROLES = ["Central stopper", "Central de salida", "Lateral ofensivo", "Lateral defensivo"]
ATT_MID = FWD_ROLES + [r for r in MID_ROLES if r not in FWD_ROLES]
CANDIDATES = {"Forward": FWD_ROLES, "Midfielder": MID_ROLES, "Defender": DEF_ROLES,
              "Goalkeeper": ["Portero"]}

# axis groups for the card face / simulator (offensive / defensive / creation)
AXIS_GROUPS = {
    "attack":   ["FIN", "BOX", "CONV", "DRIB", "SETP_FIN"],
    "creation": ["XA", "BIGC", "THRU", "CROSS", "SETP_TAKE", "PROG", "LONG", "RETEN", "FOULDRAWN"],
    "defense":  ["DUEL", "AER", "INT", "RECOV", "HIGHREG", "GA_ON"],
    "gk":       ["SHOTSTOP", "AERIAL_GK", "SWEEP", "DISTRIB"],
}

# ── calibration (the "mixto" anchor: objective, then tuned to the named elite) ──
GAMMA = 2.3                 # emphasis: only the genuine top of a skill counts
# role composite (0..1) -> overall. Interp calibrated so the elite clear 90 and
# a normal starter sits 75-82. Validated against Kane/Mbappé/Haaland (>90),
# Malen (mid-80s), a pure crosser (<82).
# steeper at the top so the true elite pull AWAY from the merely very-good
# (the user's rule: compress the pack down, keep the gap). Median stays ~75.
OVR_XS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.63, 0.70]
OVR_YS = [72.0, 76.0, 79.0, 82.0, 85.0, 89.0, 95.0]
# NO hard cap here: the overall stays CONTINUOUS (the elite reach ~93-95) and
# the card's soft_cap_actual (a tanh) brings the ceiling to 92 PROPORTIONALLY,
# so the top keeps its gaps (Kane over Haaland over ...) instead of a flat wall.
LEAGUE_PTS = 2.0           # how much a full sd of league strength moves the composite-rating


def emphasis(p):
    return np.power(np.clip(p, 0.0, 1.0), GAMMA)


GROUP_OF = {s: g for g, members in AXIS_GROUPS.items() for s in members}


def role_grouped(prof: pd.Series, weights: dict):
    """Score a profile under one role, DECOMPOSED by axis group.

    Returns (overall 0..1, coverage, gnum, gden) where gnum/gden are the
    role-weighted emphasis sum and weight per group (attack/creation/defense/
    gk). The overall is sum(gnum)/sum(gden) — so the group scores gnum/gden ARE
    the axes and the overall is their weighted average: better axes always mean
    a better overall. Weights renormalise over present sub-stats (a missing
    column costs coverage, not a zero)."""
    gnum = {"attack": 0.0, "creation": 0.0, "defense": 0.0, "gk": 0.0}
    gden = {"attack": 0.0, "creation": 0.0, "defense": 0.0, "gk": 0.0}
    cov_w = cov_tot = 0.0
    for s, w in weights.items():
        cov_tot += w
        v = prof.get(s, np.nan)
        if pd.notna(v):
            g = GROUP_OF.get(s, "creation")
            gnum[g] += w * emphasis(float(v))
            gden[g] += w
            cov_w += w
    den = sum(gden.values())
    if den <= 0:
        return 0.0, 0.0, gnum, gden
    return sum(gnum.values()) / den, cov_w / cov_tot, gnum, gden


def build() -> pd.DataFrame:
    d = pd.read_csv(MICRO)
    rows = []
    for r in d.itertuples(index=False):
        prof = r._asdict()
        pos = str(prof["position"])
        cands = CANDIDATES.get(pos, ATT_MID)
        best = None
        for role in cands:
            comp, cov, gnum, gden = role_grouped(prof, ROLE_WEIGHTS[role])
            comp_eff = comp if cov >= 0.5 else comp * cov
            if best is None or comp_eff > best[0]:
                best = (comp_eff, role, comp, gnum, gden)
        comp_eff, best_role, comp, gnum, gden = best
        # the axes ARE the role-weighted group scores; where the role does not
        # value a group, fall back to a flat mean so the bar is still shown
        ax = {}
        for a, members in AXIS_GROUPS.items():
            if gden.get(a, 0.0) > 0:
                ax[a] = gnum[a] / gden[a]
            else:
                vals = [emphasis(float(prof[m])) for m in members if pd.notna(prof.get(m, np.nan))]
                ax[a] = float(np.mean(vals)) if vals else np.nan
        # league nudges the composite; overall stays continuous (card caps to 92)
        c = float(np.clip(comp + LEAGUE_W * LEAGUE_PTS / 100.0 * float(prof.get("league_z", 0.0)), 0, 1))
        rows.append({
            "player": prof["player"], "position": pos, "league": prof["league"],
            "role": best_role, "role_fit": round(comp_eff, 3), "n90": prof["n90"],
            "overall": round(float(np.interp(c, OVR_XS, OVR_YS)), 1),
            "attack": round(ax["attack"], 4) if pd.notna(ax["attack"]) else np.nan,
            "creation": round(ax["creation"], 4) if pd.notna(ax["creation"]) else np.nan,
            "defense": round(ax["defense"], 4) if pd.notna(ax["defense"]) else np.nan,
            "gk": round(ax["gk"], 4) if pd.notna(ax["gk"]) else np.nan,
        })
    out = pd.DataFrame(rows)
    print(f"  {len(out):,} jugadores valorados · media {out['overall'].mean():.1f} · "
          f">=90: {(out['overall'] >= 90).sum()}")
    for pos in ("Forward", "Midfielder", "Defender", "Goalkeeper"):
        s = out[out["position"] == pos]
        top = ", ".join(f"{p.player.split()[-1]} {p.overall:.0f}" for p in s.nlargest(5, "overall").itertuples())
        print(f"    {pos:11}: {top}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = build()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nESCRITO {args.out}")


if __name__ == "__main__":
    main()
