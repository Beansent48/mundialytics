#!/usr/bin/env python3
"""A deep, role-agnostic PROFILE of concrete skills per player.

WHY THIS EXISTS. The card rating used to compress every player onto three
generic axes (attack / defence / creation) and separate roles only by REWEIGHTING
those same three numbers. That cannot tell the KIND of thing a player is good at
— an extremo's creation (dribble, cross, cut-in) and a mediapunta's creation
(through balls, big chances) came out as the same "creation" number. And because
every axis was a within-position percentile, a merely good season read as elite:
Malen's 18 goals sat in the 97th percentile of forwards and the curve sent him
near the top, even though he is a normal forward, not a Kane.

WHAT THIS PRODUCES. One row per player with ~19 outfield sub-stats and a keeper
set, each expressed as a within-(season, position) PERCENTILE in [0,1], recency-
blended across the three seasons (25/26 leads) and shrunk toward the middle by
minutes so a thin sample cannot spike. This file is deliberately role-agnostic:
`build_role_ratings.py` turns these percentiles into a rating by (a) an
elite-emphasis curve so only the genuine top of each skill counts, (b) role
value-weights, and (c) a best-fit role assignment. Keeping the two apart means
the same profile also feeds player props and a scouting view, each with its own
emphasis.

PROVIDER-NEUTRAL BY CONSTRUCTION. The three seasons come from different providers
(23/24 & 24/25 FBref, 25/26 a FBref+Understat+SofaScore merge whose columns are
named and scaled differently). Every sub-stat is turned into a percentile WITHIN
its own season+position before anything is blended, so a provider that runs hot
cannot lift the players it happened to measure — only the RANK within the season
crosses over. Where a season lacks a column the sub-stat is left blank for that
season and the others carry it (renormalised), which is why 25/26's missing
progressive-pass and through-ball detail does not zero those skills out.

Run: .venv/Scripts/python.exe scripts/build_player_micro_stats.py
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

SRC = ROOT / "data/external/advanced/fbref_kaggle"
OUT = ROOT / "data/processed/player_micro_stats.csv"

# The recency and minutes conventions are shared with the attacking builder so
# the two stay consistent; import rather than re-declare.
from build_player_attack_fbref import (  # noqa: E402
    SEASON_RECENCY, MIN_90S, LEAGUE_2526, _pos, align_positions, league_elo, LEAGUE_W,
)

CRED_90S = 10.0     # minutes credibility: a percentile is shrunk toward 0.5 by this

# The outfield profile. Every value is stored so that HIGHER IS BETTER (the two
# "bad" events, being dribbled past and losing the ball, are inverted at load).
# CHAIN is deliberately absent — the user parked xG-chain/buildup as experimental
# because it can move everything; it is plumbed in load_2526 but not emitted.
OUTFIELD = [
    "FIN", "BOX", "CONV", "XA", "BIGC", "THRU", "CROSS", "SETP_TAKE", "SETP_FIN",
    "DRIB", "PROG", "LONG", "RETEN", "FOULDRAWN", "DUEL", "AER", "INT", "RECOV",
    "HIGHREG",
]
KEEPER = ["SHOTSTOP", "AERIAL_GK", "SWEEP", "DISTRIB"]
SUBSTATS = OUTFIELD + KEEPER


def _num(s):
    return pd.to_numeric(s, errors="coerce")


# ── 25/26: the scouting merge (SofaScore + Understat) ──────────────────────────
def load_2526() -> pd.DataFrame:
    p = SRC / "scouting_2526.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, low_memory=False)
    # a transferred player appears once per club with the SAME season totals; keep
    # the club he played most (decides his league) and use each provider's OWN
    # minutes as the denominator (see build_player_attack_fbref.load_2526).
    d = d.sort_values("90s_", ascending=False).drop_duplicates("player")
    nu = (_num(d["minutes_under"]) / 90.0).clip(lower=0.5)      # Understat 90s
    ns = (_num(d["minutesPlayed"]) / 90.0).clip(lower=0.5)      # SofaScore 90s
    npg, npxg = _num(d["np_goals_under"]), _num(d["np_xg_under"])
    out = pd.DataFrame({
        "player": d["player"], "season": "2526", "league": d["league"].map(LEAGUE_2526).fillna(d["league"]),
        "position": d["pos_"].map(_pos), "n90": _num(d["90s_"]),
        "FIN": npxg / nu,
        "BOX": _num(d["shotsFromInsideTheBox"]) / ns,
        "CONV": _num(d["goalConversionPercentage"]),
        "XA": _num(d["xa_under"]) / nu,
        "BIGC": _num(d["bigChancesCreated"]) / ns,
        "THRU": _num(d["accurateFinalThirdPasses"]) / ns,       # scouting has no true-TB; final-third proxy
        "CROSS": _num(d["accurateCrosses"]) / ns,
        "SETP_TAKE": _num(d["freeKickGoal"]) / ns,              # thin here; FBref dead-ball SCA carries it
        "SETP_FIN": _num(d["headedGoals"]) / ns,
        "DRIB": _num(d["successfulDribbles"]) / ns,
        "PROG": _num(d["accurateOppositionHalfPasses"]) / ns,   # no PrgP/PrgC in scouting; opp-half proxy
        "LONG": _num(d["accurateLongBalls"]) / ns,
        "RETEN": _num(d["accuratePassesPercentage"]),
        "FOULDRAWN": _num(d["wasFouled"]) / ns,
        "DUEL": _num(d["groundDuelsWonPercentage"]),
        "AER": _num(d["aerialDuelsWonPercentage"]),
        "INT": (_num(d["interceptions"]) + _num(d["blockedShots"]).fillna(0)
                + _num(d["clearances"]).fillna(0)) / ns,
        "RECOV": _num(d["ballRecovery"]) / ns,
        "HIGHREG": _num(d["possessionWonAttThird"]) / ns,
        # keeper
        "SHOTSTOP": _num(d["goalsPrevented"]) / ns,
        "AERIAL_GK": (_num(d["highClaims"]).fillna(0) + _num(d["punches"]).fillna(0)) / ns,
        "SWEEP": _num(d["successfulRunsOut"]) / ns,
        "DISTRIB": _num(d["accurateLongBallsPercentage"]),
    })
    return out


# ── 24/25: the FBref wide dump ─────────────────────────────────────────────────
def load_2425() -> pd.DataFrame:
    p = SRC / "fbref_players_2425.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, low_memory=False)
    n = _num(d["90s"]).clip(lower=0.5)

    def col(name):
        return _num(d[name]) if name in d.columns else pd.Series(np.nan, index=d.index)

    out = pd.DataFrame({
        "player": d["Player"], "season": "2425", "league": d["Comp"],
        "position": d["Pos"].map(_pos), "n90": _num(d["90s"]),
        "FIN": col("npxG") / n,
        "BOX": col("Att Pen") / n,                     # touches in the opp box
        "CONV": col("G/Sh"),
        "XA": col("xAG") / n,
        "BIGC": col("PPA") / n,                        # passes into the box (proxy)
        "THRU": col("TB") / n,                         # through balls
        "CROSS": col("CrsPA") / n,                     # crosses into the box
        "SETP_TAKE": col("PassDead") / n,              # dead-ball shot-creating passes
        "SETP_FIN": np.nan,                            # header goals not in FBref; 25/26 carries it
        "DRIB": col("Succ") / n,                       # take-ons completed
        "PROG": (col("PrgP").fillna(0) + col("PrgC").fillna(0)) / n,
        "LONG": col("Sw") / n,                         # switches of play
        "RETEN": col("Cmp%"),
        "FOULDRAWN": col("Fld") / n,
        "DUEL": col("Tkl%"),
        "AER": col("Won%"),
        "INT": (col("Int").fillna(0) + col("Blocks").fillna(0) + col("Clr").fillna(0)) / n,
        "RECOV": col("Recov") / n,
        "HIGHREG": col("Att 3rd") / n,                 # tackles in the attacking third
        "SHOTSTOP": col("PSxG+/-") / n,
        "AERIAL_GK": col("Stp%"),                      # % of crosses stopped
        "SWEEP": col("#OPA/90"),
        "DISTRIB": col("Launch%"),
    })
    return out


# ── 23/24: the FBref denormalised tables ───────────────────────────────────────
def load_2324() -> pd.DataFrame:
    def tbl(name):
        f = SRC / f"2324_{name}.csv"
        if not f.exists():
            return None
        t = pd.read_csv(f)
        return t.sort_values("minutes_90s", ascending=False).drop_duplicates("player").set_index("player")
    sh, pa, po = tbl("Shooting"), tbl("Passing"), tbl("Possession")
    de, mi, gs, pt = tbl("Defense"), tbl("Misc"), tbl("GSC"), tbl("Pass_types")
    ga = tbl("Goalkeeping_adv")
    if sh is None:
        return pd.DataFrame()
    idx = sh.index
    n = _num(sh["minutes_90s"]).clip(lower=0.5)

    def g(t, c):
        if t is None or c not in t.columns:
            return pd.Series(np.nan, index=idx)
        return _num(t[c]).reindex(idx)

    out = pd.DataFrame({
        "player": idx, "season": "2324", "league": sh["league"].values,
        "position": sh["position"].map(_pos).values, "n90": _num(sh["minutes_90s"]).values,
        "FIN": (g(sh, "npxg") / n).values,
        "BOX": (g(po, "touches_apa") / n).values,
        "CONV": g(sh, "goals_per_shot").values,
        "XA": (g(pa, "xAG") / n).values,
        "BIGC": (g(pa, "P_into_PA") / n).values,
        "THRU": (g(pt, "through_balls") / n).values,
        "CROSS": (g(pa, "C_into_PA") / n).values,
        "SETP_TAKE": (g(gs, "sca_passes_dead") / n).values,
        "SETP_FIN": np.nan,
        "DRIB": (g(po, "TO_won") / n).values,
        "PROG": ((g(pa, "prog_passes").fillna(0) + g(po, "prog_carries").fillna(0)) / n).values,
        "LONG": (g(pt, "passes_switches") / n).values,
        "RETEN": g(pa, "passes_pct").values,
        "FOULDRAWN": (g(mi, "fouled") / n).values,
        "DUEL": g(de, "challenge_tackles_pct").values,
        "AER": g(mi, "aerials_won_pct").values,
        "INT": ((g(de, "intr").fillna(0) + g(de, "blocks").fillna(0) + g(de, "clr").fillna(0)) / n).values,
        "RECOV": (g(mi, "ball_recoveries") / n).values,
        "HIGHREG": (g(de, "att_3rd") / n).values,
        "SHOTSTOP": (g(ga, "psxg_net") / n).values if ga is not None else np.nan,
        "AERIAL_GK": g(ga, "crosses_stopped_pct").values if ga is not None else np.nan,
        "SWEEP": g(ga, "def_actions_outside_pen_area_per90").values if ga is not None else np.nan,
        "DISTRIB": g(ga, "launch_pct").values if ga is not None else np.nan,
    })
    return out


def build() -> pd.DataFrame:
    frames = [f for f in (load_2526(), load_2425(), load_2324()) if not f.empty]
    if not frames:
        raise SystemExit("faltan los volcados FBref/scouting — ejecuta scripts/download_fbref_kaggle.py")
    full = pd.concat(frames, ignore_index=True)
    full = full[_num(full["n90"]).fillna(0) >= MIN_90S].copy()
    full = align_positions(full)
    ns = {k: int(v) for k, v in full["season"].value_counts().items()}

    # ── each sub-stat -> within-(season, position) PERCENTILE, provider-neutral ──
    key = [full["season"], full["position"]]
    pct = pd.DataFrame(index=full.index)
    for s in SUBSTATS:
        col = _num(full[s]) if s in full.columns else pd.Series(np.nan, index=full.index)
        pct[s] = col.groupby(key).rank(pct=True)

    # ── recency blend (25/26 leads) + minutes credibility, per player ──────────
    sw = full["season"].astype(str).map(SEASON_RECENCY).fillna(0.3)
    w = sw * _num(full["n90"]).clip(lower=0.1)
    full["_w"], full["_swn"] = w, sw * _num(full["n90"])
    rows = []
    for player, gi in full.groupby("player", sort=False):
        gi = gi.sort_values("season")
        wv = gi["_w"].to_numpy(float)
        n90 = float(np.nansum(gi["_swn"].to_numpy(float)))
        cred = n90 / (n90 + CRED_90S)
        row = {"player": player, "position": gi["position"].iloc[-1],
               "league": gi["league"].iloc[-1], "n90": round(n90, 1)}
        for s in SUBSTATS:
            v = pct.loc[gi.index, s].to_numpy(float)
            ok = np.isfinite(v) & (wv > 0)
            if ok.any():
                p = float(np.average(v[ok], weights=wv[ok]))
                row[s] = round(0.5 + (p - 0.5) * cred, 4)     # shrink toward the median by minutes
            else:
                row[s] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)

    # league strength as a stored column (the role layer folds it in at LEAGUE_W)
    strength = league_elo()
    if strength:
        e = out["league"].map(strength).astype(float)
        out["league_z"] = ((e - e.mean()) / (e.std() or 1.0)).fillna(0.0).round(3)
    else:
        out["league_z"] = 0.0

    cov = out[OUTFIELD].notna().mean(axis=1)
    print(f"  {len(out):,} jugadores · filas/temporada {ns}")
    print(f"  cobertura media de sub-stats outfield: {cov.mean():.0%}")
    cols = ["player", "position", "league", "league_z", "n90"] + SUBSTATS
    return out[cols].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = build()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nESCRITO {args.out}  ({len(out):,} jugadores, {len(SUBSTATS)} sub-stats)")


if __name__ == "__main__":
    main()
