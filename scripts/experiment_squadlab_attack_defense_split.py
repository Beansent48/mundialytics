#!/usr/bin/env python3
"""Does a SquadLab eleven's SHAPE predict goals beyond its level?

Today a drafted eleven is one number: Elo = a + b * mean(card overall). An XI
of elite forwards and weak defenders plays exactly like a balanced one with the
same mean. This asks whether the card axes carry match information that the
level does not, i.e. whether the engine should split the squad into an attack
and a defence.

Setup, on real clubs' own best elevens drawn from the same card catalogue:

    log lam_h = c + hfa + b*d400 + gA*A_h + gD*D_a + gG*G_a
    log lam_a = c       - b*d400 + gA*A_a + gD*D_h + gG*G_h

A, D, G are the attack / defence / keeper axes of the eleven, RESIDUALISED on
its mean overall across clubs (what the axes say beyond the level). Two
baselines for d400:
    real   the club's pre-match ClubElo (what the rest of the site uses)
    game   Elo read off the squad line, which is what a drafted eleven gets

Honesty: the cards are built on data up to May 2026, so 2024/25 and 2025/26
are IN-SAMPLE for the features (they measured those very seasons). The
coefficients are fitted there and judged only on 2026/27, which no card has
seen. 250 matches is small; a bootstrap CI over matches is reported.

Run:
    python scripts/experiment_squadlab_attack_defense_split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_elo_lambda import clubelo_name  # noqa: E402
from mundialytics.statistical_core.squadlab.cards import best_eleven, load_cards  # noqa: E402

TRAIN = ["2024-2025", "2025-2026"]
TEST = "2026-2027"


def xi_features(cards: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for club in cards.loc[cards["kind"] == "actual", "club"].unique():
        xi = best_eleven(cards, club)
        if len(xi) < 10:
            continue
        out = [c for c in xi if c.position != "Goalkeeper"]
        gk = [c for c in xi if c.position == "Goalkeeper"]
        att = sorted((c.raw_attack for c in out), reverse=True)[:5]
        cre = sorted((c.raw_creation for c in out), reverse=True)[:5]
        dfn = [c.raw_defense for c in out if c.position in ("Defender", "Midfielder")]
        rows.append({
            "club": str(club),
            "ovr": float(np.mean([c.overall for c in xi])),
            "att": float(np.mean(att)), "cre": float(np.mean(cre)),
            "def": float(np.mean(dfn)) if dfn else np.nan,
            "gk": float(gk[0].raw_gk) if gk else np.nan,
            "gk_ovr": float(gk[0].overall) if gk else np.nan,
            "def_ovr": float(np.mean([c.overall for c in xi if c.position == "Defender"] or [np.nan])),
            "fwd_ovr": float(np.mean([c.overall for c in xi if c.position == "Forward"] or [np.nan])),
        })
    f = pd.DataFrame(rows).dropna()
    # residualise every axis on the level: what the shape says beyond the mean
    for col in ("att", "cre", "def", "gk", "gk_ovr", "def_ovr", "fwd_ovr"):
        b, a = np.polyfit(f["ovr"], f[col], 1)
        r = f[col] - (a + b * f["ovr"])
        f[col + "_r"] = r / r.std()
    return f.set_index("club")


def load_matches() -> pd.DataFrame:
    found = pd.read_csv(ROOT / "data/processed/foundation_big5_multi_season.csv", low_memory=False)
    found = found[found["season"].isin(TRAIN + [TEST])].copy()
    found["date"] = pd.to_datetime(found["date"], errors="coerce")
    hist = {}
    for t in set(found.home_team) | set(found.away_team):
        p = ROOT / f"data/external/clubelo/teams/{clubelo_name(t)}.csv"
        if p.exists():
            h = pd.read_csv(p)
            h["From"] = pd.to_datetime(h["From"], errors="coerce")
            h["To"] = pd.to_datetime(h["To"], errors="coerce")
            hist[t] = h[["From", "To", "Elo"]].dropna()

    def elo_at(t, d):
        h = hist.get(t)
        if h is None:
            return np.nan
        # the rating in force the day BEFORE kick-off
        m = h[(h.From <= d - pd.Timedelta(days=1)) & (h.To >= d - pd.Timedelta(days=1))]
        return float(m.Elo.iloc[0]) if len(m) else np.nan

    found["eh"] = [elo_at(t, d) for t, d in zip(found.home_team, found.date)]
    found["ea"] = [elo_at(t, d) for t, d in zip(found.away_team, found.date)]
    return found.dropna(subset=["eh", "ea", "home_goals", "away_goals"])


def nll(lh, la, hg, ag):
    return -(hg * np.log(lh) - lh - gammaln(hg + 1) + ag * np.log(la) - la - gammaln(ag + 1))


def design(m: pd.DataFrame, f: pd.DataFrame, cols: list[str], elo_kind: str):
    H, A = f.loc[m.home_team], f.loc[m.away_team]
    if elo_kind == "real":
        d400 = (m.eh.to_numpy() - m.ea.to_numpy()) / 400.0
    else:
        d400 = (H["elo_hat"].to_numpy() - A["elo_hat"].to_numpy()) / 400.0
    # for the home goals: own attack-type cols vs the away side's defence-type cols
    own = [c for c in cols if c.split("_")[0] in ("att", "cre", "fwd")]
    opp = [c for c in cols if c not in own]
    Xh = np.column_stack([H[c].to_numpy() for c in own] + [A[c].to_numpy() for c in opp]) if cols else np.zeros((len(m), 0))
    Xa = np.column_stack([A[c].to_numpy() for c in own] + [H[c].to_numpy() for c in opp]) if cols else np.zeros((len(m), 0))
    return d400, Xh, Xa, m.home_goals.to_numpy(float), m.away_goals.to_numpy(float), own + opp


def fit(d400, Xh, Xa, hg, ag):
    k = Xh.shape[1]

    def obj(p):
        c, hfa, b, g = p[0], p[1], p[2], p[3:]
        lh = np.exp(c + hfa + b * d400 + Xh @ g)
        la = np.exp(c - b * d400 + Xa @ g)
        return nll(lh, la, hg, ag).sum()

    return minimize(obj, np.r_[0.2, 0.2, 0.7, np.zeros(k)], method="L-BFGS-B").x


def per_match_nll(p, d400, Xh, Xa, hg, ag):
    c, hfa, b, g = p[0], p[1], p[2], p[3:]
    return nll(np.exp(c + hfa + b * d400 + Xh @ g), np.exp(c - b * d400 + Xa @ g), hg, ag)


def main() -> None:
    cards = load_cards()
    f = xi_features(cards)
    m = load_matches()
    m = m[m.home_team.isin(f.index) & m.away_team.isin(f.index)]
    # the game's squad line, refitted here on the same clubs: elo ~ mean ovr
    pre = {r.home_team: r.eh for r in m[m.season == TEST].sort_values("date").drop_duplicates("home_team").itertuples()}
    xs = f.loc[[t for t in pre if t in f.index], "ovr"]
    b1, a1 = np.polyfit(xs, [pre[t] for t in xs.index], 1)
    f["elo_hat"] = a1 + b1 * f["ovr"]
    print(f"clubs with an XI: {len(f)} | matches train {int(m.season.isin(TRAIN).sum())} test {int((m.season == TEST).sum())}")
    print(f"game line on 26/27 clubs: elo = {a1:.0f} + {b1:.1f} * mean_ovr\n")

    variants = {
        "level only": [],
        "att+def": ["att_r", "def_r"],
        "att+def+gk": ["att_r", "def_r", "gk_r"],
        "att+cre+def+gk": ["att_r", "cre_r", "def_r", "gk_r"],
        "by-line ovr (fwd/def/gk)": ["fwd_ovr_r", "def_ovr_r", "gk_ovr_r"],
    }
    tr, te = m[m.season.isin(TRAIN)], m[m.season == TEST]
    rng = np.random.default_rng(0)
    for elo_kind in ("real", "game"):
        print(f"== baseline Elo: {elo_kind} ==")
        base = None
        for name, cols in variants.items():
            p = fit(*design(tr, f, cols, elo_kind)[:5])
            d = design(te, f, cols, elo_kind)
            ll = per_match_nll(p, *d[:5])
            if base is None:
                base = ll
                print(f"  {name:26s} test NLL/match {ll.mean():.4f}")
                continue
            diff = ll - base
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [5, 95])
            coefs = ", ".join(f"{n}={g:+.3f}" for n, g in zip(d[5], p[3:]))
            print(f"  {name:26s} test ΔNLL {diff.mean():+.4f} [90% {lo:+.4f},{hi:+.4f}]  ({coefs})")
        print()


if __name__ == "__main__":
    main()
