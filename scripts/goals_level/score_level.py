"""Score an in-season GOALS-LEVEL correction on the walk-forward lambda dumps.

The hypothesis (2026-10-17): 2026/27 runs at 3.09 goals per match against
2.77-2.89 in each of the five seasons before it, and the deployed chain prices
50.2% over 2.5 where 62.6% happen (n=187, z=3.44). A league-season level term
would close that -- IF the same correction would have helped in the past rather
than chased noise. `scripts/experiment_league_level.py` found the in-season
level signal real for events "but not for goals"; this measures goals properly,
on the lambdas as served.

Forms, each a common multiplier m on both lambdas (m commutes with the
squad-value and lineup tilts, whose s depends only on log(lh/la)):
  res_league  r = running actual/predicted total-goal ratio of this
              league-season before this DATE, shrunk to 1 with K matches:
              m = (n*r + K)/(n + K)
  res_pooled  the same ratio pooled over the five leagues (2026/27's drift
              shows up in all five, so pooling cuts the noise ~5x)
  res_prev    as res_league, but shrunk toward this league's ratio over the
              PREVIOUS season instead of toward 1 -- a persistent per-league
              bias rather than an in-season shift

m is clipped to [0.90, 1.10]; the report says how often that binds.

usage: python scripts/goals_level/score_level.py [--k 20,40,70,120,200,400]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS as K, deployed_markets_from_lambdas, sharpen_trio)

DUMPS = sorted((ROOT / "outputs/goals_level").glob("wf_*.csv"))
RHO = K.get("outcome_rho", K.get("ad_rho", 0.0))
TEMP = K.get("goal_temper", 1.0)
GAM = float(np.clip(K.get("sharpen_gamma_1x2", 1.0), 0.5, 2.0))
CONTRAST_ONLY = bool(K.get("sharpen_contrast_only", False))
COH = bool(K.get("coherent_markets", False))
G = np.arange(11)
MG = np.add.outer(G, G)
I, J = np.indices((11, 11))


def chain(lh, la):
    """(1X2 trio, P(over 2.5)) -- the deployed chain, vectorized.

    Checked against deployed_markets_from_lambdas in `verify()`.
    """
    lh = np.clip(np.asarray(lh, float), 0.01, 8.0)[:, None]
    la = np.clip(np.asarray(la, float), 0.01, 8.0)[:, None]
    ph, pa = poisson.pmf(G[None, :], lh), poisson.pmf(G[None, :], la)
    if abs(TEMP - 1.0) > 1e-9:
        ph = ph ** TEMP
        ph /= ph.sum(1, keepdims=True)
        pa = pa ** TEMP
        pa /= pa.sum(1, keepdims=True)
    M = ph[:, :, None] * pa[:, None, :]
    if abs(RHO) > 1e-12:
        l1, l2 = lh[:, 0], la[:, 0]
        taus = (((0, 0), 1 - RHO * l1 * l2), ((0, 1), 1 + RHO * l1),
                ((1, 0), 1 + RHO * l2), ((1, 1), np.full(len(l1), 1 - RHO)))
        for (i, j), tau in taus:
            M[:, i, j] *= np.clip(tau, 0.05, 3.0)
    M /= M.sum((1, 2), keepdims=True)
    regions = [I > J, I == J, I < J]
    P = np.stack([(M * m).sum((1, 2)) for m in regions], 1)
    if abs(GAM - 1.0) > 1e-9:
        # the engine's own map, not a copy of it -- it takes (N, 3) for exactly this
        P = sharpen_trio(P, GAM, CONTRAST_ONLY)
    if COH and abs(GAM - 1.0) > 1e-9:
        # reweight each outcome region to carry the sharpened 1X2, then read the
        # totals off that matrix -- what the deployed config serves
        for k, m in enumerate(regions):
            mass = (M * m).sum((1, 2), keepdims=True)
            scale = np.where(mass > 0, P[:, k][:, None, None] / np.maximum(mass, 1e-300), 1.0)
            M = np.where(m, M * scale, M)
        M /= M.sum((1, 2), keepdims=True)
    return P, (M * (MG > 2.5)).sum((1, 2))


def verify(lh, la):
    P, o25 = chain(lh, la)
    eh = eo = 0.0
    for i in range(len(lh)):
        probs, _ = deployed_markets_from_lambdas(float(lh[i]), float(la[i]))
        eh = max(eh, abs(probs["p_home_win"] - P[i, 0]), abs(probs["p_draw"] - P[i, 1]),
                 abs(probs["p_away_win"] - P[i, 2]))
        eo = max(eo, abs(probs["p_over_25"] - o25[i]))
    print(f"verify vs deployed_markets_from_lambdas on n={len(lh)}: "
          f"max 1X2 err {eh:.2e}, max over2.5 err {eo:.2e}")
    assert eh < 1e-9 and eo < 1e-9, "vectorized chain does not match the deployed one"


def running(d, by):
    """Actual and predicted total goals accumulated STRICTLY BEFORE each date."""
    day = (d.groupby(by + ["date"], sort=True)
             .agg(Gd=("tot", "sum"), Ld=("lam", "sum"), nd=("tot", "size")).reset_index())
    gp = day.groupby(by, sort=False)
    for c, s in (("Gc", "Gd"), ("Lc", "Ld"), ("nc", "nd")):
        day[c] = gp[s].cumsum() - day[s]
    return day[by + ["date", "Gc", "Lc", "nc"]]


def main():
    ks = [int(x) for x in (sys.argv[sys.argv.index("--k") + 1].split(",")
                           if "--k" in sys.argv else [20, 40, 70, 120, 200, 400])]
    d = pd.concat([pd.read_csv(p) for p in DUMPS], ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    d["tot"] = d.hg + d.ag
    d["lam"] = d.lh + d.la
    d["o"] = np.where(d.hg > d.ag, 0, np.where(d.hg == d.ag, 1, 2))
    d["over"] = (d.tot > 2.5).astype(float)
    d = d.sort_values("date").reset_index(drop=True)
    print(f"{len(d)} matches | folds {sorted(d.season.unique())}")

    sample = np.random.default_rng(0).choice(len(d), 300, replace=False)
    verify(d.lh.values[sample], d.la.values[sample])

    d = d.merge(running(d, ["season", "comp"]).rename(
        columns={"Gc": "Gl", "Lc": "Ll", "nc": "nl"}),
        on=["season", "comp", "date"], how="left")
    d = d.merge(running(d, ["season"]).rename(
        columns={"Gc": "Gp", "Lc": "Lp", "nc": "npool"}),
        on=["season", "date"], how="left")

    full = d.groupby(["season", "comp"]).agg(G=("tot", "sum"), L=("lam", "sum")).reset_index()
    full["r_full"] = full.G / full.L
    rprev = {(s, c): v for s, c, v in zip(full.season, full.comp, full.r_full)}
    seasons = sorted(d.season.unique())
    back = {s: seasons[i - 1] for i, s in enumerate(seasons) if i}
    d["r_prev"] = [rprev.get((back.get(s), c), 1.0) for s, c in zip(d.season, d.comp)]

    def mult(form, k):
        if form == "res_pooled":
            n, r, tgt = d.npool.values, (d.Gp / d.Lp.replace(0, np.nan)).values, 1.0
        else:
            n, r = d.nl.values, (d.Gl / d.Ll.replace(0, np.nan)).values
            tgt = d.r_prev.values if form == "res_prev" else 1.0
        r = np.where(np.isfinite(r), r, 1.0)
        n = np.nan_to_num(np.asarray(n, float), nan=0.0)
        return np.clip((n * r + k * tgt) / (n + k), 0.90, 1.10)

    Y = np.eye(3)[d.o.values]

    def score(m):
        P, o25 = chain(d.lh.values * m, d.la.values * m)
        rps = ((P.cumsum(1)[:, :2] - Y.cumsum(1)[:, :2]) ** 2).sum(1) / 2
        ll = -(d.over.values * np.log(np.clip(o25, 1e-9, 1))
               + (1 - d.over.values) * np.log(np.clip(1 - o25, 1e-9, 1)))
        return rps, ll

    b_rps, b_ou = score(np.ones(len(d)))
    print(f"\nBASELINE (deployed chain): 1X2 RPS {b_rps.mean():.5f} | "
          f"O/U2.5 logloss {b_ou.mean():.5f}")
    per_base = d.assign(rps=b_rps, ou=b_ou).groupby("season").agg(
        n=("rps", "size"), rps=("rps", "mean"), ou=("ou", "mean"),
        goals=("tot", "mean"), lam=("lam", "mean"))
    per_base["bias"] = per_base.goals / per_base.lam
    print(per_base.round(4).to_string())

    rows = []
    for form in ("res_league", "res_pooled", "res_prev"):
        for k in ks:
            m = mult(form, k)
            rps, ou = score(m)
            per = d.assign(a=rps - b_rps, b=ou - b_ou).groupby("season")[["a", "b"]].mean()
            rows.append({"form": form, "K": k,
                         "d_rps": (rps - b_rps).mean(),
                         "rps_folds": f"{int((per.a < 0).sum())}/{len(per)}",
                         "d_ou": (ou - b_ou).mean(),
                         "ou_folds": f"{int((per.b < 0).sum())}/{len(per)}",
                         "m_mean": m.mean(),
                         "clip%": float(((m <= 0.9 + 1e-9) | (m >= 1.1 - 1e-9)).mean() * 100)})
    r = pd.DataFrame(rows)
    print("\n=== level correction vs deployed (negative = better) ===")
    print(r.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    best = r.sort_values("d_ou").iloc[0]
    m = mult(best.form, int(best.K))
    rps, ou = score(m)
    per = d.assign(d_rps=rps - b_rps, d_ou=ou - b_ou).groupby("season")[["d_rps", "d_ou"]].mean()
    print(f"\n=== best on O/U: {best.form} K={int(best.K)} -- per fold ===")
    print(per.round(5).to_string())

    # Does it help WHERE a shift exists? A correction can be worth nothing on
    # average simply because history holds no drift to learn from -- which is a
    # different verdict from "it chases noise". Bucket by how far the running
    # ratio had moved by the time the match was priced.
    rr = np.where(np.isfinite((d.Gl / d.Ll.replace(0, np.nan)).values),
                  (d.Gl / d.Ll.replace(0, np.nan)).values, 1.0)
    cut = pd.cut(np.abs(rr - 1), [0, 0.05, 0.10, 0.15, 10],
                 labels=["<5%", "5-10%", "10-15%", ">15%"])
    cond = (d.assign(b=cut, d_rps=rps - b_rps, d_ou=ou - b_ou, drift=np.abs(rr - 1),
                     seen=d.nl.fillna(0))
            .groupby("b", observed=True)
            .agg(n=("d_ou", "size"), seen=("seen", "mean"),
                 d_rps=("d_rps", "mean"), d_ou=("d_ou", "mean")))
    print(f"\n=== {best.form} K={int(best.K)} by in-season drift already observed ===")
    print(cond.round(5).to_string())




def gated(argv=None):
    """POST-HOC: a correction that only fires on a drift this big.

    Chosen AFTER seeing that the main grid loses on history (where the bias
    wobbles +/-2% with no persistence) and wins only on 2026/27 (+12.9%). So
    this is not a validated gain -- it is a guardrail whose historical folds
    measure only that it stays out of the way. Read it that way.
    """
    d = pd.concat([pd.read_csv(p) for p in DUMPS], ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    d["tot"] = d.hg + d.ag
    d["lam"] = d.lh + d.la
    d["o"] = np.where(d.hg > d.ag, 0, np.where(d.hg == d.ag, 1, 2))
    d["over"] = (d.tot > 2.5).astype(float)
    d = d.sort_values("date").reset_index(drop=True)
    d = d.merge(running(d, ["season"]).rename(
        columns={"Gc": "Gp", "Lc": "Lp", "nc": "npool"}), on=["season", "date"], how="left")
    r = np.where(d.Lp.values > 0, d.Gp.values / np.where(d.Lp.values > 0, d.Lp.values, 1), 1.0)
    n = np.nan_to_num(d.npool.values.astype(float))
    Y = np.eye(3)[d.o.values]

    def score(m):
        P, o25 = chain(d.lh.values * m, d.la.values * m)
        return (((P.cumsum(1)[:, :2] - Y.cumsum(1)[:, :2]) ** 2).sum(1) / 2,
                -(d.over.values * np.log(np.clip(o25, 1e-9, 1))
                  + (1 - d.over.values) * np.log(np.clip(1 - o25, 1e-9, 1))))
    b_rps, b_ou = score(np.ones(len(d)))
    rows = []
    for nmin in (50, 100, 150):
        for g in (0.04, 0.06, 0.08, 0.10):
            for k in (120, 200):
                m = np.clip((n * r + k) / (n + k), 0.90, 1.10)
                m = np.where((n >= nmin) & (np.abs(r - 1) > g), m, 1.0)
                rps, ou = score(m)
                per = d.assign(a=rps - b_rps, b=ou - b_ou).groupby("season")[["a", "b"]].mean()
                hist = per.drop(index="2026-2027")
                rows.append({"n_min": nmin, "gate": g, "K": k,
                             "fires%": float((m != 1.0).mean() * 100),
                             "hist_d_ou": hist.b.mean(),
                             "hist_worse": int((hist.b > 1e-6).sum()),
                             "hist_flat": int((hist.b.abs() <= 1e-6).sum()),
                             "hist_d_rps": hist.a.mean(),
                             "d_ou_2627": per.b.get("2026-2027", np.nan),
                             "d_rps_2627": per.a.get("2026-2027", np.nan)})
    out = pd.DataFrame(rows)
    print("\n\n=== POST-HOC gated correction (fires only above the gate) ===")
    print("hist_* = mean over the SIX historical folds; hist_flat = folds it never fired in")
    print(out.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))


if __name__ == "__main__":
    if "--gated" in sys.argv:
        gated()
    else:
        main()
