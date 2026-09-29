import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

sys.path.insert(0, "src")
from mundialytics.statistical_core.prediction_engine import DEPLOYED_CLUB_ENGINE_KWARGS as K  # noqa

SP = "data/external/transfermarkt/work/"
d = pd.read_pickle(SP + "wfr.pkl")
RHO, TEMP, GAM = K["outcome_rho"], K["goal_temper"], K["sharpen_gamma_1x2"]
G = np.arange(11)


def trio(lh, la):
    lh = np.clip(lh, 0.01, 8.0)[:, None]
    la = np.clip(la, 0.01, 8.0)[:, None]
    ph = poisson.pmf(G[None, :], lh) ** TEMP
    ph /= ph.sum(1, keepdims=True)
    pa = poisson.pmf(G[None, :], la) ** TEMP
    pa /= pa.sum(1, keepdims=True)
    M = ph[:, :, None] * pa[:, None, :]
    l1, l2 = lh[:, 0], la[:, 0]
    M[:, 0, 0] *= 1 - l1 * l2 * RHO
    M[:, 0, 1] *= 1 + l1 * RHO
    M[:, 1, 0] *= 1 + l2 * RHO
    M[:, 1, 1] *= 1 - RHO
    M /= M.sum((1, 2), keepdims=True)
    i, j = np.indices((11, 11))
    P = np.stack([(M * (i > j)).sum((1, 2)), (M * (i == j)).sum((1, 2)), (M * (i < j)).sum((1, 2))], 1)
    P = np.clip(P, 1e-9, 1) ** GAM
    P /= P.sum(1, keepdims=True)
    tot = np.add.outer(G, G)
    po25 = (M * (tot > 2.5)).sum((1, 2))
    return P, po25


o = np.where(d.hg > d.ag, 0, np.where(d.hg == d.ag, 1, 2))
Y = np.eye(3)[o]
w, ls = d.w.values, d.ls.values
LH = np.clip((w * d.xr_h + (1 - w) * d.lh_ad) * ls, 0.05, 6.0).values
LA = np.clip((w * d.xr_a + (1 - w) * d.la_ad) * ls, 0.05, 6.0).values
P0, O0 = trio(LH, LA)
print("max |vectorized - engine|", np.abs(P0 - d[["ph", "pd", "pa"]].values).max())


def rps(P):
    return (((np.cumsum(P, 1) - np.cumsum(Y[:len(P)] if len(P) == len(Y) else None, 1)) ** 2)[:, :2].sum(1)) / 2


def rps_rows(P, idx):
    return (((np.cumsum(P, 1) - np.cumsum(Y[idx], 1)) ** 2)[:, :2].sum(1)) / 2


over = (d.hg + d.ag > 2.5).values
d2 = pd.read_csv(SP + "d2feat.csv")
feat = {(r.promo_season, r.t): r for r in d2.itertuples()}


def z(season, team, col):
    r = feat.get((season, team))
    return getattr(r, col) if r is not None else 0.0


for c in ["z_ppg", "z_gd", "z_sotd"]:
    d["h_" + c] = [z(s, t, c) if n else 0.0 for s, t, n in zip(d.season, d.home, d.h_new)]
    d["a_" + c] = [z(s, t, c) if n else 0.0 for s, t, n in zip(d.season, d.away, d.a_new)]
d["h_hasd2"] = np.array([(s, t) in feat for s, t in zip(d.season, d.home)]) & d.h_new.values
d["a_hasd2"] = np.array([(s, t) in feat for s, t in zip(d.season, d.away)]) & d.a_new.values
print("new-team rows with D2 stats:", (d.h_hasd2 | d.a_hasd2).sum(), "of", (d.h_new | d.a_new).sum())

rel = (d.h_new | d.a_new).values
idx = np.where(rel)[0]
hn, an = d.h_new.values[idx].astype(float), d.a_new.values[idx].astype(float)


def adjusted(theta, spec, rows):
    """multipliers on the final lambdas: a new team's own lambda *exp(-A), its opponent's *exp(+B)."""
    ii = idx[rows]
    h, a = hn[rows], an[rows]
    if spec == "const":
        Ah = Aa = theta[0]
        Bh = Ba = theta[1]
    else:
        zc = spec
        zh, za = d["h_" + zc].values[ii], d["a_" + zc].values[ii]
        Ah, Aa = theta[0] + theta[2] * zh, theta[0] + theta[2] * za
        Bh, Ba = theta[1] + theta[3] * zh, theta[1] + theta[3] * za
    lh = LH[ii] * np.exp(-h * Ah + a * Ba)
    la = LA[ii] * np.exp(-a * Aa + h * Bh)
    return trio(lh, la)



seasons = sorted(d.season.unique())
sidx = d.season.values[idx]
HG, AG = d.hg.values[idx], d.ag.values[idx]
for spec, npar in [("const", 2), ("z_ppg", 4), ("z_sotd", 4)]:
    deltas, dll = [], []
    for s in seasons:
        tr, te = sidx != s, sidx == s
        def nll(th):
            ii = idx[tr]; h, a = hn[tr], an[tr]
            if spec == "const":
                Ah = Aa = th[0]; Bh = Ba = th[1]
            else:
                zh, za = d["h_" + spec].values[ii], d["a_" + spec].values[ii]
                Ah, Aa = th[0] + th[2] * zh, th[0] + th[2] * za
                Bh, Ba = th[1] + th[3] * zh, th[1] + th[3] * za
            lh = LH[ii] * np.exp(-h * Ah + a * Ba); la = LA[ii] * np.exp(-a * Aa + h * Bh)
            return -(HG[tr] * np.log(lh) - lh + AG[tr] * np.log(la) - la).sum()
        th = minimize(nll, np.zeros(npar), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 4000}).x
        P1, O1 = adjusted(th, spec, te)
        n_all = (d.season.values == s).sum()
        deltas.append((rps_rows(P1, idx[te]).sum() - rps_rows(P0[idx[te]], idx[te]).sum()) / n_all)
        y = over[idx[te]]
        ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum() / n_all
        dll.append(ll(O1) - ll(O0[idx[te]]))
        print(f"  {spec:7s} hold {s}: theta={np.round(th, 3)}  dRPS={deltas[-1]:+.5f}  dLL_OU25={dll[-1]:+.5f}", flush=True)
    dd = np.array(deltas)
    print(f"{spec}: pooled dRPS {dd.mean():+.5f}  folds better {int((dd < 0).sum())}/6   OU2.5 LL {np.mean(dll):+.5f} ({int((np.array(dll) < 0).sum())}/6)", flush=True)
