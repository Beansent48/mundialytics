import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import itertools
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



def rpsv(P):
    return (((np.cumsum(P, 1) - np.cumsum(Y, 1)) ** 2)[:, :2].sum(1)) / 2
import mundialytics.statistical_core.prediction_engine as pe  # noqa
S = d.season.values
seasons = sorted(set(S))
grid_w = [0.4, 0.5, 0.6, 0.7]
grid_g = [1.2, 1.3, 1.4, 1.5]
table = {}
for wv, gv in itertools.product(grid_w, grid_g):
    GAM = gv
    globals()["GAM"] = gv
    lh = np.clip((wv * d.xr_h + (1 - wv) * d.lh_ad) * ls, 0.05, 6.0).values
    la = np.clip((wv * d.xr_a + (1 - wv) * d.la_ad) * ls, 0.05, 6.0).values
    P, _ = trio(lh, la)
    r = rpsv(P)
    table[(wv, gv)] = pd.Series(r).groupby(S).mean()
    print(wv, gv, round(r.mean(), 5), flush=True)
T = pd.DataFrame(table).T
base = T.loc[(0.6, 1.3)]
print("per-season RPS, deployed (0.6,1.3):", base.round(5).to_dict())
# LOSO: pick best config on the other 5 seasons, score the held-out one
dl = []
for s in seasons:
    best = T.drop(columns=s).mean(1).idxmin()
    dl.append(T.loc[best, s] - base[s])
    print(f"hold {s}: pick {best}  dRPS {dl[-1]:+.5f}")
print("LOSO pooled", round(np.mean(dl), 5), "better", sum(x < 0 for x in dl), "/6")
print(T.mean(1).sort_values().head(6).round(5))
