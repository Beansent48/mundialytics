"""Per-league variants of the squad-value shift (LOSO), on one squad-value file.

  global        one (beta, kappa) for all leagues (reference)
  no_bundes     global params fitted and applied on 4 leagues; Bundesliga untouched
  per_league    one (beta, kappa) per league, each fitted on its own league's train seasons
  shrunk        per-league params shrunk halfway to the global fit

usage: python scripts/squad_value/tm_eval_league.py FILE.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

FILE = sys.argv[1]
sys.argv = ["x"]
exec(open("scripts/squad_value/lofo.py").read().split("def rps(P):")[0])  # d, trio, Y, LH, LA, P0, O0
d["date"] = pd.to_datetime(d.date)
S = d.season.values
C = d.comp.values
seasons = sorted(set(S))
leagues = sorted(set(C))
stretch = np.log(LH / LA)
tv = pd.read_csv(FILE, parse_dates=["snap"]).sort_values("snap")


def attach(side):
    q = d[["match_id", "date", side]].rename(columns={side: "team"}).sort_values("date")
    return pd.merge_asof(q, tv.rename(columns={"snap": "date"}), on="date", by="team", direction="backward",
                         allow_exact_matches=False, tolerance=pd.Timedelta(days=10)).set_index("match_id")


dv = np.log(d.match_id.map(attach("home").v18).values) - np.log(d.match_id.map(attach("away").v18).values)
dv = np.where(np.isfinite(dv), dv, 0.0)


def rps_of(P, m):
    return (((np.cumsum(P, 1) - np.cumsum(Y[m], 1)) ** 2)[:, :2].sum(1)) / 2


def fit(m):
    def obj(th):
        sh = th[0] * dv[m] + th[1] * stretch[m]
        P, _ = trio(LH[m] * np.exp(sh / 2), LA[m] * np.exp(-sh / 2))
        return rps_of(P, m).mean()
    return minimize(obj, np.zeros(2), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-9}).x


r0 = rps_of(P0, np.ones(len(d), bool))
out = {k: np.zeros(len(d)) for k in ("global", "no_bundes", "per_league", "shrunk")}
params = []
for s in seasons:
    tr, te = S != s, S == s
    g = fit(tr)
    g4 = fit(tr & (C != "Bundesliga"))
    th = {"global": {c: g for c in leagues}, "no_bundes": {c: (g4 if c != "Bundesliga" else np.zeros(2)) for c in leagues}}
    pl = {c: fit(tr & (C == c)) for c in leagues}
    th["per_league"] = pl
    th["shrunk"] = {c: 0.5 * pl[c] + 0.5 * g for c in leagues}
    params.append({c: np.round(pl[c], 3) for c in leagues})
    for k, tk in th.items():
        for c in leagues:
            m = te & (C == c)
            sh = tk[c][0] * dv[m] + tk[c][1] * stretch[m]
            P, _ = trio(LH[m] * np.exp(sh / 2), LA[m] * np.exp(-sh / 2))
            out[k][m] = rps_of(P, m) - r0[m]
    print("hold", s, "per-league (beta,kappa):", params[-1], flush=True)

for k, delta in out.items():
    per = pd.Series(delta).groupby(S).mean()
    print(f"{k:11s} pooled dRPS {delta.mean():+.5f}  better {(per < 0).sum()}/6  per-season(1e-4) {np.round(per.values * 1e4, 1)}  | "
          + "  ".join(f"{c[:6]} {delta[C == c].mean():+.5f}" for c in leagues))
