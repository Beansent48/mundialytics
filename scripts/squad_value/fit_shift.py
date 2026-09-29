"""Fit the deployed squad-value shift (beta, kappa) on the production values file.

Reads data/processed/squad_values.csv (scripts/build_squad_values.py --history) and the
weekly-refit walk-forward components, re-checks the LOSO result on that file, then fits
(beta, kappa) on all six seasons -- the constants for DEPLOYED_CLUB_ENGINE_KWARGS.

    python scripts/squad_value/fit_shift.py
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.argv = ["x"]
exec(open("scripts/squad_value/lofo.py").read().split("def rps(P):")[0])  # d, trio, Y, LH, LA, P0, O0
over = (d.hg + d.ag > 2.5).values
d["date"] = pd.to_datetime(d.date)
S = d.season.values
seasons = sorted(set(S))
stretch = np.log(LH / LA)
tv = pd.read_csv("data/processed/squad_values.csv", parse_dates=["snap"]).sort_values("snap")


def attach(side):
    q = d[["match_id", "date", side]].rename(columns={side: "team"}).sort_values("date")
    return pd.merge_asof(q, tv.rename(columns={"snap": "date"}), on="date", by="team", direction="backward",
                         allow_exact_matches=False, tolerance=pd.Timedelta(days=10)).set_index("match_id")


dv = np.log(d.match_id.map(attach("home").v18).values) - np.log(d.match_id.map(attach("away").v18).values)
print("coverage", round(np.isfinite(dv).mean(), 4))
has = np.isfinite(dv)
dv = np.where(has, dv, 0.0)


def rps_of(P, m):
    return (((np.cumsum(P, 1) - np.cumsum(Y[m], 1)) ** 2)[:, :2].sum(1)) / 2


def apply(th, m):
    # the engine skips the whole shift when a value is missing
    sh = np.where(has[m], th[0] * dv[m] + th[1] * stretch[m], 0.0)
    return trio(LH[m] * np.exp(sh / 2), LA[m] * np.exp(-sh / 2))


def fit(m):
    return minimize(lambda th: rps_of(apply(th, m)[0], m).mean(), np.zeros(2), method="Nelder-Mead",
                    options={"xatol": 1e-5, "fatol": 1e-9}).x


r0 = rps_of(P0, np.ones(len(d), bool))
delta = np.zeros(len(d))
dll = []
for s in seasons:
    tr, te = S != s, S == s
    th = fit(tr)
    P1, O1 = apply(th, te)
    delta[te] = rps_of(P1, te) - r0[te]
    y = over[te]
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    dll.append(ll(O1) - ll(O0[te]))
    print(f"  hold {s}: beta {th[0]:.3f} kappa {th[1]:.3f}  dRPS {delta[te].mean():+.5f}", flush=True)
per = pd.Series(delta).groupby(S).mean()
print(f"LOSO pooled dRPS {delta.mean():+.5f}  better {(per < 0).sum()}/6  OU2.5 dLL {np.mean(dll):+.5f}  "
      f"RPS {r0.mean():.5f} -> {(r0 + delta).mean():.5f}")
th = fit(np.ones(len(d), bool))
print(f"ALL-SEASON FIT: beta={th[0]:.4f} kappa={th[1]:.4f}")
