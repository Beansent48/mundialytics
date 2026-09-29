"""LOSO test of the squad-value lambda shift for one squad-value file.

usage: python scripts/squad_value/tm_test_prod.py FILE.csv [FILE2.csv ...]
Prints pooled / per-season dRPS, O/U 2.5 log-loss, and the split by promoted side and league.
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SP = "data/external/transfermarkt/work/"
FILES = sys.argv[1:]
sys.argv = ["x"]
exec(open("scripts/squad_value/lofo.py").read().split("def rps(P):")[0])  # d, trio, Y, LH, LA, P0, O0
over = (d.hg + d.ag > 2.5).values
d["date"] = pd.to_datetime(d.date)
S = d.season.values
seasons = sorted(set(S))
stretch = np.log(LH / LA)
new = (d.h_new | d.a_new).values


def rps_of(P, m):
    return (((np.cumsum(P, 1) - np.cumsum(Y[m], 1)) ** 2)[:, :2].sum(1)) / 2


r0 = rps_of(P0, np.ones(len(d), bool))
for f in FILES:
    tv = pd.read_csv(f, parse_dates=["snap"]).sort_values("snap")

    def attach(side):
        q = d[["match_id", "date", side]].rename(columns={side: "team"}).sort_values("date")
        return pd.merge_asof(q, tv.rename(columns={"snap": "date"}), on="date", by="team", direction="backward",
                             allow_exact_matches=False, tolerance=pd.Timedelta(days=10)).set_index("match_id")

    dv = np.log(d.match_id.map(attach("home").v18).values) - np.log(d.match_id.map(attach("away").v18).values)
    cov = np.isfinite(dv).mean()
    dv = np.where(np.isfinite(dv), dv, 0.0)
    print(f"\n=== {f}  (coverage {cov:.3f})")
    for spec in ("value", "value+stretch"):
        npar = 1 if spec == "value" else 2
        P1 = np.zeros_like(P0)
        O1 = np.zeros_like(O0)
        ths = []
        for s in seasons:
            tr, te = S != s, S == s

            def shift(th, m):
                return th[0] * dv[m] + (th[1] * stretch[m] if npar == 2 else 0.0)

            def obj(th):
                sh = shift(th, tr)
                P, _ = trio(LH[tr] * np.exp(sh / 2), LA[tr] * np.exp(-sh / 2))
                return rps_of(P, tr).mean()
            th = minimize(obj, np.zeros(npar), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-9}).x
            sh = shift(th, te)
            P1[te], O1[te] = trio(LH[te] * np.exp(sh / 2), LA[te] * np.exp(-sh / 2))
            ths.append(th)
        delta = rps_of(P1, np.ones(len(d), bool)) - r0
        per = pd.Series(delta).groupby(S).mean()
        ll = lambda p, m: -(over[m] * np.log(p[m]) + (1 - over[m]) * np.log(1 - p[m])).mean()
        dll = np.mean([ll(O1, S == s) - ll(O0, S == s) for s in seasons])
        print(f"{spec:14s} pooled dRPS {delta.mean():+.5f}  better {(per < 0).sum()}/6  per-season(1e-4) "
              f"{np.round(per.values * 1e4, 1)}  OU2.5 dLL {dll:+.5f}  theta~{np.round(np.mean(ths, 0), 3)}")
        print(f"{'':14s} promoted {delta[new].mean():+.5f}  rest {delta[~new].mean():+.5f}  | "
              + "  ".join(f"{c} {delta[(d.comp == c).values].mean():+.5f}" for c in sorted(d.comp.unique())))
