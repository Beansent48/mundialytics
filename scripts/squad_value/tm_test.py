import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SP = "data/external/transfermarkt/work/"
sys.path.insert(0, SP)
sys.argv = ["x"]
src = open(SP + "lofo.py").read().split("def rps(P):")[0]
exec(src)  # d, trio, Y, LH, LA, P0, O0, over

over = (d.hg + d.ag > 2.5).values
tv = pd.read_csv(SP + "tm_squad_value.csv", parse_dates=["snap"]).sort_values("snap")
d["date"] = pd.to_datetime(d.date)


def attach(side):
    q = d[["match_id", "date", side]].rename(columns={side: "team"}).sort_values("date")
    r = pd.merge_asof(q, tv.rename(columns={"snap": "date"}), on="date", by="team",
                      direction="backward", allow_exact_matches=False, tolerance=pd.Timedelta(days=10))
    return r.set_index("match_id")


results = {}
for VCOL in ["v11", "v18", "v25"]:
    vh = attach("home")[VCOL]
    va = attach("away")[VCOL]
    dv = np.log(d.match_id.map(vh).values) - np.log(d.match_id.map(va).values)
    miss = ~np.isfinite(dv)
    dv = np.where(miss, 0.0, dv)
    if VCOL == "v18":
        print("value coverage", round(1 - miss.mean(), 4))
    stretch = np.log(LH / LA)
    kmin = np.minimum(d.hk.values, d.ak.values)
    early = np.exp(-kmin / 8.0)
    new = (d.h_new | d.a_new).values.astype(float)
    S = d.season.values
    seasons = sorted(set(S))

    def build(th, spec):
        if spec == "value":
            s = th[0] * dv
        elif spec == "value_early":
            s = (th[0] + th[1] * early) * dv
        elif spec == "value_promo":
            s = (th[0] + th[1] * new) * dv
        elif spec == "stretch":
            s = th[0] * stretch
        elif spec == "value+stretch":
            s = th[0] * dv + th[1] * stretch
        elif spec == "all":
            s = (th[0] + th[1] * early + th[2] * new) * dv + th[3] * stretch
        return s

    NP = {"value": 1, "value_early": 2, "value_promo": 2, "stretch": 1, "value+stretch": 2, "all": 4}

    def rps_of(P, m):
        return (((np.cumsum(P, 1) - np.cumsum(Y[m], 1)) ** 2)[:, :2].sum(1)) / 2

    for spec in (["stretch", "value", "value_early", "value_promo", "value+stretch", "all"] if VCOL == "v18" else ["value", "value+stretch"]):
        dl, dll, ths = [], [], []
        for s in seasons:
            tr, te = S != s, S == s

            def f(th):
                sh = build(th, spec)[tr]
                P, _ = trio(LH[tr] * np.exp(sh / 2), LA[tr] * np.exp(-sh / 2))
                return rps_of(P, tr).mean()
            th = minimize(f, np.zeros(NP[spec]), method="Nelder-Mead",
                          options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 3000}).x
            sh = build(th, spec)[te]
            P1, O1 = trio(LH[te] * np.exp(sh / 2), LA[te] * np.exp(-sh / 2))
            dl.append(rps_of(P1, te).mean() - rps_of(P0[te], te).mean())
            y = over[te]
            ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
            dll.append(ll(O1) - ll(O0[te]))
            ths.append(np.round(th, 3))
        dl = np.array(dl)
        print(f"{VCOL} {spec:14s} pooled dRPS {dl.mean():+.5f}  better {int((dl < 0).sum())}/6  "
              f"per-season {np.round(dl * 1e4, 1)}  OU2.5 dLL {np.mean(dll):+.5f}  theta(last) {ths[-1]}", flush=True)
