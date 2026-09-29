import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SP = "data/external/transfermarkt/work/"
sys.path.insert(0, SP)
sys.argv = ["x"]
src = open("scripts/squad_value/lofo.py").read().split("def rps(P):")[0]
exec(src)  # d, trio, Y, LH, LA, P0, O0, over

over = (d.hg + d.ag > 2.5).values
tv = pd.read_csv(SP + "tm_squad_value.csv", parse_dates=["snap"]).sort_values("snap")
d["date"] = pd.to_datetime(d.date)


def attach(side):
    q = d[["match_id", "date", side]].rename(columns={side: "team"}).sort_values("date")
    r = pd.merge_asof(q, tv.rename(columns={"snap": "date"}), on="date", by="team",
                      direction="backward", allow_exact_matches=False, tolerance=pd.Timedelta(days=10))
    return r.set_index("match_id")



vh = attach("home")["v18"]; va = attach("away")["v18"]
dv = np.log(d.match_id.map(vh).values) - np.log(d.match_id.map(va).values)
dv = np.where(np.isfinite(dv), dv, 0.0)
stretch = np.log(LH / LA)
S = d.season.values
seasons = sorted(set(S))
def rps_of(P, m):
    return (((np.cumsum(P, 1) - np.cumsum(Y[m], 1)) ** 2)[:, :2].sum(1)) / 2
P1 = np.zeros_like(P0)
for s in seasons:
    tr, te = S != s, S == s
    def f(th):
        sh = th[0] * dv[tr] + th[1] * stretch[tr]
        P, _ = trio(LH[tr] * np.exp(sh / 2), LA[tr] * np.exp(-sh / 2))
        return rps_of(P, tr).mean()
    th = minimize(f, np.zeros(2), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-9}).x
    sh = th[0] * dv[te] + th[1] * stretch[te]
    P1[te], _ = trio(LH[te] * np.exp(sh / 2), LA[te] * np.exp(-sh / 2))
    print("hold", s, "beta_value", round(th[0], 3), "kappa_stretch", round(th[1], 3))
d["r0"] = rps_of(P0, np.ones(len(d), bool)); d["r1"] = rps_of(P1, np.ones(len(d), bool))
d["new"] = d.h_new | d.a_new
d["kmin"] = np.minimum(d.hk, d.ak)
d["phase"] = pd.cut(d.kmin, [-1, 4, 9, 18, 40], labels=["md1-5", "md6-10", "md11-19", "md20+"])
d["delta"] = d.r1 - d.r0
print(d.groupby("new").delta.agg(["size", "mean"]).round(5))
print(d.groupby("phase", observed=True).delta.agg(["size", "mean"]).round(5))
print(d.groupby("comp").delta.agg(["size", "mean"]).round(5))
# log loss + accuracy
ll = lambda P: -np.log(np.clip(P[np.arange(len(P)), np.argmax(Y, 1)], 1e-12, 1)).mean()
print("1X2 logloss", round(ll(P0), 5), "->", round(ll(P1), 5), " acc", round((P0.argmax(1) == Y.argmax(1)).mean(), 4), "->", round((P1.argmax(1) == Y.argmax(1)).mean(), 4))
print("RPS", round(d.r0.mean(), 5), "->", round(d.r1.mean(), 5))
# vs Bet365
x = pd.read_pickle(SP + "wfr_odds.pkl")[["match_id", "r_bk", "r_us"]]
z = d.merge(x, on="match_id")
print("matched", len(z), "gap to Bet365:", round((z.r0 - z.r_bk).mean(), 5), "->", round((z.r1 - z.r_bk).mean(), 5))
print(z.groupby("new").apply(lambda q: pd.Series({"gap_before": (q.r0 - q.r_bk).mean(), "gap_after": (q.r1 - q.r_bk).mean()})).round(5))
d[["match_id", "r0", "r1"]].assign(ph1=P1[:, 0], pd1=P1[:, 1], pa1=P1[:, 2]).to_pickle(SP + "tm_preds.pkl")
