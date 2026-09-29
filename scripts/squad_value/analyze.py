import glob
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, "src")
from mundialytics.statistical_core.prediction_engine import deployed_markets_from_lambdas  # noqa

SP = "data/external/transfermarkt/work/"
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(SP + "wfr_*.csv"))], ignore_index=True)
d["date"] = pd.to_datetime(d.date)
d = d.sort_values("date").reset_index(drop=True)

# league membership from the full canonical file (2000+), to tag status
m = pd.read_csv("data/processed/enriched/understat_xg/canonical_matches_with_xg.csv", low_memory=False)
from mundialytics.statistical_core.schemas import canonical_name  # noqa
m["home_team"] = m.home_team.map(canonical_name)
m["away_team"] = m.away_team.map(canonical_name)
teams = pd.concat([m[["season", "competition", "home_team"]].rename(columns={"home_team": "t"}),
                   m[["season", "competition", "away_team"]].rename(columns={"away_team": "t"})]).drop_duplicates()
seasons = sorted(m.season.dropna().unique())
prev = {s: seasons[i - 1] for i, s in enumerate(seasons) if i > 0}
mem = set(zip(teams.season, teams.competition, teams.t))
d["h_new"] = [(prev[s], c, t) not in mem for s, c, t in zip(d.season, d.comp, d.home)]
d["a_new"] = [(prev[s], c, t) not in mem for s, c, t in zip(d.season, d.comp, d.away)]
long = pd.concat([d[["season", "match_id", "home"]].rename(columns={"home": "t"}),
                  d[["season", "match_id", "away"]].rename(columns={"away": "t"})])
long = long.merge(d[["match_id", "date"]], on="match_id").sort_values("date", kind="stable")
long["k"] = long.groupby(["season", "t"]).cumcount()  # matches already played this season
kmap = dict(zip(zip(long.match_id, long.t), long.k))
d["hk"] = [kmap[(a, b)] for a, b in zip(d.match_id, d.home)]
d["ak"] = [kmap[(a, b)] for a, b in zip(d.match_id, d.away)]
out = np.where(d.hg > d.ag, 0, np.where(d.hg == d.ag, 1, 2))


def probs(lh, la):
    P = np.empty((len(lh), 3))
    for i, (x, y) in enumerate(zip(lh, la)):
        p, _ = deployed_markets_from_lambdas(float(x), float(y))
        P[i] = (p["p_home_win"], p["p_draw"], p["p_away_win"])
    return P


def rps(P, o):
    Y = np.eye(3)[o]
    return (((np.cumsum(P, 1) - np.cumsum(Y, 1)) ** 2)[:, :2].sum(1)) / 2


def base_lambdas(dd):
    w, ls = dd.w.values, dd.ls.values
    lh = np.clip((w * dd.xr_h + (1 - w) * dd.lh_ad) * ls, 0.05, 6.0)
    la = np.clip((w * dd.xr_a + (1 - w) * dd.la_ad) * ls, 0.05, 6.0)
    return lh.values, la.values


lh0, la0 = base_lambdas(d)
P0 = probs(lh0, la0)
d["rps0"] = rps(P0, out)
d["ph"], d["pd"], d["pa"] = P0.T
print("weekly-refit deployed RPS", round(d.rps0.mean(), 5), "n", len(d))
print(d.groupby("season").rps0.mean().round(5).to_string())
d.to_pickle(SP + "wfr.pkl")

# bias table, team perspective
rows = []
for s, o in (("h", "a"), ("a", "h")):
    pw = d.ph if s == "h" else d.pa
    g = d.hg if s == "h" else d.ag
    go = d.ag if s == "h" else d.hg
    lam = lh0 if s == "h" else la0
    lamo = la0 if s == "h" else lh0
    rows.append(pd.DataFrame({"season": d.season, "new": d[s + "_new"], "known": d[("h" if s == "h" else "a") + "_known"],
                              "k": d[s + "k"], "xpts": 3 * pw + d.pd, "pts": 3 * (g > go) + (g == go),
                              "lam": lam, "g": g, "lamo": lamo, "go": go,
                              "ad": d.lh_ad if s == "h" else d.la_ad, "xr": d.xr_h if s == "h" else d.xr_a,
                              "ado": d.la_ad if s == "h" else d.lh_ad, "xro": d.xr_a if s == "h" else d.xr_h}))
T = pd.concat(rows)
T["kb"] = pd.cut(T.k, [-1, 4, 9, 18, 40], labels=["0-4", "5-9", "10-18", "19+"])
T["grp"] = np.where(~T.new, "stay", np.where(T.known, "promoted", "debut"))
g = T.groupby(["grp", "kb"], observed=True).agg(n=("pts", "size"), xpts=("xpts", "mean"), pts=("pts", "mean"),
                                               lam=("lam", "mean"), g=("g", "mean"), lamo=("lamo", "mean"), go=("go", "mean"),
                                               ad=("ad", "mean"), xr=("xr", "mean"), ado=("ado", "mean"), xro=("xro", "mean"))
g["bias_pts"] = g.xpts - g.pts
print(g.round(3).to_string())
print(T[T.new].groupby("season").apply(lambda x: pd.Series({"n": len(x), "bias": (x.xpts - x.pts).mean()})).round(3).to_string())
