import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from mundialytics.statistical_core.schemas import canonical_name  # noqa

SP = "data/external/transfermarkt/work/"
d = pd.read_csv("data/processed/foundation_second_divisions.csv", low_memory=False)
d["home_team"] = d.home_team.map(canonical_name)
d["away_team"] = d.away_team.map(canonical_name)
rows = []
for s, o in (("home", "away"), ("away", "home")):
    g, go = d[s + "_goals"], d[o + "_goals"]
    rows.append(pd.DataFrame({"season": d.season, "d2": d.competition, "t": d[s + "_team"],
                              "gd": g - go, "gf": g, "ga": go, "pts": 3 * (g > go) + (g == go),
                              "sotd": d[s + "_sot"] - d[o + "_sot"]}))
L = pd.concat(rows)
agg = L.groupby(["season", "d2", "t"]).agg(n=("gd", "size"), gd=("gd", "mean"), gf=("gf", "mean"), ga=("ga", "mean"),
                                         ppg=("pts", "mean"), sotd=("sotd", "mean")).reset_index()
# standardize within league-season so Championship vs Serie B are comparable
for c in ["gd", "gf", "ga", "ppg", "sotd"]:
    agg["z_" + c] = agg.groupby(["season", "d2"])[c].transform(lambda x: (x - x.mean()) / x.std())
seasons = sorted(agg.season.unique())
nxt = {s: seasons[i + 1] if i + 1 < len(seasons) else None for i, s in enumerate(seasons)}
# the season a team would be promoted INTO is the following one
yrs = {s: f"{int(s[:4]) + 1}-{int(s[:4]) + 2}" for s in seasons}
agg["promo_season"] = agg.season.map(yrs)
agg.to_csv(SP + "d2feat.csv", index=False)
print(agg.sort_values(["season", "d2", "ppg"], ascending=[True, True, False]).groupby(["season", "d2"]).head(3).tail(15).round(2).to_string())
