"""Production-faithful walk-forward: refit weekly (prod refits daily), update xG form
between refits, debutants priced by the serving stand-in. Dumps the lambda components
so lambda-level corrections can be scored without refitting.

usage: python wf_refit.py SEASON OUTCSV
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from mundialytics.statistical_core.prediction_engine import DEPLOYED_CLUB_ENGINE_KWARGS, PredictionEngine  # noqa
from mundialytics.serving.debutants import relegated_last_season  # noqa
from mundialytics.statistical_core.schemas import canonical_name  # noqa

season, out = sys.argv[1], sys.argv[2]
df = pd.read_csv("data/processed/enriched/understat_xg/canonical_matches_with_xg.csv", low_memory=False)
df = df[df["xg_available"] == True].copy()  # noqa: E712
for c in ["home_goals", "away_goals", "home_xg", "away_xg"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["home_goals", "away_goals", "home_xg", "away_xg", "date"]).sort_values("date")
df["home_team"] = df["home_team"].map(canonical_name)
df["away_team"] = df["away_team"].map(canonical_name)

test = df[df["season"] == season].sort_values("date")
# weekly refit boundaries (Mondays)
weeks = test["date"].dt.to_period("W-SUN")
rows = []
t0 = time.time()
for wk, blk in test.groupby(weeks, sort=True):
    cutoff = blk["date"].min()
    train = df[df["date"] < cutoff]
    eng = PredictionEngine(**DEPLOYED_CLUB_ENGINE_KWARGS).fit(train)
    known = set(train["home_team"]) | set(train["away_team"])
    for _, r in blk.iterrows():
        h, a, comp = r.home_team, r.away_team, r.competition
        cur = set(test.loc[test.competition == comp, "home_team"])
        hs = [h] if h in known else [t for t in relegated_last_season(train, comp, season, cur) if t in known]
        as_ = [a] if a in known else [t for t in relegated_last_season(train, comp, season, cur) if t in known]
        comps = []
        for hh in hs:
            for aa in as_:
                if hh == aa:
                    continue
                lh_ad, la_ad = eng._lambdas_ad(hh, aa, comp, False)
                xr_h, xr_a = eng.xg_rate_model_.predict_lambda(hh, aa, neutral=False)
                comps.append((lh_ad, la_ad, xr_h, xr_a))
        c = np.mean(np.array(comps), axis=0)
        rows.append({"season": season, "match_id": r.match_id, "date": r.date, "comp": comp,
                     "home": h, "away": a, "hg": int(r.home_goals), "ag": int(r.away_goals),
                     "h_known": h in known, "a_known": a in known,
                     "lh_ad": c[0], "la_ad": c[1], "xr_h": c[2], "xr_a": c[3],
                     "w": eng.xg_rate_weight, "ls": eng.lambda_scale_})
        if eng.xg_rate_model_ is not None:
            eng.xg_rate_model_.update_form(h, a, r.home_xg, r.away_xg)
    print(f"{season} {wk} n={len(rows)} {time.time()-t0:.0f}s", flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
print("DONE", season, len(rows), flush=True)
