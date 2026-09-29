"""End-to-end check: the real PredictionEngine with squad_value_shift on, weekly refit,
squad values read as of each refit date. Compares its 1X2 to the flag-off components
from wf_refit.py for the same season, and to the offline shift computed from them.

    python scripts/squad_value/validate_engine.py 2025-2026
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS, PredictionEngine, deployed_markets_from_lambdas)
from mundialytics.statistical_core.schemas import canonical_name  # noqa: E402

SEASON = sys.argv[1] if len(sys.argv) > 1 else "2025-2026"
SHIFT = {"beta": 0.1051, "kappa": -0.2148}
W = "data/external/transfermarkt/work/"

df = pd.read_csv("data/processed/enriched/understat_xg/canonical_matches_with_xg.csv", low_memory=False)
df = df[df["xg_available"] == True].copy()  # noqa: E712
for c in ["home_goals", "away_goals", "home_xg", "away_xg"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["home_goals", "away_goals", "home_xg", "away_xg", "date"]).sort_values("date")
df["home_team"] = df["home_team"].map(canonical_name)
df["away_team"] = df["away_team"].map(canonical_name)
test = df[df["season"] == SEASON]
base = pd.read_csv(W + f"wfr_{SEASON}.csv").set_index("match_id")

rows = []
t0 = time.time()
for wk, blk in test.groupby(test["date"].dt.to_period("W-SUN"), sort=True):
    cutoff = blk["date"].min()
    train = df[df["date"] < cutoff]
    eng = PredictionEngine(**{**DEPLOYED_CLUB_ENGINE_KWARGS, "squad_value_shift": SHIFT})
    eng.fit(train, squad_value_asof=cutoff - pd.Timedelta(days=1))
    known = set(train["home_team"]) | set(train["away_team"])
    for r in blk.itertuples(index=False):
        if r.home_team in known and r.away_team in known and r.match_id in base.index:
            p = eng.predict_match(r.home_team, r.away_team, competition=r.competition)
            b = base.loc[r.match_id]
            lh0 = np.clip((b.w * b.xr_h + (1 - b.w) * b.lh_ad) * b.ls, 0.05, 6.0)
            la0 = np.clip((b.w * b.xr_a + (1 - b.w) * b.la_ad) * b.ls, 0.05, 6.0)
            q, _ = deployed_markets_from_lambdas(lh0, la0)
            rows.append({"match_id": r.match_id, "hg": int(r.home_goals), "ag": int(r.away_goals),
                         "ph": p.p_home_win, "pd": p.p_draw, "pa": p.p_away_win,
                         "ph0": q["p_home_win"], "pd0": q["p_draw"], "pa0": q["p_away_win"],
                         "tilted": p.model_source.endswith("_sv"), "src": p.model_source})
        eng.xg_rate_model_.update_form(r.home_team, r.away_team, r.home_xg, r.away_xg)
    print(f"{wk} n={len(rows)} {time.time() - t0:.0f}s", flush=True)

x = pd.DataFrame(rows)
o = np.where(x.hg > x.ag, 0, np.where(x.hg == x.ag, 1, 2))
Y = np.eye(3)[o]


def rps(P):
    return ((((np.cumsum(P, 1) - np.cumsum(Y, 1)) ** 2)[:, :2].sum(1)) / 2).mean()


print(f"\n{SEASON}: n={len(x)}  tilted {x.tilted.mean():.1%}")
print(f"RPS flag off {rps(x[['ph0', 'pd0', 'pa0']].values):.5f}  ->  engine with shift {rps(x[['ph', 'pd', 'pa']].values):.5f}")
x.to_csv(W + f"validate_engine_{SEASON}.csv", index=False)
