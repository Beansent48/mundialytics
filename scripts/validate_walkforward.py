from __future__ import annotations

"""Isolate the walk-forward-form gain: same fitted engine, predict the test season
FROZEN (form at training end) vs WALK-FORWARD (form updated after each match, in
date order). Leakage-safe: each match is predicted before its own result updates form.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402

TEST_SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
COVERED = ["2014-2015","2015-2016","2016-2017","2017-2018","2018-2019","2019-2020",
           "2020-2021","2021-2022","2022-2023","2023-2024","2024-2025","2025-2026"]


def rps_markets(preds):
    o = preds["oc"].to_numpy()
    ph, pdw, pa = preds.ph.to_numpy(float), preds.pd.to_numpy(float), preds.pa.to_numpy(float)
    yh, yd = (o == "home").astype(float), (o == "draw").astype(float)
    rps = (0.5 * ((ph - yh) ** 2 + (ph + pdw - yh - yd) ** 2)).mean()
    ap = np.where(o == "home", ph, np.where(o == "draw", pdw, pa))
    ll = -np.log(np.clip(ap, 1e-9, 1)).mean()
    acc = (np.argmax(np.c_[ph, pdw, pa], axis=1) == np.where(o == "home", 0, np.where(o == "draw", 1, 2))).mean()
    return {"rps": float(rps), "ll": float(ll), "acc": float(acc), "n": len(preds)}


def main() -> None:
    df = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv", low_memory=False)
    df = df[(df["season"].isin(COVERED)) & (df["xg_available"] == True)].copy()  # noqa: E712
    for c in ["home_goals","away_goals","home_xg","away_xg"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_goals","away_goals","home_xg","away_xg","date"]).sort_values("date")
    hg, ag = df.home_goals.astype(int), df.away_goals.astype(int)
    df["oc"] = np.where(hg > ag, "home", np.where(hg < ag, "away", "draw"))

    def predict_row(engine, r):
        p = engine.predict_match(str(r.home_team), str(r.away_team), competition=str(r.competition), neutral=bool(r.get("neutral", 0)))
        return {"ph": p.p_home_win, "pd": p.p_draw, "pa": p.p_away_win, "oc": r.oc}

    fro, wf = [], []
    print(f"{'season':10s} {'frozen':>8s} {'walkfwd':>8s} {'dRPS':>8s}")
    for s in TEST_SEASONS:
        test = df[df["season"] == s].sort_values("date"); train = df[df["date"] < test["date"].min()]
        if len(test) == 0 or len(train) < 500: continue
        t0 = time.time()
        eng = PredictionEngine(use_xg_rate=True, blend_weight_gl=0.30).fit(train)
        # FROZEN pass (no form updates)
        frp = pd.DataFrame([predict_row(eng, r) for _, r in test.iterrows()])
        mf = rps_markets(frp)
        # WALK-FORWARD pass (form starts at training end, updates after each match in date order)
        rows = []
        for _, r in test.iterrows():
            rows.append(predict_row(eng, r))
            if eng.xg_rate_model_ is not None:
                eng.xg_rate_model_.update_form(r.home_team, r.away_team, r.home_xg, r.away_xg)
        mw = rps_markets(pd.DataFrame(rows))
        fro.append(mf); wf.append(mw)
        print(f"{s:10s} {mf['rps']:8.4f} {mw['rps']:8.4f} {mw['rps']-mf['rps']:+8.4f}   ({time.time()-t0:.0f}s)", flush=True)

    def pool(a, k): return sum(m[k]*m["n"] for m in a)/sum(m["n"] for m in a)
    print("\n===== POOLED =====")
    for k in ["rps", "ll", "acc"]:
        print(f"  {k:4s}  frozen {pool(fro,k):.4f}   walk-forward {pool(wf,k):.4f}   delta {pool(wf,k)-pool(fro,k):+.4f}")


if __name__ == "__main__":
    main()
