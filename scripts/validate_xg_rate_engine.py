from __future__ import annotations

"""Honest end-to-end validation of the xG-rate engine as it will actually deploy
(frozen per-team form cache), vs the previous blended engine, on the same folds.
Reports RPS / log-loss / Brier / accuracy / O-U / BTTS per config, pooled + per fold.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_az = importlib.util.spec_from_file_location("az", ROOT / "scripts" / "analyze_xg_components.py")
az = importlib.util.module_from_spec(_az); _az.loader.exec_module(az)
from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402

TEST_SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
COVERED = az.__dict__.get("COVERED_SEASONS", None)


def predict_all(engine, test):
    rows = []
    for _, r in test.iterrows():
        p = engine.predict_match(str(r["home_team"]), str(r["away_team"]),
                                 competition=str(r.get("competition", "unknown")), neutral=bool(r.get("neutral", 0)))
        rows.append({"p_home_win": p.p_home_win, "p_draw": p.p_draw, "p_away_win": p.p_away_win,
                     "p_over_25": p.p_over_25, "p_btts": p.p_btts,
                     "actual_outcome": r["actual_outcome"], "actual_over_25": r["actual_over_25"],
                     "actual_btts": r["actual_btts"]})
    return pd.DataFrame(rows)


def metrics(pr):
    o = pr["actual_outcome"].to_numpy()
    ph, pdw, pa = pr.p_home_win.to_numpy(float), pr.p_draw.to_numpy(float), pr.p_away_win.to_numpy(float)
    yh, yd = (o == "home").astype(float), (o == "draw").astype(float)
    rps = (0.5 * ((ph - yh) ** 2 + (ph + pdw - yh - yd) ** 2)).mean()
    ap = np.where(o == "home", ph, np.where(o == "draw", pdw, pa))
    ll = -np.log(np.clip(ap, 1e-9, 1)).mean()
    acc = (np.argmax(np.c_[ph, pdw, pa], axis=1) == np.where(o == "home", 0, np.where(o == "draw", 1, 2))).mean()
    over = pr.actual_over_25.to_numpy(float); po = np.clip(pr.p_over_25.to_numpy(float), 1e-9, 1 - 1e-9)
    btts = pr.actual_btts.to_numpy(float); pb = np.clip(pr.p_btts.to_numpy(float), 1e-9, 1 - 1e-9)
    llo = -(over * np.log(po) + (1 - over) * np.log(1 - po)).mean()
    llb = -(btts * np.log(pb) + (1 - btts) * np.log(1 - pb)).mean()
    return {"rps": float(rps), "ll": float(ll), "acc": float(acc), "llo": float(llo), "llb": float(llb), "n": len(pr)}


def main() -> None:
    df = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv", low_memory=False)
    covered = ["2014-2015","2015-2016","2016-2017","2017-2018","2018-2019","2019-2020",
               "2020-2021","2021-2022","2022-2023","2023-2024","2024-2025","2025-2026"]
    df = df[(df["season"].isin(covered)) & (df["xg_available"] == True)].copy()  # noqa: E712
    for c in ["home_goals","away_goals","home_xg","away_xg"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_goals","away_goals","date"]).sort_values("date")
    hg, ag = df.home_goals.astype(int), df.away_goals.astype(int)
    df["actual_outcome"] = np.where(hg > ag, "home", np.where(hg < ag, "away", "draw"))
    df["actual_over_25"] = ((hg + ag) > 2.5).astype(int); df["actual_btts"] = ((hg > 0) & (ag > 0)).astype(int)

    agg = {"xr": [], "old": []}
    print(f"{'season':10s} {'xr_RPS':>8s} {'old_RPS':>8s} {'dRPS':>8s}")
    for s in TEST_SEASONS:
        test = df[df["season"] == s]; train = df[df["date"] < test["date"].min()]
        if len(test) == 0 or len(train) < 500: continue
        t0 = time.time()
        eng_xr = PredictionEngine(use_xg_rate=True, blend_weight_gl=0.30).fit(train)
        eng_old = PredictionEngine(use_xg_rate=False, use_xg=False, blend_weight_gl=0.30).fit(train)
        mx = metrics(predict_all(eng_xr, test)); mo = metrics(predict_all(eng_old, test))
        agg["xr"].append(mx); agg["old"].append(mo)
        print(f"{s:10s} {mx['rps']:8.4f} {mo['rps']:8.4f} {mx['rps']-mo['rps']:+8.4f}   ({time.time()-t0:.0f}s)", flush=True)

    def pool(a, k): return sum(m[k] * m["n"] for m in a) / sum(m["n"] for m in a)
    print("\n===== POOLED (weighted) =====")
    print(f"{'metric':6s} {'xG-rate':>10s} {'old blend':>10s} {'delta':>10s}")
    for k in ["rps", "ll", "acc", "llo", "llb"]:
        x, o = pool(agg["xr"], k), pool(agg["old"], k)
        print(f"{k:6s} {x:10.4f} {o:10.4f} {x-o:+10.4f}")
    per = [m["rps"] for m in agg["xr"]]
    print(f"\nxG-rate per-fold RPS spread: min {min(per):.4f} max {max(per):.4f} std {np.std(per):.4f}")


if __name__ == "__main__":
    main()
