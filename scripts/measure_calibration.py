from __future__ import annotations

"""Step 1 of the calibration work: MEASURE the deployed walk-forward model's
reliability (only add a calibration layer if a miscalibration is actually there —
the title-forecast lesson). Runs the deployed engine walk-forward over the temporal
folds, caches per-match predictions, and prints reliability curves + ECE for
1X2 / Over2.5 / BTTS. Isolated: no engine changes.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402

TEST_SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
COVERED = ["2014-2015","2015-2016","2016-2017","2017-2018","2018-2019","2019-2020",
           "2020-2021","2021-2022","2022-2023","2023-2024","2024-2025","2025-2026"]
CACHE = ROOT / "data/processed/enriched/understat_xg/walkforward_preds.csv"


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        gap = p[m].mean() - y[m].mean()
        ece += m.sum() / len(p) * abs(gap)
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                     "pred": round(float(p[m].mean()), 3), "emp": round(float(y[m].mean()), 3),
                     "gap": round(float(gap), 3)})
    return pd.DataFrame(rows), float(ece)


def main() -> None:
    if CACHE.exists():
        preds = pd.read_csv(CACHE)
        print(f"Reusing cached walk-forward predictions: {len(preds)} matches\n")
    else:
        df = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv", low_memory=False)
        df = df[(df["season"].isin(COVERED)) & (df["xg_available"] == True)].copy()  # noqa: E712
        for c in ["home_goals","away_goals","home_xg","away_xg"]: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["home_goals","away_goals","home_xg","away_xg","date"]).sort_values("date")
        rows = []
        for s in TEST_SEASONS:
            test = df[df["season"] == s].sort_values("date"); train = df[df["date"] < test["date"].min()]
            if len(test) == 0 or len(train) < 500: continue
            t0 = time.time()
            eng = PredictionEngine(use_xg_rate=True, blend_weight_gl=0.30).fit(train)
            for _, r in test.iterrows():
                p = eng.predict_match(str(r.home_team), str(r.away_team), competition=str(r.competition), neutral=bool(r.get("neutral", 0)))
                rows.append({"season": s, "match_id": r.match_id, "hg": int(r.home_goals), "ag": int(r.away_goals),
                             "ph": p.p_home_win, "pd": p.p_draw, "pa": p.p_away_win,
                             "po25": p.p_over_25, "pbtts": p.p_btts,
                             "lh": p.lambda_home, "la": p.lambda_away})
                if eng.xg_rate_model_ is not None:
                    eng.xg_rate_model_.update_form(r.home_team, r.away_team, r.home_xg, r.away_xg)
            print(f"  {s}: {len(test)} matches ({time.time()-t0:.0f}s)", flush=True)
        preds = pd.DataFrame(rows)
        preds.to_csv(CACHE, index=False)
        print(f"WROTE {CACHE}\n")

    o = np.where(preds.hg > preds.ag, "home", np.where(preds.hg < preds.ag, "away", "draw"))
    y1 = np.concatenate([(o == "home").astype(float), (o == "draw").astype(float), (o == "away").astype(float)])
    p1 = np.concatenate([preds.ph.to_numpy(float), preds.pd.to_numpy(float), preds.pa.to_numpy(float)])
    tab, ece = reliability(y1, p1)
    print("===== 1X2 (pooled selections) =====")
    print(tab.to_string(index=False)); print(f"  ECE = {ece:.4f}   (<0.02 well-calibrated)")

    over = ((preds.hg + preds.ag) > 2.5).astype(float).to_numpy()
    tab, ece_o = reliability(over, preds.po25.to_numpy(float))
    print("\n===== Over 2.5 =====")
    print(tab.to_string(index=False)); print(f"  ECE = {ece_o:.4f}")

    btts = ((preds.hg > 0) & (preds.ag > 0)).astype(float).to_numpy()
    tab, ece_b = reliability(btts, preds.pbtts.to_numpy(float))
    print("\n===== BTTS =====")
    print(tab.to_string(index=False)); print(f"  ECE = {ece_b:.4f}")

    print("\nVERDICT: add a calibration layer only for markets with ECE clearly > 0.02 "
          "and a systematic (monotone-fixable) gap pattern.")


if __name__ == "__main__":
    main()
