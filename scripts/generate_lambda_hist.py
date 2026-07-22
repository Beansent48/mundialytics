from __future__ import annotations

"""Generate walk-forward engine lambdas for 2016/17-2019/20 (the seasons the
existing walkforward_preds.csv cache does NOT cover — it starts 2020/21).

Purpose: the ASYM supremacy features in team props currently use raw goal EWMs;
the deployed engine's lambdas are a much sharper strength signal. To A/B that
WITHOUT leakage the training seasons need walk-forward lambdas too. Same engine
config as measure_calibration.py so the lambda scale is consistent with the
existing cache. Writes a SEPARATE file; the deployed cache is never touched.
"""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402

SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020"]
OUT = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_hist.csv"


def main() -> None:
    df = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv",
                     low_memory=False)
    df = df[df["xg_available"] == True].copy()  # noqa: E712
    for c in ["home_goals", "away_goals", "home_xg", "away_xg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals", "home_xg", "away_xg", "date"]).sort_values("date")

    rows = []
    for s in SEASONS:
        test = df[df["season"] == s].sort_values("date")
        train = df[df["date"] < test["date"].min()]
        if len(test) == 0 or len(train) < 500:
            print(f"  {s}: skipped (train {len(train)})", flush=True)
            continue
        t0 = time.time()
        eng = PredictionEngine(use_xg_rate=True, blend_weight_gl=0.30).fit(train)
        for _, r in test.iterrows():
            p = eng.predict_match(str(r.home_team), str(r.away_team),
                                  competition=str(r.competition), neutral=bool(r.get("neutral", 0)))
            rows.append({"season": s, "match_id": r.match_id, "lh": p.lambda_home, "la": p.lambda_away})
            if eng.xg_rate_model_ is not None:
                eng.xg_rate_model_.update_form(r.home_team, r.away_team, r.home_xg, r.away_xg)
        print(f"  {s}: {len(test)} matches ({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"WROTE {OUT} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
