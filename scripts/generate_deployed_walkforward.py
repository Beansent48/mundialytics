from __future__ import annotations

"""Walk-forward predictions of the FULL DEPLOYED chain (sharpening, outcome
rho, lambda rescale, EWMA) for the Resultados page — the old cache
(walkforward_preds.csv) was generated with a bare engine for other purposes
and does NOT reflect served probabilities."""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402

SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
OUT = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_deployed.csv"


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
            continue
        t0 = time.time()
        eng = PredictionEngine(blend_weight_gl=0.30, sharpen_gamma_1x2=1.3,
                               rescale_lambda_to_goals=True, outcome_rho=-0.17,
                               xg_rate_kwargs={"use_ewma": True}).fit(train)
        for _, r in test.iterrows():
            p = eng.predict_match(str(r.home_team), str(r.away_team),
                                  competition=str(r.competition), neutral=bool(r.get("neutral", 0)))
            rows.append({"season": s, "match_id": r.match_id, "hg": int(r.home_goals),
                         "ag": int(r.away_goals), "ph": p.p_home_win, "pd": p.p_draw,
                         "pa": p.p_away_win, "po25": p.p_over_25})
            if eng.xg_rate_model_ is not None:
                eng.xg_rate_model_.update_form(r.home_team, r.away_team, r.home_xg, r.away_xg)
        print(f"  {s}: {len(test)} matches ({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"WROTE {OUT} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
