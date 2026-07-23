from __future__ import annotations

"""END-TO-END validation of goal_temper=1.05 on the full deployed chain
(the SETPIECE lesson: distribution-level wins must survive the stacked chain —
sharpening, outcome-rho, rescale — before deployment).

Arms: deployed config vs deployed + goal_temper=1.05 (theta was train-stable
across folds in experiment_tempered_goals). Usual 5 folds, big-5 eval.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402

TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
DEPLOYED = dict(blend_weight_gl=0.30, sharpen_gamma_1x2=1.3,
                rescale_lambda_to_goals=True, outcome_rho=-0.17,
                xg_rate_kwargs={"use_ewma": True})


def rps3(y_idx, P):
    Y = np.zeros_like(P)
    Y[np.arange(len(y_idx)), y_idx] = 1.0
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return float(((cp - cy) ** 2)[:, :2].sum(axis=1).mean() / 2)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    df = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv",
                     low_memory=False)
    df = df[df["xg_available"] == True].copy()  # noqa: E712
    for c in ["home_goals", "away_goals", "home_xg", "away_xg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals", "home_xg", "away_xg", "date"]).sort_values("date")

    res = {arm: {m: [] for m in ["rps", "ll", "o15", "o25", "o35"]} for arm in ["BASE", "TEMPER"]}
    for s in TEST_SEASONS:
        te = df[df.season == s].sort_values("date")
        tr = df[df.date < te.date.min()]
        for arm, extra in [("BASE", {}), ("TEMPER", {"goal_temper": 1.05})]:
            t0 = time.time()
            eng = PredictionEngine(**DEPLOYED, **extra).fit(tr)
            P, y, po = [], [], {1.5: [], 2.5: [], 3.5: []}
            for _, r in te.iterrows():
                p = eng.predict_match(str(r.home_team), str(r.away_team),
                                      competition=str(r.competition), neutral=False)
                P.append([p.p_home_win, p.p_draw, p.p_away_win])
                y.append(0 if r.home_goals > r.away_goals else (1 if r.home_goals == r.away_goals else 2))
                po[1.5].append(p.p_over_15)
                po[2.5].append(p.p_over_25)
                po[3.5].append(p.p_over_35)
                if eng.xg_rate_model_ is not None:
                    eng.xg_rate_model_.update_form(r.home_team, r.away_team, r.home_xg, r.away_xg)
            P, y = np.array(P), np.array(y)
            tot = (te.home_goals + te.away_goals).to_numpy()
            res[arm]["rps"].append((rps3(y, P), len(y), s))
            res[arm]["ll"].append((float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean()), len(y), s))
            for ln, key in [(1.5, "o15"), (2.5, "o25"), (3.5, "o35")]:
                res[arm][key].append((bll((tot > ln).astype(float), np.array(po[ln])), len(y), s))
            print(f"  {s} {arm}: done ({time.time()-t0:.0f}s)", flush=True)

    pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
    print("\n===== VERDICT (end-to-end deployed chain) =====")
    for met, label in [("rps", "1X2 RPS"), ("ll", "1X2 LL"),
                       ("o15", "O/U 1.5 LL"), ("o25", "O/U 2.5 LL"), ("o35", "O/U 3.5 LL")]:
        b, t = res["BASE"][met], res["TEMPER"][met]
        folds = " ".join(f"{x[2][-2:]}{'+' if yv[0] < x[0] else '-'}" for x, yv in zip(b, t))
        print(f"{label}: BASE {pool(b):.4f} -> TEMPER {pool(t):.4f} (d {pool(t)-pool(b):+.4f})  [{folds}]")


if __name__ == "__main__":
    main()
