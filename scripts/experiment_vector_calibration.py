from __future__ import annotations

"""A) Home-advantage drift fix via walk-forward VECTOR calibration.

Measured bias of the deployed chain (10,403 walk-forward matches): home
predicted 44.7% vs 43.0% real (+1.7pp), away 29.4% vs 31.6% (-2.2pp) — the
post-COVID home-advantage decline the model hasn't fully tracked.

Candidate: multinomial logistic recalibration on the log-trio (the 3-class
generalization of Platt / 'vector scaling', the standard practice), fitted
each season ONLY on previous seasons' out-of-fold predictions of the same
deployed chain. Pure post-hoc layer — the engine is untouched.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_deployed.csv"
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]


def rps3(y_idx, P):
    Y = np.zeros_like(P)
    Y[np.arange(len(y_idx)), y_idx] = 1.0
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return float(((cp - cy) ** 2)[:, :2].sum(axis=1).mean() / 2)


def ll3(y_idx, P):
    return float(-np.log(np.clip(P[np.arange(len(y_idx)), y_idx], 1e-9, 1)).mean())


def main() -> None:
    w = pd.read_csv(CACHE)
    w["y"] = np.where(w.hg > w.ag, 0, np.where(w.hg == w.ag, 1, 2))
    seasons = sorted(w.season.unique())

    res = {"raw": {"rps": [], "ll": []}, "cal": {"rps": [], "ll": []}}
    bias_rows = []
    for s in TEST_SEASONS:
        te = w[w.season == s]
        tr = w[w.season < s]
        if len(te) == 0 or len(tr) < 1000:
            continue
        Xtr = np.log(np.clip(tr[["ph", "pd", "pa"]].to_numpy(float), 1e-9, 1))
        Xte = np.log(np.clip(te[["ph", "pd", "pa"]].to_numpy(float), 1e-9, 1))
        clf = LogisticRegression(C=1e3, max_iter=2000).fit(Xtr, tr["y"])
        P_raw = te[["ph", "pd", "pa"]].to_numpy(float)
        P_cal = clf.predict_proba(Xte)  # classes 0,1,2 in order
        y = te["y"].to_numpy()
        for tag, P in [("raw", P_raw), ("cal", P_cal)]:
            res[tag]["rps"].append((rps3(y, P), len(y), s))
            res[tag]["ll"].append((ll3(y, P), len(y), s))
        bias_rows.append({"season": s,
                          "home_pred_raw": P_raw[:, 0].mean(), "home_pred_cal": P_cal[:, 0].mean(),
                          "home_real": (y == 0).mean(),
                          "away_pred_raw": P_raw[:, 2].mean(), "away_pred_cal": P_cal[:, 2].mean(),
                          "away_real": (y == 2).mean()})

    pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
    print(f"n test = {sum(n for _, n, _ in res['raw']['rps'])}")
    for met in ["rps", "ll"]:
        folds = " ".join(f"{r[2][-2:]}{'+' if c[0] < r[0] else '-'}"
                         for r, c in zip(res["raw"][met], res["cal"][met]))
        print(f"{met.upper()}: raw {pool(res['raw'][met]):.4f} -> cal {pool(res['cal'][met]):.4f} "
              f"(d {pool(res['cal'][met]) - pool(res['raw'][met]):+.4f})  [{folds}]")
    b = pd.DataFrame(bias_rows)
    print("\nsesgo local (pred-real) raw vs cal por temporada:")
    b["home_bias_raw"] = b.home_pred_raw - b.home_real
    b["home_bias_cal"] = b.home_pred_cal - b.home_real
    b["away_bias_raw"] = b.away_pred_raw - b.away_real
    b["away_bias_cal"] = b.away_pred_cal - b.away_real
    print(b[["season", "home_bias_raw", "home_bias_cal", "away_bias_raw", "away_bias_cal"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
