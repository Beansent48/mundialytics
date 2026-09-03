from __future__ import annotations

"""Out-of-sample evaluation of the HALF-TIME markets.

The half-time layer is a stateless transformation of the engine's full-time
lambdas: first-half goals are the full-time lambda times a global share (see
props/half_time.py for why a per-team share was rejected). So the honest test is
to feed it the walk-forward lambdas the club benchmark already uses — leakage-safe
by construction — and score the result against the real half-time scores carried
in the foundation.

Baseline is the league base rate for each market, which is exactly what a global
share ought to beat if it is worth anything.

    python scripts/evaluate_half_time.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.props.half_time import HalfTimeModel  # noqa: E402

PREDS = ROOT / "data/processed/enriched/understat_xg/walkforward_preds.csv"
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"


def rps3(y: np.ndarray, P: np.ndarray) -> float:
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    return float(((np.cumsum(P, 1) - np.cumsum(Y, 1)) ** 2)[:, :2].sum(1).mean() / 2)


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    tot = 0.0
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def main() -> None:
    preds = pd.read_csv(PREDS)
    found = pd.read_csv(FOUND, low_memory=False,
                        usecols=["match_id", "home_goals_ht", "away_goals_ht"])
    d = preds.merge(found, on="match_id", how="inner").dropna(
        subset=["lh", "la", "home_goals_ht", "away_goals_ht"])
    print(f"{len(preds)} walk-forward predictions | {len(d)} with a half-time score")

    ht = HalfTimeModel()
    rows = [ht.predict_half_time(float(r.lh), float(r.la)) for r in d.itertuples()]
    P = np.array([[r["p_home"], r["p_draw"], r["p_away"]] for r in rows])
    lines = sorted(rows[0]["over"])

    hg, ag = d.home_goals_ht.to_numpy(int), d.away_goals_ht.to_numpy(int)
    y = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    rates = np.bincount(y, minlength=3) / len(y)

    print("\n===== HALF-TIME 1X2 =====")
    print(f"  RPS model      {rps3(y, P):.4f}")
    print(f"  RPS base rates {rps3(y, np.tile(rates, (len(y), 1))):.4f}")
    print(f"  predicted H/D/A {P.mean(0)[0]:.3f}/{P.mean(0)[1]:.3f}/{P.mean(0)[2]:.3f}"
          f"   actual {rates[0]:.3f}/{rates[1]:.3f}/{rates[2]:.3f}")

    print("\n===== HALF-TIME OVER/UNDER =====")
    tot = hg + ag
    for ln in lines:
        p = np.array([r["over"][ln] for r in rows])
        yb = (tot > ln).astype(float)
        base = np.full_like(p, yb.mean())
        print(f"  over {ln:<4} model LL {logloss(yb, p):.4f} | base {logloss(yb, base):.4f}"
              f" | delta {logloss(yb, p) - logloss(yb, base):+.4f}"
              f" | ECE {ece(yb, p):.4f} | pred {p.mean():.3f} actual {yb.mean():.3f}")


if __name__ == "__main__":
    main()
