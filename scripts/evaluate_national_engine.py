from __future__ import annotations

"""Temporal out-of-sample evaluation of the NATIONAL-team engine.

Trains on internationals up to a cutoff and scores everything after it, against
the same league-base-rate anchor the club benchmark uses. Written because the
national scope had never been measured — only the club engine had — and the
README was claiming both worked equally well.

    python scripts/evaluate_national_engine.py [--cutoff 2023-01-01]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.ratings.elo import EloConfig, EloRater  # noqa: E402
from mundialytics.statistical_core.engine_utils import load_international_data  # noqa: E402
from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402


def rps3(y_idx: np.ndarray, P: np.ndarray) -> float:
    """Mean RPS over the ordered outcomes home/draw/away."""
    Y = np.zeros_like(P)
    Y[np.arange(len(y_idx)), y_idx] = 1.0
    return float(((np.cumsum(P, 1) - np.cumsum(Y, 1)) ** 2)[:, :2].sum(1).mean() / 2)


def outcomes(hg: pd.Series, ag: pd.Series) -> np.ndarray:
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2023-01-01")
    ap.add_argument("--min-year", type=int, default=2010)
    args = ap.parse_args()

    d = load_international_data(min_year=args.min_year).sort_values("date")
    train, test = d[d.date < args.cutoff], d[d.date >= args.cutoff]
    print(f"train {len(train)} internationals (<{args.cutoff}) | test {len(test)}")

    # same configuration the Streamlit app builds for the national scope
    elo = EloRater(EloConfig(season_reset_fraction=0.35, k_base=28.0))
    elo.fit(train)
    eng = PredictionEngine(blend_weight_gl=0.45, ad_rho=-0.06)
    eng.fit(train, elo_history=pd.DataFrame(elo.history))

    rows = []
    for r in test.itertuples():
        try:
            # pass the real competition: AttackDefenseModel keeps a per-competition
            # mu and home advantage, and an unknown name silently falls to index 0
            p = eng.predict_match(r.home_team, r.away_team,
                                  competition=str(getattr(r, "competition", "") or ""),
                                  neutral=bool(getattr(r, "neutral", 0) or 0))
        except Exception:
            continue
        rows.append((p.p_home_win, p.p_draw, p.p_away_win,
                     r.home_goals, r.away_goals, p.lambda_home, p.lambda_away))

    a = pd.DataFrame(rows, columns=["ph", "pd", "pa", "hg", "ag", "lh", "la"])
    y = outcomes(a.hg, a.ag)
    P = a[["ph", "pd", "pa"]].to_numpy(float)
    rates = np.bincount(outcomes(train.home_goals, train.away_goals), minlength=3) / len(train)

    print(f"\nscored {len(a)} matches")
    print(f"  RPS engine          {rps3(y, P):.4f}")
    print(f"  RPS base rates      {rps3(y, np.tile(rates, (len(y), 1))):.4f}")
    print("\n  calibration:")
    print(f"    mean lambda       {a.lh.mean():.2f}-{a.la.mean():.2f}"
          f"   (actual goals {a.hg.mean():.2f}-{a.ag.mean():.2f})")
    print(f"    draws predicted   {P[:, 1].mean():.3f}   (actual {(y == 1).mean():.3f})")
    print(f"    home wins pred.   {P[:, 0].mean():.3f}   (actual {(y == 0).mean():.3f})")


if __name__ == "__main__":
    main()
