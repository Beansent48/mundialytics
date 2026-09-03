from __future__ import annotations

"""Temporal out-of-sample evaluation of the EUROPEAN competition layer.

The European layer is a pure Elo -> lambda model:

    lambda_home = exp(c + hfa + b * (elo_h - elo_a) / 400)
    lambda_away = exp(c       - b * (elo_h - elo_a) / 400)

with c, hfa, b shipped in data/processed/elo_lambda_calibration_euro.json. That
calibration was fitted on ~1000 European matches, which is most of the 2021-24
results in data/external/uefa/ — so scoring it on those is in-sample and proves
nothing. This refits c and hfa on the earlier seasons only (b is inherited from
the Big 5 fit, as it is in the shipped file) and scores the held-out season.

Elo is read point-in-time from the per-team ClubElo histories, so a match is
priced with the ratings that existed on its own date.

    python scripts/evaluate_european_layer.py [--test-season 2024]
"""

import argparse
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.competition.european import (  # noqa: E402
    make_resolver,
    normalize_club,
)

UEFA = ROOT / "data/external/uefa"
ELO_DIR = ROOT / "data/external/clubelo/teams"
MAXG = 10


def load_elo_history() -> dict[str, pd.DataFrame]:
    hist = {}
    for f in glob.glob(str(ELO_DIR / "*.csv")):
        d = pd.read_csv(f, usecols=["Club", "Elo", "From", "To"])
        d["From"] = pd.to_datetime(d["From"], errors="coerce")
        d["To"] = pd.to_datetime(d["To"], errors="coerce")
        d = d.dropna(subset=["From", "Elo"]).sort_values("From")
        if len(d):
            hist[str(d["Club"].iloc[0])] = d
    return hist


def elo_on(hist: dict[str, pd.DataFrame], club: str, when: pd.Timestamp) -> float | None:
    d = hist.get(club)
    if d is None:
        return None
    row = d[(d["From"] <= when) & (d["To"] >= when)]
    if len(row):
        return float(row["Elo"].iloc[-1])
    before = d[d["From"] <= when]
    return float(before["Elo"].iloc[-1]) if len(before) else None


def load_matches() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(UEFA / "raw_*.csv")):
        name = Path(f).stem                      # raw_champions-league_2023
        comp, season = name.split("_")[1], int(name.split("_")[2])
        d = pd.read_csv(f)
        if "Result" not in d.columns:
            continue
        for r in d.itertuples():
            res = str(getattr(r, "Result", "") or "")
            if "-" not in res:
                continue
            try:
                hg, ag = (int(x.strip()) for x in res.split("-")[:2])
            except ValueError:
                continue
            rows.append({
                "comp": comp, "season": season,
                "date": pd.to_datetime(getattr(r, "Date"), dayfirst=True, errors="coerce"),
                "home": getattr(r, "_5"), "away": getattr(r, "_6"),
                "hg": hg, "ag": ag,
            })
    return pd.DataFrame(rows).dropna(subset=["date"])


def lambdas(d400: np.ndarray, c: float, hfa: float, b: float):
    return np.exp(c + hfa + b * d400), np.exp(c - b * d400)


def nll(params, d400, hg, ag, b) -> float:
    c, hfa = params
    lh, la = lambdas(d400, c, hfa, b)
    lh, la = np.clip(lh, 1e-6, 12), np.clip(la, 1e-6, 12)
    return float(-(poisson.logpmf(hg, lh) + poisson.logpmf(ag, la)).sum())


def probs_1x2(lh: np.ndarray, la: np.ndarray) -> np.ndarray:
    g = np.arange(MAXG + 1)
    out = np.zeros((len(lh), 3))
    for i in range(len(lh)):
        ph, pa = poisson.pmf(g, lh[i]), poisson.pmf(g, la[i])
        m = np.outer(ph, pa)
        out[i] = [np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]
    return out / out.sum(axis=1, keepdims=True)


def rps3(y: np.ndarray, P: np.ndarray) -> float:
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    return float(((np.cumsum(P, 1) - np.cumsum(Y, 1)) ** 2)[:, :2].sum(1).mean() / 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-season", type=int, default=2024)
    args = ap.parse_args()

    hist = load_elo_history()
    m = load_matches()
    resolve = make_resolver(list(hist))

    keep = []
    for r in m.itertuples():
        h, a = resolve(normalize_club(r.home)), resolve(normalize_club(r.away))
        if not h or not a:
            continue
        eh, ea = elo_on(hist, h, r.date), elo_on(hist, a, r.date)
        if eh is None or ea is None:
            continue
        keep.append({"season": r.season, "comp": r.comp, "d400": (eh - ea) / 400.0,
                     "hg": r.hg, "ag": r.ag})
    df = pd.DataFrame(keep)
    print(f"{len(m)} European results parsed | {len(df)} priced with point-in-time Elo "
          f"({len(df) / max(len(m), 1):.0%} resolved)")

    b = 0.7393927801465834                     # inherited from the Big 5 fit
    train, test = df[df.season < args.test_season], df[df.season == args.test_season]
    if not len(test):
        print(f"no matches for season {args.test_season}")
        return
    print(f"train seasons {sorted(train.season.unique())} ({len(train)}) | "
          f"test {args.test_season} ({len(test)})")

    fit = minimize(nll, [0.2, 0.28], args=(train.d400.to_numpy(), train.hg.to_numpy(),
                                           train.ag.to_numpy(), b), method="Nelder-Mead")
    c, hfa = fit.x
    print(f"refit on train only: c={c:.4f} hfa={hfa:.4f} (shipped: c=0.2043 hfa=0.2806)")

    lh, la = lambdas(test.d400.to_numpy(), c, hfa, b)
    P = probs_1x2(lh, la)
    y = np.where(test.hg > test.ag, 0, np.where(test.hg == test.ag, 1, 2))
    rates = np.bincount(np.where(train.hg > train.ag, 0,
                                 np.where(train.hg == train.ag, 1, 2)), minlength=3) / len(train)

    print(f"\nheld-out season {args.test_season}, {len(test)} matches")
    print(f"  RPS Elo layer   {rps3(y, P):.4f}")
    print(f"  RPS base rates  {rps3(y, np.tile(rates, (len(y), 1))):.4f}")
    print("\n  calibration:")
    print(f"    mean lambda   {lh.mean():.2f}-{la.mean():.2f}"
          f"   (actual goals {test.hg.mean():.2f}-{test.ag.mean():.2f})")
    print(f"    draws pred.   {P[:, 1].mean():.3f}   (actual {(y == 1).mean():.3f})")
    print(f"    home wins     {P[:, 0].mean():.3f}   (actual {(y == 0).mean():.3f})")


if __name__ == "__main__":
    main()
