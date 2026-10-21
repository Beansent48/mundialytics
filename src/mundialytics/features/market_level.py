"""The goals level the betting market implies for a league-season.

Why this exists. Reacting to our OWN in-season residual fails -- shots
(v0.56.0, 2/5 folds) and goals (scripts/goals_level/, 1/6), both the same way:
in a normal season the residual wobbles +/-2% with no persistence, so any
shrinkage fast enough to catch 2026/27's +12.9% buys that noise in the other
six. The estimator was the problem, not the correction. The market's implied
level is essentially unbiased every season (0.989-1.019 across 2020/21-2025/26)
and in 2026/27 it is 4.5% short where we are 12.9% short -- the book saw two
thirds of the shift.

Validated at K=70 on six historical folds: over/under 2.5 log loss -0.00047
+/- 0.00025 (t = -1.90), 5/6 folds, 1X2 untouched (-0.00000, 4/6); 2026/27
-0.00538. Flat across K=40-400 and robust to the odds source (B365 closing,
B365 opening, market average all 5/6). A hierarchical version, shrinking the
league toward the pooled level, was better on 2026/27 and worse on history
(4/6) -- not deployed. See scripts/market_level/README.md.

**The line this does not cross.** Only a league-season LEVEL comes from the
market, from matches that had already been played; never the price of the match
being predicted, and never a price from its own date. All per-match
discrimination stays the model's.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FD = ROOT / "data/raw/football_data"
SERVED = ROOT / "data/processed/served_lambdas_current.csv"
PRED_LOG = ROOT / "data/processed/logs/predictions_log.csv"
# football-data's league codes -> the foundation's competition names
LEAGUES = {"E0": "Premier League", "SP1": "LaLiga", "D1": "Bundesliga",
           "I1": "Serie A", "F1": "Ligue 1"}
CLIP = (0.90, 1.10)


def book_over25(seasons: tuple[str, ...] | None = None) -> pd.DataFrame:
    """fecha + raw team names + competition -> de-vigged P(over 2.5), B365 closing.

    Falls back to the market average and then to the opening line per file; all
    three measured 5/6, so a missing column is no reason to drop a league.
    """
    rows = []
    for p in sorted(FD.glob("*.csv")):
        stem = p.stem
        if len(stem) < 5 or not stem[:4].isdigit():
            continue
        if seasons and stem[:4] not in seasons:
            continue
        comp = LEAGUES.get(stem.split("_")[-1])
        if comp is None:
            continue
        try:
            o = pd.read_csv(p, low_memory=False, encoding="latin-1")
        except (OSError, UnicodeDecodeError, pd.errors.ParserError):
            continue
        pair = None
        for hi, lo in (("B365C>2.5", "B365C<2.5"), ("AvgC>2.5", "AvgC<2.5"),
                       ("B365>2.5", "B365<2.5")):
            if {hi, lo, "Date", "HomeTeam", "AwayTeam"} <= set(o.columns):
                pair = (hi, lo)
                break
        if pair is None:
            continue
        hi, lo = pair
        o = o.dropna(subset=[hi, lo, "HomeTeam", "AwayTeam", "Date"])
        if o.empty:
            continue
        io_, iu = 1.0 / pd.to_numeric(o[hi], errors="coerce"), \
            1.0 / pd.to_numeric(o[lo], errors="coerce")
        rows.append(pd.DataFrame({
            "fecha": pd.to_datetime(o["Date"], dayfirst=True,
                                    errors="coerce").dt.strftime("%Y-%m-%d"),
            "home_team_raw": o["HomeTeam"].astype(str),
            "away_team_raw": o["AwayTeam"].astype(str),
            "competition": comp,
            "p_book": (io_ / (io_ + iu)).to_numpy()}))
    if not rows:
        return pd.DataFrame(columns=["fecha", "home_team_raw", "away_team_raw",
                                     "competition", "p_book"])
    out = pd.concat(rows, ignore_index=True).dropna(subset=["fecha", "p_book"])
    return out.drop_duplicates(["fecha", "home_team_raw", "away_team_raw"])


def served_lambdas() -> pd.DataFrame:
    """fecha + canonical teams -> the lambdas we actually served, current season.

    Two sources, in order: a ONE-OFF backfill for the part of the season that
    predates PR #25 (`scripts/build_served_lambdas_current.py`), and the
    pre-kickoff log, which has carried lambda_home/lambda_away since then and
    grows every day. Nothing needs regenerating: the backfill covers up to the
    log's first row and the log covers everything after it, so the level keeps
    strengthening on its own as the season goes.

    Reconstructing these with today's fit instead would not be what was
    validated: the harness used the lambda the model serving that week produced.
    """
    frames = []
    if SERVED.exists():
        try:
            c = pd.read_csv(SERVED)
            if {"fecha", "home", "away", "lh", "la"} <= set(c.columns):
                frames.append(c[["fecha", "home", "away", "lh", "la"]])
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            pass
    if PRED_LOG.exists():
        try:
            lg = pd.read_csv(PRED_LOG, low_memory=False)
            lg = lg[(lg.get("mercado") == "1X2") & lg.get("lambda_home").notna()]
            if len(lg):
                frames.append(lg.rename(columns={"lambda_home": "lh",
                                                 "lambda_away": "la"})
                              [["fecha", "home", "away", "lh", "la"]])
        except (OSError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError):
            pass
    if not frames:
        return pd.DataFrame(columns=["fecha", "home", "away", "lh", "la"])
    out = pd.concat(frames, ignore_index=True).dropna(subset=["lh", "la"])
    # cache first, so a regenerated walk-forward wins over a logged row
    return out.drop_duplicates(["fecha", "home", "away"], keep="first")


def _implied_scale(lh: np.ndarray, la: np.ndarray, p_book: np.ndarray,
                   iters: int = 14) -> np.ndarray:
    """s such that the deployed chain on (lh*s, la*s) prices the book's P(over 2.5).

    Bisection on the whole vector: P(over) rises monotonically in a common
    multiplier, so 14 halvings of [0.3, 3.0] land inside 2e-4 -- far below the
    shrinkage that follows. It calls the real deployed chain rather than a copy of
    it: a reimplementation drifting from what we serve is the bug this audit kept
    finding.
    """
    from mundialytics.statistical_core.prediction_engine import (
        DEPLOYED_CLUB_ENGINE_KWARGS as K, markets_from_lambdas)

    kw = dict(rho=K.get("outcome_rho", K.get("ad_rho", 0.0)),
              temper=K.get("goal_temper", 1.0),
              sharpen_gamma=float(np.clip(K.get("sharpen_gamma_1x2", 1.0), 0.5, 2.0)),
              sharpen_contrast_only=bool(K.get("sharpen_contrast_only", False)),
              coherent=bool(K.get("coherent_markets", False)))

    def p_over(s):
        return np.array([markets_from_lambdas(float(a * si), float(b * si), **kw)[0]["p_over_25"]
                         for a, b, si in zip(lh, la, s)])

    lo, hi = np.full(len(lh), 0.3), np.full(len(lh), 3.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        below = p_over(mid) < p_book
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo + hi) / 2


def market_levels(matches: pd.DataFrame, asof, k: float = 70.0) -> dict[str, float]:
    """{competition: level multiplier} -- see market_level_detail."""
    return {c: m for c, (_, m) in market_level_detail(matches, asof, k).items()}


def market_level_detail(matches: pd.DataFrame, asof,
                        k: float = 70.0) -> dict[str, tuple[int, float]]:
    """{competition: (matches measured on, level multiplier)}.

    Only matches played strictly before `asof` contribute, and the multiplier is
    shrunk to 1 with `k` pseudo-matches, so a league with little history barely
    moves. Returns {} when there is nothing to measure, which leaves the lambdas
    untouched.

    The count is returned so the daily refresh can print it: the correction can
    only ever weaken silently -- if the pre-kickoff log stopped carrying
    lambda_home/lambda_away, n would fall and nothing else would complain.
    """
    if matches is None or matches.empty or "date" not in matches.columns:
        return {}
    asof = pd.Timestamp(asof) if asof is not None else \
        pd.to_datetime(matches["date"], errors="coerce").max()
    if pd.isna(asof):
        return {}
    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    season = str(m.loc[m["date"].idxmax(), "season"]) if "season" in m.columns else ""
    m = m[(m["date"] < asof) & (m["season"].astype(str) == season)] if season else \
        m[m["date"] < asof]
    if m.empty or "home_team_raw" not in m.columns:
        return {}
    m["fecha"] = m["date"].dt.strftime("%Y-%m-%d")
    lam = served_lambdas()
    if lam.empty:
        return {}
    j = (m[["fecha", "home_team", "away_team", "home_team_raw", "away_team_raw",
            "competition"]]
         .rename(columns={"home_team": "home", "away_team": "away"})
         .merge(lam, on=["fecha", "home", "away"], how="inner"))
    if j.empty:
        return {}
    j = j.merge(book_over25((season[2:4] + season[7:9],) if len(season) == 9 else None),
                on=["fecha", "home_team_raw", "away_team_raw", "competition"],
                how="inner")
    if j.empty:
        return {}
    j["s"] = _implied_scale(j.lh.to_numpy(float), j.la.to_numpy(float),
                            j.p_book.to_numpy(float))
    g = j.groupby("competition")["s"].agg(["sum", "size"])
    return {str(c): (int(r["size"]),
                     float(np.clip((r["sum"] + k) / (r["size"] + k), *CLIP)))
            for c, r in g.iterrows()}
