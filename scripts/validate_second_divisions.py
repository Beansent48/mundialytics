from __future__ import annotations

"""Walk-forward validation of the engine on SECOND divisions + market benchmark.

Why this exists: the market-gap diagnosis ([[market-gap-diagnosis]]) showed we
are ~3% behind Bet365 CLOSING on Big5 1X2 and that the remaining gap is not
recoverable from public data (calibration, home advantage and even an ORACLE
lineup feed were all tested and closed). The open question is whether the same
engine sits CLOSER to the market where bookmakers price less sharply.

So this measures two things on E1/SP2/D2/I2/F2:
  1. our own RPS/log-loss walk-forward, season by season;
  2. the gap to Bet365 closing odds, side by side with the Big5 gap.

If the gap is materially smaller here, second divisions are where the edge is
and wiring them into the app is worth it. If it is the same or worse, say so.

Note: second divisions have NO xG (Understat covers top flights only), so the
engine runs without the xG-rate component — which is exactly what could be
deployed there, making this an honest like-for-like.

Nothing here is deployed; the Big5 foundation and models are untouched.
"""

import glob
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402
from diagnose_market_gap import rps3  # noqa: E402

FOUND2 = ROOT / "data/processed/foundation_second_divisions.csv"
OUT = ROOT / "data/processed/walkforward_preds_second_divisions.csv"
SECOND_CODES = "E1|SP2|D2|I2|F2"
WANT = {"Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A",
        "B365CH", "B365CD", "B365CA"}


def load_odds_second() -> pd.DataFrame:
    """Bet365 1X2 odds for the second divisions (closing preferred)."""
    rows = []
    for p in glob.glob(str(ROOT / "data/raw/football_data/**/*.csv"), recursive=True):
        if not re.search(rf"\d{{4}}_({SECOND_CODES})\.csv$", Path(p).name):
            continue
        try:
            df = pd.read_csv(p, encoding="latin-1", on_bad_lines="skip",
                             usecols=lambda c: c in WANT)
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
        rows.append(df)
    r = pd.concat(rows, ignore_index=True).dropna(subset=["date"])
    r = r.drop_duplicates(subset=["date", "HomeTeam", "AwayTeam"])
    r["home_team_raw"] = r["HomeTeam"].astype(str).str.strip()
    r["away_team_raw"] = r["AwayTeam"].astype(str).str.strip()
    for c in ["B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA"]:
        r[c] = pd.to_numeric(r.get(c), errors="coerce")
    r["oh"] = r["B365CH"].fillna(r["B365H"])
    r["od"] = r["B365CD"].fillna(r["B365D"])
    r["oa"] = r["B365CA"].fillna(r["B365A"])
    r = r[["date", "home_team_raw", "away_team_raw", "oh", "od", "oa"]].dropna(
        subset=["oh", "od", "oa"])
    # Decimal odds are > 1 by definition; football-data encodes a few missing
    # values as 0.0. Left in, a single such row turns 1/odds into inf and NaNs
    # the whole aggregate — it silently voided the entire Championship number.
    return r[(r[["oh", "od", "oa"]] > 1.0).all(axis=1)]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-only", action="store_true",
                    help="reuse cached walk-forward predictions (skip the ~20min refit)")
    args = ap.parse_args()
    if args.benchmark_only and OUT.exists():
        benchmark(pd.read_csv(OUT))
        return

    df = pd.read_csv(FOUND2, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_goals", "away_goals"]).sort_values("date")
    seasons = sorted(df["season"].unique())
    print(f"second divisions: {len(df):,} matches, {len(seasons)} seasons "
          f"({seasons[0]} -> {seasons[-1]})")

    rows = []
    for s in seasons:
        test = df[df["season"] == s]
        train = df[df["date"] < test["date"].min()]
        if len(test) == 0 or len(train) < 2000:
            continue
        t0 = time.time()
        # same deployed config, minus the xG-rate component (no xG down here)
        eng = PredictionEngine(blend_weight_gl=0.30, sharpen_gamma_1x2=1.3,
                               rescale_lambda_to_goals=True, outcome_rho=-0.17).fit(train)
        for r in test.itertuples(index=False):
            p = eng.predict_match(str(r.home_team), str(r.away_team),
                                  competition=str(r.competition), neutral=False)
            rows.append({"season": s, "match_id": r.match_id, "competition": r.competition,
                         "date": r.date, "home_team_raw": r.home_team_raw,
                         "away_team_raw": r.away_team_raw,
                         "hg": int(r.home_goals), "ag": int(r.away_goals),
                         "ph": p.p_home_win, "pd": p.p_draw, "pa": p.p_away_win})
        print(f"  {s}: {len(test)} matches ({time.time()-t0:.0f}s)", flush=True)

    pr = pd.DataFrame(rows)
    pr.to_csv(OUT, index=False)
    print(f"WROTE {OUT} ({len(pr):,} rows)")
    benchmark(pr)


def benchmark(pr: pd.DataFrame) -> None:
    y = np.where(pr.hg > pr.ag, 0, np.where(pr.hg == pr.ag, 1, 2))
    P = pr[["ph", "pd", "pa"]].to_numpy()
    P = P / P.sum(axis=1, keepdims=True)
    print(f"\n=== NUESTRO RENDIMIENTO EN SEGUNDAS ===")
    print(f"  RPS global {rps3(y, P):.4f}   (Big5 desplegado: 0.2007)")

    # ── benchmark vs Bet365 ──────────────────────────────────────────────────
    odds = load_odds_second()
    pr["date"] = pd.to_datetime(pr["date"]).dt.normalize()
    odds["date"] = odds["date"].dt.normalize()
    m = pr.merge(odds, on=["date", "home_team_raw", "away_team_raw"], how="inner")
    print(f"\n=== BENCHMARK vs BET365 ({len(m):,} partidos con cuotas) ===")
    if m.empty:
        print("  sin emparejamiento de cuotas")
        return

    ym = np.where(m.hg > m.ag, 0, np.where(m.hg == m.ag, 1, 2))
    Pm = m[["ph", "pd", "pa"]].to_numpy()
    Pm = Pm / Pm.sum(axis=1, keepdims=True)
    inv = np.c_[1 / m.oh, 1 / m.od, 1 / m.oa]
    M = inv / inv.sum(axis=1, keepdims=True)
    r_us, r_mkt = rps3(ym, Pm), rps3(ym, M)
    gap = r_us - r_mkt
    print(f"  nosotros  RPS {r_us:.4f}")
    print(f"  Bet365    RPS {r_mkt:.4f}")
    print(f"  BRECHA    {gap:+.4f}  ({gap / r_mkt * 100:+.1f}%)")
    print(f"  -- Big5 de referencia: brecha +0.0062 (+3.2%) --")
    verdict = ("MEJOR que en Big5: aqui el mercado es mas blando"
               if gap / r_mkt < 0.032 else
               "IGUAL O PEOR que en Big5: el mercado no es mas blando aqui")
    print(f"  -> {verdict}")

    print("\n  por liga:")
    for comp, g in m.groupby("competition"):
        if len(g) < 200:
            continue
        idx = g.index.to_numpy()
        gi = np.isin(m.index.to_numpy(), idx)
        a, b = rps3(ym[gi], Pm[gi]), rps3(ym[gi], M[gi])
        print(f"    {comp:16s} n={gi.sum():5d}  nosotros {a:.4f} | Bet365 {b:.4f} "
              f"| brecha {a-b:+.4f} ({(a-b)/b*100:+.1f}%)")

    print("\n  por temporada:")
    for s, g in m.groupby("season"):
        idx = g.index.to_numpy()
        gi = np.isin(m.index.to_numpy(), idx)
        a, b = rps3(ym[gi], Pm[gi]), rps3(ym[gi], M[gi])
        print(f"    {s:12s} n={gi.sum():5d}  {a:.4f} | {b:.4f} | {a-b:+.4f}")


if __name__ == "__main__":
    main()
