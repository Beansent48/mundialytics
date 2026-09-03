from __future__ import annotations

"""Fit the local Elo update rule, and prove that rolling forward beats freezing.

Gates the whole local-ClubElo idea (see src/mundialytics/ratings/clubelo_local.py).
Two questions, in order:

  1. WHICH CONSTANTS? ClubElo's own are not reproducible here — there is no
     ClubElo history in the repo to fit against (the *_clubelo feature files
     record snapshot dates but are 0% populated). So K and home advantage are
     fitted on our own matches by out-of-sample predictive accuracy instead of
     asserted from memory.

  2. DOES UPDATING EVEN HELP? A rating rolled forward with a wrong rule is worse
     than one left alone. So: freeze ratings at a cutoff, roll the same ratings
     through the following window using results, and compare how each predicts
     that window. Rolling must win, or this does not ship.

Run: .venv/Scripts/python.exe scripts/fit_local_elo.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.ratings.clubelo_local import (  # noqa: E402
    EloParams, expected_home, goal_diff_multiplier)

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
OUT = ROOT / "data/processed/local_elo_params.json"


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(FOUND, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    return df.sort_values("date").reset_index(drop=True)


def run_elo(df: pd.DataFrame, k: float, hfa: float, start: float = 1500.0,
            score_from: pd.Timestamp | None = None,
            freeze_at: pd.Timestamp | None = None) -> tuple[float, int]:
    """Walk the matches, scoring 1X2 log-loss on those at/after `score_from`.

    `freeze_at` stops the ratings updating from that date on — the control arm.
    Draw probability is a fixed share of the remaining mass; it is identical for
    both arms, so it cannot favour either.
    """
    elo: dict[str, float] = {}
    ll, n = 0.0, 0
    for r in df.itertuples(index=False):
        h, a = r.home_team, r.away_team
        eh = elo.setdefault(h, start)
        ea = elo.setdefault(a, start)
        p_home_raw = expected_home(eh, ea, hfa)

        if score_from is not None and r.date >= score_from:
            # split the two-way expectation into 1X2 with a flat draw rate
            p_draw = 0.26
            ph = p_home_raw * (1 - p_draw)
            pa = (1 - p_home_raw) * (1 - p_draw)
            hg, ag = float(r.home_goals), float(r.away_goals)
            p = ph if hg > ag else (pa if hg < ag else p_draw)
            ll -= np.log(max(p, 1e-12))
            n += 1

        if freeze_at is not None and r.date >= freeze_at:
            continue
        hg, ag = float(r.home_goals), float(r.away_goals)
        s = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        delta = k * goal_diff_multiplier(hg - ag) * (s - p_home_raw)
        elo[h] += delta
        elo[a] -= delta
    return (ll / max(n, 1), n)


def main() -> None:
    df = load_matches()
    print(f"partidos: {len(df):,}  ({df.date.min():%Y-%m-%d} -> {df.date.max():%Y-%m-%d})")

    # fit on everything up to the holdout, score on the last two seasons
    holdout = pd.Timestamp("2024-08-01")
    print(f"\n=== 1. AJUSTE de K y factor campo (holdout desde {holdout:%Y-%m-%d}) ===")
    best = None
    for k in (12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 40.0):
        for hfa in (40.0, 55.0, 65.0, 80.0):
            ll, n = run_elo(df, k, hfa, score_from=holdout)
            if best is None or ll < best[0]:
                best = (ll, k, hfa, n)
    ll, k, hfa, n = best
    print(f"  mejor: K={k:.0f}  factor campo={hfa:.0f}  logloss={ll:.5f}  (n={n:,})")
    for kk in (12.0, 20.0, 32.0):
        print(f"    K={kk:<5.0f} hfa={hfa:.0f} -> {run_elo(df, kk, hfa, score_from=holdout)[0]:.5f}")

    # ── 2. does rolling forward beat freezing? ───────────────────────────────
    print("\n=== 2. ¿AVANZAR bate a CONGELAR? ===")
    print(f"  {'corte':12s} {'n':>6s} {'congelado':>10s} {'avanzado':>10s} {'mejora':>9s}")
    wins = 0
    cuts = ["2021-08-01", "2022-08-01", "2023-08-01", "2024-08-01", "2025-08-01"]
    for c in cuts:
        cut = pd.Timestamp(c)
        end = cut + pd.Timedelta(days=300)
        window = df[(df.date >= cut) & (df.date < end)]
        if len(window) < 500:
            continue
        sub = df[df.date < end]
        frozen, _ = run_elo(sub, k, hfa, score_from=cut, freeze_at=cut)
        rolled, nn = run_elo(sub, k, hfa, score_from=cut)
        wins += int(rolled < frozen)
        print(f"  {c:12s} {nn:6d} {frozen:10.5f} {rolled:10.5f} {frozen-rolled:+9.5f}")
    print(f"\n  avanzar gana en {wins}/{len(cuts)} cortes")
    verdict = "OK — el avance local aporta" if wins >= len(cuts) - 1 else "NO — no desplegar"
    print(f"  -> {verdict}")

    import json
    OUT.write_text(json.dumps({"k": k, "hfa": hfa, "fitted_logloss": round(ll, 6),
                               "holdout_from": str(holdout.date()),
                               "rolling_beats_freezing": f"{wins}/{len(cuts)}"}, indent=2),
                   encoding="utf-8")
    print(f"\nescrito {OUT}")


if __name__ == "__main__":
    main()
