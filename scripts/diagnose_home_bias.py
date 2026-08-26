from __future__ import annotations

"""Follow-up to diagnose_market_gap.py: is our home/away bias COVID or chronic?

diagnose_market_gap showed the whole Bet365 gap sits on away wins (+0.0290),
with the model at home 0.4471 / away 0.2938 against a reality of 0.4302 /
0.3170. Two very different explanations:

  COVID ARTEFACT  2020/21 was played in empty stadiums and home advantage
                  collapsed. If the bias is concentrated there, the deployed
                  model is fine and the walk-forward window is just unlucky.
  CHRONIC         we carry a stale home-advantage estimate every season. Then a
                  time-varying home advantage is worth real RPS.

Also measures the value of the single cheapest possible fix — one global
home->away probability shift, fitted out-of-fold — to size the prize before
touching any deployed code.

EVALUATION ONLY. Odds are used solely as a yardstick, never as model input.
"""

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from diagnose_market_gap import load_odds, rps3  # noqa: E402

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
PREDS = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_deployed.csv"


def _shift_home_away(P: np.ndarray, t: float) -> np.ndarray:
    """Move probability mass between home and away in log space, renormalised."""
    L = np.log(np.clip(P, 1e-9, 1))
    L[:, 0] -= t
    L[:, 2] += t
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


def main() -> None:
    preds = pd.read_csv(PREDS)
    found = pd.read_csv(FOUND, low_memory=False)
    found["date"] = pd.to_datetime(found["date"], errors="coerce")
    m = preds.merge(found[["match_id", "date", "competition", "home_team", "away_team"]],
                    on="match_id", how="left").dropna(subset=["date"])
    m = m.merge(load_odds(), on=["date", "home_team", "away_team"], how="inner")

    inv = np.c_[1 / m.oh, 1 / m.od, 1 / m.oa]
    M = inv / inv.sum(axis=1, keepdims=True)
    P = m[["ph", "pd", "pa"]].to_numpy()
    P = P / P.sum(axis=1, keepdims=True)
    y = np.where(m.hg > m.ag, 0, np.where(m.hg == m.ag, 1, 2))
    m = m.reset_index(drop=True)

    print(f"n={len(m):,}  {m.date.min():%Y-%m-%d} -> {m.date.max():%Y-%m-%d}")

    # ── 1. bias by season: COVID or chronic? ──────────────────────────────────
    print("\n=== 1. SESGO POR TEMPORADA (modelo - real) ===")
    print(f"  {'temporada':12s} {'n':>5s}  {'local':>18s}  {'visitante':>18s}   "
          f"{'RPS mod':>8s} {'RPS mkt':>8s}")
    for ssn, s in m.groupby("season"):
        idx = s.index.to_numpy()
        ph, pa = P[idx, 0].mean(), P[idx, 2].mean()
        mh, ma = M[idx, 0].mean(), M[idx, 2].mean()
        ah, aa = (y[idx] == 0).mean(), (y[idx] == 2).mean()
        print(f"  {ssn:12s} {len(s):5d}  {ph:.3f}/{ah:.3f} ({ph-ah:+.3f})  "
              f"{pa:.3f}/{aa:.3f} ({pa-aa:+.3f})   "
              f"{rps3(y[idx], P[idx]):.4f} {rps3(y[idx], M[idx]):.4f}")
    print("  (formato: modelo/real (sesgo).  el mercado, de referencia:")
    print(f"   media mercado local {M[:,0].mean():.3f} vs real {(y==0).mean():.3f})")

    # ── 2. how much is a single global home->away shift worth? ────────────────
    print("\n=== 2. PREMIO DE UN SOLO AJUSTE GLOBAL local->visitante (OOF) ===")
    from sklearn.model_selection import KFold
    base = rps3(y, P)
    oof = np.zeros_like(P)
    ts = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(P):
        grid = np.arange(-0.30, 0.31, 0.01)
        best_t = min(grid, key=lambda t: rps3(y[tr], _shift_home_away(P[tr], t)))
        ts.append(best_t)
        oof[te] = _shift_home_away(P[te], best_t)
    r_shift = rps3(y, oof)
    print(f"  t medio={np.mean(ts):+.3f}  RPS {base:.4f} -> {r_shift:.4f} ({r_shift-base:+.4f})")
    gap = base - rps3(y, M)
    print(f"  cierra {(base - r_shift)/gap*100:.0f}% de la brecha con Bet365")

    # ── 3. is the bias stable enough to correct per-season? ───────────────────
    print("\n=== 3. AJUSTE OPTIMO POR TEMPORADA (¿es un valor estable o deriva?) ===")
    grid = np.arange(-0.40, 0.41, 0.01)
    for ssn, s in m.groupby("season"):
        idx = s.index.to_numpy()
        t = min(grid, key=lambda t: rps3(y[idx], _shift_home_away(P[idx], t)))
        r0, r1 = rps3(y[idx], P[idx]), rps3(y[idx], _shift_home_away(P[idx], t))
        print(f"  {ssn:12s} t*={t:+.2f}   RPS {r0:.4f} -> {r1:.4f} ({r1-r0:+.4f})")

    # ── 4. same, per league ───────────────────────────────────────────────────
    print("\n=== 4. AJUSTE OPTIMO POR LIGA ===")
    for comp, s in m.groupby("competition"):
        idx = s.index.to_numpy()
        if len(idx) < 300:
            continue
        t = min(grid, key=lambda t: rps3(y[idx], _shift_home_away(P[idx], t)))
        r0, r1 = rps3(y[idx], P[idx]), rps3(y[idx], _shift_home_away(P[idx], t))
        print(f"  {comp:16s} t*={t:+.2f}   RPS {r0:.4f} -> {r1:.4f} ({r1-r0:+.4f})")


if __name__ == "__main__":
    main()
