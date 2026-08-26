from __future__ import annotations

"""WHERE do we lose to Bet365 on 1X2? Diagnosis before treatment.

EVALUATION ONLY — odds are never model inputs (user rule). This script only
measures, it changes nothing and deploys nothing.

The 4.1% gap (RPS 0.2025 vs 0.1946) can come from two very different places and
the fix is completely different for each:

  CALIBRATION  our probabilities are shaped right but mis-scaled (e.g. we are
               systematically over/under-confident, or the draw is off). Cheap
               to fix: recalibrate. Upper bound measured here by CV-fitting a
               vector scaling on our own probabilities.
  INFORMATION  the market simply knows things we do not (lineups, injuries,
               rotation, motivation). No amount of recalibration recovers it;
               only new signal does.

Also reports the optimal model/market blend weight. That is NOT deployable
(it would make odds an input) but it is the cleanest measure of whether we hold
any information the market lacks: if the best blend puts weight on us and beats
the market alone, our signal is genuinely orthogonal.

Run: .venv/Scripts/python.exe scripts/diagnose_market_gap.py
"""

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
PREDS = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_deployed.csv"
WANT = {"Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A",
        "B365CH", "B365CD", "B365CA"}


def load_odds() -> pd.DataFrame:
    rows = []
    for p in glob.glob(str(ROOT / "data/raw/football_data/**/*.csv"), recursive=True):
        m = re.search(r"(\d{2})(\d{2})_(E0|SP1|D1|I1|F1)\.csv$", p)
        if not m or int(m.group(1)) < 19:
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
    r["home_team"] = r["HomeTeam"].astype(str).str.lower().str.strip()
    r["away_team"] = r["AwayTeam"].astype(str).str.lower().str.strip()
    for c in ["B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA"]:
        r[c] = pd.to_numeric(r.get(c), errors="coerce")
    r["oh"] = r["B365CH"].fillna(r["B365H"])
    r["od"] = r["B365CD"].fillna(r["B365D"])
    r["oa"] = r["B365CA"].fillna(r["B365A"])
    return r[["date", "home_team", "away_team", "oh", "od", "oa"]].dropna(
        subset=["oh", "od", "oa"])


def rps3(y: np.ndarray, P: np.ndarray) -> float:
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return float(((cp - cy) ** 2)[:, :2].sum(axis=1).mean() / 2)


def rps_each(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return ((cp - cy) ** 2)[:, :2].sum(axis=1) / 2


def logloss(y: np.ndarray, P: np.ndarray) -> float:
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())


def oracle_recalibrated(P: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Best achievable by pure recalibration: CV-fit vector scaling on log-probs.

    Out-of-fold so it is an honest ceiling, not an overfit one.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    X = np.log(np.clip(P, 1e-9, 1))
    out = np.zeros_like(P)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        lr = LogisticRegression(max_iter=2000, C=1.0)  # multinomial by default
        lr.fit(X[tr], y[tr])
        out[te] = lr.predict_proba(X[te])
    return out


def main() -> None:
    preds = pd.read_csv(PREDS)
    found = pd.read_csv(FOUND, low_memory=False)
    found["date"] = pd.to_datetime(found["date"], errors="coerce")
    m = preds.merge(found[["match_id", "date", "competition", "home_team", "away_team"]],
                    on="match_id", how="left").dropna(subset=["date"])
    odds = load_odds()
    m = m.merge(odds, on=["date", "home_team", "away_team"], how="inner")
    print(f"matches with model preds AND Bet365 odds: {len(m):,}"
          f"  ({m.date.min():%Y-%m-%d} -> {m.date.max():%Y-%m-%d})")

    # de-vig the market proportionally
    inv = np.c_[1 / m.oh, 1 / m.od, 1 / m.oa]
    M = inv / inv.sum(axis=1, keepdims=True)
    P = m[["ph", "pd", "pa"]].to_numpy()
    P = P / P.sum(axis=1, keepdims=True)
    y = np.where(m.hg > m.ag, 0, np.where(m.hg == m.ag, 1, 2))

    r_model, r_mkt = rps3(y, P), rps3(y, M)
    print(f"\n=== PUNTO DE PARTIDA ===")
    print(f"  modelo   RPS {r_model:.4f}   logloss {logloss(y, P):.4f}")
    print(f"  Bet365   RPS {r_mkt:.4f}   logloss {logloss(y, M):.4f}")
    print(f"  brecha   {r_model - r_mkt:+.4f}  ({(r_model - r_mkt) / r_mkt * 100:+.1f}%)")

    # ── 1. calibration ceiling ────────────────────────────────────────────────
    Pr = oracle_recalibrated(P, y)
    r_recal = rps3(y, Pr)
    closed = (r_model - r_recal) / (r_model - r_mkt) * 100
    print(f"\n=== 1. ¿ES CALIBRACION? (techo de recalibrar, OOF) ===")
    print(f"  modelo recalibrado  RPS {r_recal:.4f}  ({r_recal - r_model:+.4f})")
    print(f"  cierra {closed:.0f}% de la brecha")
    print("  -> " + ("CALIBRACION: recalibrar es rentable" if closed > 25 else
                     "NO es calibracion: nuestras probabilidades ya estan bien escaladas;"
                     " la brecha es INFORMACION que el mercado tiene y nosotros no"))

    # ── 2. do we hold orthogonal information? ────────────────────────────────
    print(f"\n=== 2. ¿TENEMOS INFO QUE EL MERCADO NO TIENE? (mezcla, NO desplegable) ===")
    best_w, best_r = 0.0, r_mkt
    for w in np.arange(0, 1.01, 0.05):
        B = w * P + (1 - w) * M
        r = rps3(y, B)
        if r < best_r:
            best_w, best_r = w, r
    print(f"  mejor mezcla: {best_w:.0%} modelo + {1-best_w:.0%} mercado -> RPS {best_r:.4f}")
    if best_w > 0.05:
        print(f"  -> SI: aporta {best_r - r_mkt:+.4f} sobre el mercado solo."
              f" Tenemos señal real e independiente.")
    else:
        print("  -> NO: el mercado ya contiene todo lo nuestro.")

    # ── 3. where does the loss concentrate? ──────────────────────────────────
    m["rps_model"] = rps_each(y, P)
    m["rps_mkt"] = rps_each(y, M)
    m["loss"] = m.rps_model - m.rps_mkt
    m["y"] = y
    m["fav"] = M.max(axis=1)

    print(f"\n=== 3. ¿DONDE PERDEMOS? (perdida media vs mercado) ===")
    print("  por resultado real:")
    for k, lbl in [(0, "local gana"), (1, "empate"), (2, "visitante gana")]:
        s = m[m.y == k]
        print(f"     {lbl:16s} n={len(s):5d}  {s.loss.mean():+.4f}")

    print("  por claridad del partido (prob. del favorito segun mercado):")
    for lo, hi, lbl in [(0, .40, "muy igualado <40%"), (.40, .50, "igualado 40-50%"),
                        (.50, .65, "favorito 50-65%"), (.65, 1.01, "claro >65%")]:
        s = m[(m.fav >= lo) & (m.fav < hi)]
        if len(s) < 50:
            continue
        print(f"     {lbl:18s} n={len(s):5d}  {s.loss.mean():+.4f}")

    print("  por liga:")
    for comp, s in m.groupby("competition"):
        if len(s) < 100:
            continue
        print(f"     {comp:16s} n={len(s):5d}  {s.loss.mean():+.4f}"
              f"   (modelo {s.rps_model.mean():.4f} vs mkt {s.rps_mkt.mean():.4f})")

    # ── 4. are we systematically over/under-confident? ───────────────────────
    print(f"\n=== 4. SESGO DE CONFIANZA (media de nuestras probs vs mercado vs real) ===")
    for i, lbl in [(0, "local"), (1, "empate"), (2, "visitante")]:
        print(f"  {lbl:10s} modelo {P[:, i].mean():.4f} | mercado {M[:, i].mean():.4f} "
              f"| real {(y == i).mean():.4f}")


if __name__ == "__main__":
    main()
