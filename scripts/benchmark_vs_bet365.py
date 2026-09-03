from __future__ import annotations

"""Benchmark the deployed walk-forward model against Bet365 odds on the SAME
matches (2020/21-2025/26 folds). EVALUATION ONLY — odds are never model inputs
(user rule: that would be cheating); this measures our true distance to the
market's probabilities.

Odds: closing (B365C*) preferred, pre-match (B365*) fallback; proportional
de-vig. Markets: 1X2 (RPS + log-loss) and Over/Under 2.5 (log-loss).
"""

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
PREDS = ROOT / "data/processed/enriched/understat_xg/walkforward_preds.csv"
WANT = {"Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A",
        "B365CH", "B365CD", "B365CA", "B365>2.5", "B365<2.5", "B365C>2.5", "B365C<2.5"}


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
    for c in ["B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA",
              "B365>2.5", "B365<2.5", "B365C>2.5", "B365C<2.5"]:
        if c in r.columns:
            r[c] = pd.to_numeric(r[c], errors="coerce")
        else:
            r[c] = np.nan
    # closing preferred, pre-match fallback
    r["oh"] = r["B365CH"].fillna(r["B365H"])
    r["od"] = r["B365CD"].fillna(r["B365D"])
    r["oa"] = r["B365CA"].fillna(r["B365A"])
    r["oo25"] = r["B365C>2.5"].fillna(r["B365>2.5"])
    r["ou25"] = r["B365C<2.5"].fillna(r["B365<2.5"])
    r["closing_used"] = r["B365CH"].notna()
    return r[["date", "home_team", "away_team", "oh", "od", "oa", "oo25", "ou25", "closing_used"]]


def rps3(y_idx: np.ndarray, P: np.ndarray) -> float:
    """mean RPS for 3 ordered outcomes; P rows = [p_home, p_draw, p_away]."""
    Y = np.zeros_like(P)
    Y[np.arange(len(y_idx)), y_idx] = 1.0
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return float(((cp - cy) ** 2)[:, :2].sum(axis=1).mean() / 2)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    odds = load_odds()
    found = pd.read_csv(FOUND, low_memory=False)[["match_id", "date", "home_team", "away_team", "competition"]]
    found["date"] = pd.to_datetime(found["date"], errors="coerce")
    preds = pd.read_csv(PREDS)
    df = (preds.merge(found, on="match_id", how="left")
               .merge(odds, on=["date", "home_team", "away_team"], how="left")
               .drop_duplicates(subset=["match_id"]))
    ok = df.dropna(subset=["oh", "od", "oa"]).copy()
    print(f"our preds: {len(preds)} | with B365 1X2 odds: {len(ok)} "
          f"({ok.closing_used.mean():.0%} closing)")

    # de-vig 1X2
    inv = 1 / ok[["oh", "od", "oa"]].to_numpy()
    book = inv / inv.sum(axis=1, keepdims=True)
    ours = ok[["ph", "pd", "pa"]].to_numpy(float)
    y_idx = np.where(ok.hg > ok.ag, 0, np.where(ok.hg == ok.ag, 1, 2))

    print("\n===== 1X2 =====")
    print(f"  RPS   OURS {rps3(y_idx, ours):.4f} | BET365 {rps3(y_idx, book):.4f} | "
          f"gap {rps3(y_idx, ours) - rps3(y_idx, book):+.4f}")
    ll_o = float(-np.log(np.clip(ours[np.arange(len(y_idx)), y_idx], 1e-9, 1)).mean())
    ll_b = float(-np.log(np.clip(book[np.arange(len(y_idx)), y_idx], 1e-9, 1)).mean())
    print(f"  LL    OURS {ll_o:.4f} | BET365 {ll_b:.4f} | gap {ll_o - ll_b:+.4f}")
    print(f"  mean |p_ours - p_book| (home prob): {np.abs(ours[:, 0] - book[:, 0]).mean():.4f}")

    # Uninformed anchors, so the gap to the book has a scale to read it against:
    # "4% behind Bet365" means nothing until you know what knowing nothing costs.
    unif = np.full_like(ours, 1 / 3)
    rates = np.bincount(y_idx, minlength=3) / len(y_idx)
    base = np.tile(rates, (len(y_idx), 1))
    r_unif, r_base = rps3(y_idx, unif), rps3(y_idx, base)
    r_ours, r_book = rps3(y_idx, ours), rps3(y_idx, book)
    span = r_base - r_book
    print(f"  baselines: uniform 1/3 {r_unif:.4f} | league base rates {r_base:.4f} "
          f"({rates[0]:.1%}/{rates[1]:.1%}/{rates[2]:.1%})")
    print(f"  base rates -> closing line spans {span:.4f} RPS; "
          f"we cover {(r_base - r_ours) / span:.1%} of it")
    print("\n  per season (RPS ours | book | gap):")
    for s, g in ok.groupby("season"):
        gi = np.where(g.hg > g.ag, 0, np.where(g.hg == g.ag, 1, 2))
        go = g[["ph", "pd", "pa"]].to_numpy(float)
        gb_inv = 1 / g[["oh", "od", "oa"]].to_numpy()
        gb = gb_inv / gb_inv.sum(axis=1, keepdims=True)
        print(f"    {s}: {rps3(gi, go):.4f} | {rps3(gi, gb):.4f} | {rps3(gi, go)-rps3(gi, gb):+.4f}  (n={len(g)})")
    print("\n  per league (RPS gap):")
    for c, g in ok.groupby("competition"):
        gi = np.where(g.hg > g.ag, 0, np.where(g.hg == g.ag, 1, 2))
        go = g[["ph", "pd", "pa"]].to_numpy(float)
        gb_inv = 1 / g[["oh", "od", "oa"]].to_numpy()
        gb = gb_inv / gb_inv.sum(axis=1, keepdims=True)
        print(f"    {c}: {rps3(gi, go)-rps3(gi, gb):+.4f}  (n={len(g)})")

    ou = ok.dropna(subset=["oo25", "ou25"]).copy()
    ou = ou[(ou.oo25 > 1.01) & (ou.ou25 > 1.01)]
    if len(ou):
        inv2 = 1 / ou[["oo25", "ou25"]].to_numpy()
        book_o = (inv2 / inv2.sum(axis=1, keepdims=True))[:, 0]
        y_o = ((ou.hg + ou.ag) > 2.5).astype(float).to_numpy()
        print(f"\n===== OVER/UNDER 2.5 (n={len(ou)}) =====")
        print(f"  LL    OURS {bll(y_o, ou.po25.to_numpy(float)):.4f} | BET365 {bll(y_o, book_o):.4f} | "
              f"gap {bll(y_o, ou.po25.to_numpy(float)) - bll(y_o, book_o):+.4f}")
        print(f"  mean |p_ours - p_book|: {np.abs(ou.po25.to_numpy(float) - book_o).mean():.4f}")


if __name__ == "__main__":
    main()
