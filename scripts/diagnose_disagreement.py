from __future__ import annotations

"""The decisive test: when we disagree with Bet365, who is right?

diagnose_market_gap.py showed the optimal model/market blend puts 0% weight on
us, and diagnose_home_bias.py ruled out home advantage (the bias was COVID and
recent seasons are near-unbiased, yet the gap persists). What is left is a
direct question: are our disagreements with the market SIGNAL or NOISE?

If the market wins on the matches where we disagree most, then the gap is
genuine information asymmetry (lineups, injuries, rotation) and no further
modelling on the same data will close it. That is a decision-grade result: it
says stop optimising 1X2 and go where the market is softer instead.

EVALUATION ONLY. Odds are a yardstick, never a model input.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_market_gap import load_odds, rps3, rps_each  # noqa: E402

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
PREDS = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_deployed.csv"


def main() -> None:
    preds = pd.read_csv(PREDS)
    found = pd.read_csv(FOUND, low_memory=False)
    found["date"] = pd.to_datetime(found["date"], errors="coerce")
    m = preds.merge(found[["match_id", "date", "competition", "home_team", "away_team"]],
                    on="match_id", how="left").dropna(subset=["date"])
    m = m.merge(load_odds(), on=["date", "home_team", "away_team"], how="inner")
    m = m.reset_index(drop=True)

    inv = np.c_[1 / m.oh, 1 / m.od, 1 / m.oa]
    M = inv / inv.sum(axis=1, keepdims=True)
    P = m[["ph", "pd", "pa"]].to_numpy()
    P = P / P.sum(axis=1, keepdims=True)
    y = np.where(m.hg > m.ag, 0, np.where(m.hg == m.ag, 1, 2))

    # exclude the COVID season: it is a known distortion, not the steady state
    post = (m.season != "2020-2021").to_numpy()
    print(f"n total={len(m):,} | sin 2020-21 (COVID)={post.sum():,}")

    for label, mask in [("TODO", np.ones(len(m), bool)), ("SIN COVID", post)]:
        print(f"\n===== {label} =====")
        Pm, Mm, ym = P[mask], M[mask], y[mask]
        r_mod, r_mkt = rps3(ym, Pm), rps3(ym, Mm)
        print(f"  modelo {r_mod:.4f} | mercado {r_mkt:.4f} | brecha {r_mod-r_mkt:+.4f}")

        # disagreement = total variation distance between the two distributions
        dis = np.abs(Pm - Mm).sum(axis=1) / 2
        e_mod, e_mkt = rps_each(ym, Pm), rps_each(ym, Mm)
        print(f"  {'discrepancia':>16s} {'n':>5s} {'modelo':>8s} {'mercado':>8s} {'quien gana':>12s}")
        qs = np.quantile(dis, [0, .25, .50, .75, .90, 1.0])
        names = ["0-25% (acuerdo)", "25-50%", "50-75%", "75-90%", "90-100% (discrepa)"]
        for i, nm in enumerate(names):
            sel = (dis >= qs[i]) & (dis <= qs[i + 1])
            if sel.sum() < 30:
                continue
            a, b = e_mod[sel].mean(), e_mkt[sel].mean()
            who = "MERCADO" if b < a else "modelo"
            print(f"  {nm:>16s} {sel.sum():5d} {a:8.4f} {b:8.4f} {who:>12s} ({a-b:+.4f})")

        # does the model add anything on top of the market at any weight?
        best = min(((rps3(ym, w * Pm + (1 - w) * Mm), w) for w in np.arange(0, 1.001, 0.02)))
        print(f"  mejor mezcla: {best[1]:.0%} modelo -> RPS {best[0]:.4f} "
              f"({best[0]-r_mkt:+.4f} vs mercado solo)")

        # is there ANY league where we beat the market?
        print("  ¿alguna liga donde ganemos?")
        sub = m[mask]
        for comp, g in sub.groupby("competition"):
            idx = g.index.to_numpy()
            gi = np.isin(np.where(mask)[0], idx)
            if gi.sum() < 200:
                continue
            a, b = rps3(y[idx], P[idx]), rps3(y[idx], M[idx])
            mark = "  <-- GANAMOS" if a < b else ""
            print(f"     {comp:16s} modelo {a:.4f} vs mercado {b:.4f} ({a-b:+.4f}){mark}")


if __name__ == "__main__":
    main()
