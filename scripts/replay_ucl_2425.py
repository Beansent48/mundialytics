from __future__ import annotations

"""REPLAY test: simulate the REAL 2024/25 Champions League from matchday 0
(actual draw, no results) and score the simulator against what actually
happened — the end-to-end tournament-layer validation.

Metrics:
  - Brier of P(top-24) and P(top-8) vs the real league-phase table
  - rank correlation of predicted vs real league-phase points
  - the real champion's (PSG) pre-tournament probabilities
Elo used: pre-tournament (2024-09-01) — walk-forward honest.
"""

import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.statistical_core.competition.european import (  # noqa: E402
    EuropeanTournament, fetch_current_elo, load_calibration)

sys.path.insert(0, str(ROOT / "scripts"))
from validate_european_mapping import ALIASES, fetch_history, norm  # noqa: E402


def main() -> None:
    raw = pd.read_csv(ROOT / "data/external/uefa/raw_champions-league_2024.csv")
    raw["date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce", format="mixed").dt.normalize()
    res = raw["Result"].astype(str).str.extract(r"(\d+)\s*-\s*(\d+)")
    raw["hg"] = pd.to_numeric(res[0], errors="coerce")
    raw["ag"] = pd.to_numeric(res[1], errors="coerce")
    league = raw[pd.to_numeric(raw["Round Number"], errors="coerce").between(1, 8)].copy()
    print(f"league phase matches: {len(league)}")

    elo_now = fetch_current_elo(ROOT)
    by_norm = {norm(n): n for n in elo_now}

    def to_clubelo(name: str) -> str | None:
        low = str(name).lower().strip()
        if low in ALIASES:
            return ALIASES[low]
        n = norm(name)
        if n in by_norm:
            return by_norm[n]
        cands = [v for k, v in by_norm.items() if n in k or k in n]
        return cands[0] if len(cands) == 1 else None

    league["home"] = league["Home Team"].map(to_clubelo)
    league["away"] = league["Away Team"].map(to_clubelo)
    unresolved = sorted(set(league.loc[league.home.isna(), "Home Team"]) |
                        set(league.loc[league.away.isna(), "Away Team"]))
    print(f"unresolved: {unresolved}")
    league = league.dropna(subset=["home", "away"])
    teams = sorted(set(league.home) | set(league.away))
    print(f"teams resolved: {len(teams)}")

    # pre-tournament Elo (2024-09-01), from cached histories
    when = pd.Timestamp("2024-09-01")
    elo0 = {}
    for t in teams:
        h = fetch_history(t)
        if h is None:
            continue
        r = h[(h.From <= when) & (h.To >= when)]
        if len(r):
            elo0[t] = float(r.Elo.iloc[0])
    print(f"pre-tournament Elo for {len(elo0)}/{len(teams)} teams")

    fixtures = league[["home", "away"]].copy()
    fixtures["home_goals"] = np.nan     # matchday 0: nothing played
    fixtures["away_goals"] = np.nan
    calib = load_calibration(ROOT)
    tour = EuropeanTournament("champions", elo0, calib, fixtures)
    pred = tour.simulate(4000).set_index("team")

    # real league-phase table
    pts = {t: 0 for t in teams}
    for r in league.itertuples(index=False):
        if pd.isna(r.hg):
            continue
        pts[r.home] += 3 if r.hg > r.ag else (1 if r.hg == r.ag else 0)
        pts[r.away] += 3 if r.ag > r.hg else (1 if r.hg == r.ag else 0)
    table = pd.Series(pts).sort_values(ascending=False)
    real_top24 = set(table.index[:24])
    real_top8 = set(table.index[:8])

    common = [t for t in pred.index if t in table.index]
    y24 = np.array([1.0 if t in real_top24 else 0.0 for t in common])
    p24 = pred.loc[common, "p_top24"].to_numpy()
    y8 = np.array([1.0 if t in real_top8 else 0.0 for t in common])
    p8 = pred.loc[common, "p_top8"].to_numpy()
    brier24 = float(((p24 - y24) ** 2).mean())
    brier8 = float(((p8 - y8) ** 2).mean())
    base24 = float(((24 / 36 - y24) ** 2).mean())
    base8 = float(((8 / 36 - y8) ** 2).mean())
    from scipy.stats import spearmanr
    rho = spearmanr(pred.loc[common, "p_top8"], table.loc[common]).statistic
    print(f"\nBrier top-24: modelo {brier24:.4f} vs base(24/36) {base24:.4f}")
    print(f"Brier top-8 : modelo {brier8:.4f} vs base(8/36) {base8:.4f}")
    print(f"Spearman(pred top8, puntos reales): {rho:.3f}")

    champ = "ParisSG" if "ParisSG" in pred.index else "Paris SG"
    if champ in pred.index:
        r = pred.loc[champ]
        rank_c = int((pred["p_champion"] > r["p_champion"]).sum()) + 1
        print(f"\nCampeón real 24/25 (PSG): p_campeón pre-torneo {r['p_champion']:.1%} "
              f"(ranking #{rank_c}), p_top8 {r['p_top8']:.1%}")
    print("\ntop-8 predicho vs real:")
    pred8 = list(pred.nlargest(8, "p_top8").index)
    print(f"  predicho: {pred8}")
    print(f"  real:     {sorted(real_top8, key=lambda t: -table[t])}")
    print(f"  overlap:  {len(set(pred8) & real_top8)}/8")


if __name__ == "__main__":
    main()
