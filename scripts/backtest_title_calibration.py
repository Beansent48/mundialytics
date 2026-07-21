from __future__ import annotations

"""Calibration backtest of the competition layer's title (p_champion) forecasts.

Runs the resume simulator at early cutoffs (MD5, MD10) across many completed Big5
seasons, records each team's predicted p_champion and whether it actually won the
league, then measures reliability (predicted vs empirical). Purpose: quantify the
flagged early-season title OVERCONFIDENCE and give the shrinkage/calibration fix a
target. Leakage-safe: the engine trains strictly on pre-cutoff matches.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.competition.cutoff import load_league_state_from_foundation
from mundialytics.statistical_core.competition.engine_provider import train_engine_before_cutoff, fixture_lambdas
from mundialytics.statistical_core.competition.resume_simulator import simulate_rest_of_season

BIG5 = ["Bundesliga", "LaLiga", "Ligue 1", "Premier League", "Serie A"]


def final_champion(found: pd.DataFrame, comp: str, season: str) -> str | None:
    g = found[(found["competition"] == comp) & (found["season"] == season)].dropna(subset=["home_goals", "away_goals"])
    if g.empty:
        return None
    pts: dict[str, int] = {}; gf: dict[str, int] = {}; ga: dict[str, int] = {}
    for _, r in g.iterrows():
        h, a = r["home_team"], r["away_team"]; hg, ag = int(r["home_goals"]), int(r["away_goals"])
        for t in (h, a):
            pts.setdefault(t, 0); gf.setdefault(t, 0); ga.setdefault(t, 0)
        gf[h] += hg; ga[h] += ag; gf[a] += ag; ga[a] += hg
        if hg > ag: pts[h] += 3
        elif ag > hg: pts[a] += 3
        else: pts[h] += 1; pts[a] += 1
    table = sorted(pts, key=lambda t: (pts[t], gf[t] - ga[t], gf[t]), reverse=True)
    return table[0] if table else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="+", default=[f"{y}-{y+1}" for y in range(2014, 2025)])
    ap.add_argument("--matchdays", nargs="+", type=int, default=[5, 10])
    ap.add_argument("--blend", type=float, default=0.30)
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--out", default="data/processed/enriched/understat_xg/title_calibration.csv")
    args = ap.parse_args()

    found = pd.read_csv(ROOT / "data/processed/foundation_big5_multi_season.csv", low_memory=False)
    rows = []
    for season in args.seasons:
        for comp in BIG5:
            champ = final_champion(found, comp, season)
            if champ is None:
                continue
            for md in args.matchdays:
                try:
                    st = load_league_state_from_foundation(comp, season, cutoff_matchday=md, foundation=found)
                    eng = train_engine_before_cutoff(st, found, blend_weight_gl=args.blend)
                    lam = fixture_lambdas(eng, st)
                    fc = simulate_rest_of_season(lam, st, n_sims=args.sims)
                    tp = fc.team_probs if hasattr(fc, "team_probs") else fc
                    for _, tr in tp.iterrows():
                        rows.append({"comp": comp, "season": season, "md": md,
                                     "team": tr["team"], "p_champion": float(tr["p_champion"]),
                                     "is_champion": int(tr["team"] == champ)})
                    lead = tp.sort_values("p_champion", ascending=False).iloc[0]
                    print(f"{comp} {season} MD{md}: top {lead['team']} {lead['p_champion']:.0%} (champ {champ})", flush=True)
                except Exception as exc:
                    print(f"skip {comp} {season} MD{md}: {str(exc)[:80]}", flush=True)

    df = pd.DataFrame(rows)
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / args.out, index=False)
    print(f"\nWROTE {ROOT / args.out} ({len(df)} team-rows)", flush=True)

    # Reliability: bin predicted p_champion, compare to empirical champion rate.
    print("\n===== TITLE-PROB CALIBRATION (all matchdays pooled) =====", flush=True)
    for md in args.matchdays + [None]:
        d = df if md is None else df[df["md"] == md]
        label = "ALL" if md is None else f"MD{md}"
        bins = [0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.90, 1.01]
        d = d.copy(); d["bin"] = pd.cut(d["p_champion"], bins=bins, include_lowest=True)
        g = d.groupby("bin", observed=True).agg(n=("is_champion", "size"),
                                                mean_pred=("p_champion", "mean"),
                                                emp=("is_champion", "mean"))
        print(f"\n[{label}]  (predicted vs empirical; predicted>empirical = overconfident)", flush=True)
        for b, r in g.iterrows():
            gap = r["mean_pred"] - r["emp"]
            print(f"  {str(b):14s} n={int(r['n']):4d}  pred={r['mean_pred']:.3f}  emp={r['emp']:.3f}  gap={gap:+.3f}", flush=True)


if __name__ == "__main__":
    main()
