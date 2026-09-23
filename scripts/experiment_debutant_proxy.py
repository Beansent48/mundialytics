from __future__ import annotations

"""Can a club new to the Big Five be priced honestly before it has any history?

The logger skipped debutants' fixtures (unknown team) and the engine, if asked,
silently prices an unknown name as team index 0 -- another club. Candidate:
stand in last season's relegated clubs from the same league (the level a
promoted side replaces), average their lambdas against the real opponent.

Walk-forward: for each season an engine is fitted only on earlier seasons, and
each debutant's first three matches are scored three ways -- the stand-in, the
engine as-is (index 0), and the training data's 1X2 base rates.

    .venv/Scripts/python.exe scripts/experiment_debutant_proxy.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.serving.debutants import predict_match_or_proxy  # noqa: E402
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS, PredictionEngine)

SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
FIRST_N = 3


def rps(p, outcome: int) -> float:
    y = np.zeros(3)
    y[outcome] = 1
    cp, cy = np.cumsum(p), np.cumsum(y)
    return float(((cp - cy) ** 2)[:2].sum() / 2)


def main() -> None:
    df = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv",
                     low_memory=False)
    df = df[df["xg_available"] == True].copy()  # noqa: E712
    for c in ["home_goals", "away_goals", "home_xg", "away_xg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals", "home_xg", "away_xg", "date"]).sort_values("date")
    long = pd.concat([df[["season", "home_team"]].rename(columns={"home_team": "t"}),
                      df[["season", "away_team"]].rename(columns={"away_team": "t"})])
    first_season = long.groupby("t")["season"].min()

    recs = []
    for s in SEASONS:
        test = df[df["season"] == s]
        debutants = set(first_season[first_season == s].index)
        if not debutants:
            continue
        train = df[df["date"] < test["date"].min()]
        t0 = time.time()
        eng = PredictionEngine(**DEPLOYED_CLUB_ENGINE_KWARGS).fit(train)
        known = set(train["home_team"]) | set(train["away_team"])
        y = np.where(train.home_goals > train.away_goals, 0,
                     np.where(train.home_goals == train.away_goals, 1, 2))
        base = np.bincount(y, minlength=3) / len(y)
        for team in sorted(debutants):
            games = test[(test.home_team == team) | (test.away_team == team)].head(FIRST_N)
            for r in games.itertuples(index=False):
                comp_teams = test[test.competition == r.competition]
                current = set(comp_teams.home_team) | set(comp_teams.away_team)
                out = 0 if r.home_goals > r.away_goals else (1 if r.home_goals == r.away_goals else 2)
                try:
                    p, proxies = predict_match_or_proxy(
                        eng, train, r.home_team, r.away_team, r.competition, s, current,
                        known=known)
                    proxy = [p.p_home_win, p.p_draw, p.p_away_win]
                except LookupError:
                    continue
                try:
                    q = eng.predict_match(r.home_team, r.away_team, competition=r.competition)
                    idx0 = [q.p_home_win, q.p_draw, q.p_away_win]
                except Exception:
                    idx0 = None
                recs.append({"season": s, "team": team, "match": f"{r.home_team} vs {r.away_team}",
                             "proxy": rps(proxy, out),
                             "index0": rps(idx0, out) if idx0 else np.nan,
                             "base": rps(base, out),
                             "stand_in": ", ".join(next(iter(proxies.values()), []))})
        print(f"  {s}: {sorted(debutants)} ({time.time() - t0:.0f}s)", flush=True)

    d = pd.DataFrame(recs)
    print(f"\n{len(d)} partidos de {d.team.nunique()} debutantes (primeros {FIRST_N} de cada uno)")
    print(f"  RPS sustituto (descendidos) {d.proxy.mean():.4f}")
    print(f"  RPS motor tal cual (indice 0) {d.index0.mean():.4f}  (n={d.index0.notna().sum()})")
    print(f"  RPS frecuencias base         {d.base.mean():.4f}")
    wins = (d.proxy < d.base).mean()
    print(f"  sustituto mejor que la base en {wins:.0%} de los partidos")
    print("\n" + d.groupby("team")[["proxy", "index0", "base"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
