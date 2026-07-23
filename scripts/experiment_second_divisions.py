from __future__ import annotations

"""B) Second divisions in the strength fit — attack the known promoted-team
bias (+5.5pp under-prediction, measured earlier and left below the deploy bar).

Adds Championship/Segunda/2.Bundesliga/Serie B/Ligue 2 (2014+) as EXTRA
training rows for the deployed engine config (its AttackDefense has per-league
effects; team names match across divisions in football-data, so a promoted
side enters the season with real history instead of the global prior).
Evaluation is IDENTICAL to every previous fold run: big-5 test matches only.
Reports overall RPS/LL and the PROMOTED-TEAM subset (teams absent from the
big-5 the season before).
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
from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402

D2_LABELS = {"E1": "Championship", "SP2": "Segunda Division", "D2": "2. Bundesliga",
             "I2": "Serie B", "F2": "Ligue 2"}
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]


def load_d2() -> pd.DataFrame:
    rows = []
    for p in glob.glob(str(ROOT / "data/raw/football_data/*.csv")):
        m = re.search(r"(\d{2})(\d{2})_(E1|SP2|D2|I2|F2)\.csv$", p)
        if not m or int(m.group(1)) < 14:
            continue
        try:
            df = pd.read_csv(p, encoding="latin-1", on_bad_lines="skip",
                             usecols=lambda c: c in {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"})
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
        df["competition"] = D2_LABELS[m.group(3)]
        df["season"] = f"20{m.group(1)}-20{m.group(2)}"
        df["src"] = f"{m.group(1)}{m.group(2)}_{m.group(3)}"
        rows.append(df)
    r = pd.concat(rows, ignore_index=True).dropna(subset=["date", "HomeTeam", "FTHG"])
    r = r.drop_duplicates(subset=["date", "HomeTeam", "AwayTeam"])
    out = pd.DataFrame({
        "date": r["date"],
        "home_team": r["HomeTeam"].astype(str).str.lower().str.strip(),
        "away_team": r["AwayTeam"].astype(str).str.lower().str.strip(),
        "home_goals": pd.to_numeric(r["FTHG"], errors="coerce"),
        "away_goals": pd.to_numeric(r["FTAG"], errors="coerce"),
        "competition": r["competition"],
        "season": r["season"],
    }).dropna(subset=["home_goals", "away_goals"])
    out["match_id"] = ["d2_" + s + f"_{i:05d}" for i, s in enumerate(r["src"])]
    out["neutral"] = 0
    return out


def rps3(y_idx, P):
    Y = np.zeros_like(P)
    Y[np.arange(len(y_idx)), y_idx] = 1.0
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return float(((cp - cy) ** 2)[:, :2].sum(axis=1).mean() / 2)


def main() -> None:
    big5 = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv",
                       low_memory=False)
    big5 = big5[big5["xg_available"] == True].copy()  # noqa: E712
    for c in ["home_goals", "away_goals", "home_xg", "away_xg"]:
        big5[c] = pd.to_numeric(big5[c], errors="coerce")
    big5["date"] = pd.to_datetime(big5["date"], errors="coerce")
    big5 = big5.dropna(subset=["home_goals", "away_goals", "home_xg", "away_xg", "date"]).sort_values("date")
    d2 = load_d2()
    print(f"big5 {len(big5)} | D2 rows {len(d2)} ({d2.competition.nunique()} leagues)", flush=True)

    # promoted teams per big5 season: present in season s, absent in s-1 (big5)
    seasons_all = sorted(big5.season.unique())
    prev = {s: seasons_all[i - 1] for i, s in enumerate(seasons_all) if i > 0}
    team_season = big5.groupby("season").apply(
        lambda g: set(g.home_team) | set(g.away_team), include_groups=False)
    promoted = {s: team_season[s] - team_season[prev[s]] for s in TEST_SEASONS if s in prev}

    res = {arm: {"rps": [], "ll": [], "rps_pr": []} for arm in ["BIG5", "BIG5+D2"]}
    for s in TEST_SEASONS:
        te = big5[big5.season == s].sort_values("date")
        s_start = te.date.min()
        tr5 = big5[big5.date < s_start]
        tr_d2 = d2[d2.date < s_start]
        for arm, tr in [("BIG5", tr5), ("BIG5+D2", pd.concat([tr5, tr_d2], ignore_index=True))]:
            t0 = time.time()
            eng = PredictionEngine(use_xg_rate=True, blend_weight_gl=0.30).fit(tr)
            P, y, is_pr = [], [], []
            pr_set = promoted.get(s, set())
            for _, r in te.iterrows():
                p = eng.predict_match(str(r.home_team), str(r.away_team),
                                      competition=str(r.competition), neutral=False)
                P.append([p.p_home_win, p.p_draw, p.p_away_win])
                y.append(0 if r.home_goals > r.away_goals else (1 if r.home_goals == r.away_goals else 2))
                is_pr.append(r.home_team in pr_set or r.away_team in pr_set)
                if eng.xg_rate_model_ is not None:
                    eng.xg_rate_model_.update_form(r.home_team, r.away_team, r.home_xg, r.away_xg)
            P, y, is_pr = np.array(P), np.array(y), np.array(is_pr)
            res[arm]["rps"].append((rps3(y, P), len(y), s))
            res[arm]["ll"].append((float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean()), len(y), s))
            if is_pr.sum() > 50:
                res[arm]["rps_pr"].append((rps3(y[is_pr], P[is_pr]), int(is_pr.sum()), s))
            print(f"  {s} {arm}: done ({time.time()-t0:.0f}s, promoted n={int(is_pr.sum())})", flush=True)

    pool = lambda a: sum(x * n for x, n, _ in a) / max(sum(n for _, n, _ in a), 1)
    print("\n===== VERDICT =====")
    for met, label in [("rps", "RPS overall"), ("ll", "LL overall"), ("rps_pr", "RPS promoted-team matches")]:
        b, d = res["BIG5"][met], res["BIG5+D2"][met]
        folds = " ".join(f"{x[2][-2:]}{'+' if yv[0] < x[0] else '-'}" for x, yv in zip(b, d))
        print(f"{label}: BIG5 {pool(b):.4f} -> +D2 {pool(d):.4f} (d {pool(d)-pool(b):+.4f})  [{folds}]")


if __name__ == "__main__":
    main()
