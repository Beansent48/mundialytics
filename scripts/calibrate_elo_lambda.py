from __future__ import annotations

"""European-tournament backbone, phase 1 (the MEASURABLE part).

Fetches ClubElo histories for every big-5 foundation team (cached under
data/external/clubelo/teams/), joins PRE-MATCH Elo to foundation matches
2016+, and fits the Elo->goals mapping used by the tournament simulator:

    log lam_home = c + hfa + b * (Elo_h - Elo_a) / 400
    log lam_away = c       - b * (Elo_h - Elo_a) / 400

Walk-forward folds report how good Elo-only match probabilities are on OUR
matches (expected: clearly worse than the deployed engine — that is fine, the
mapping just has to be calibrated, the cross-league scale is ClubElo's job).
Outputs data/processed/elo_lambda_calibration.json with the constants.
"""

import json
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
TEAMS_DIR = ROOT / "data/external/clubelo/teams"
OUT = ROOT / "data/processed/elo_lambda_calibration.json"
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]

# foundation (football-data lowercase) -> ClubElo club name, where they differ
FD_TO_CLUBELO = {
    "man city": "ManCity", "man united": "ManUnited", "ath madrid": "Atletico",
    "ath bilbao": "Bilbao", "espanol": "Espanyol", "sociedad": "Sociedad",
    "betis": "Betis", "celta": "Celta", "vallecano": "RayoVallecano",
    "st etienne": "SaintEtienne", "paris sg": "ParisSG", "leverkusen": "Leverkusen",
    "bayern munich": "Bayern", "dortmund": "Dortmund", "ein frankfurt": "Frankfurt",
    "fc koln": "Koeln", "m'gladbach": "Gladbach", "hertha": "Hertha",
    "hoffenheim": "Hoffenheim", "mainz": "Mainz", "rb leipzig": "RBLeipzig",
    "schalke 04": "Schalke", "werder bremen": "Werder", "wolfsburg": "Wolfsburg",
    "inter": "Inter", "milan": "Milan", "juventus": "Juventus", "napoli": "Napoli",
    "roma": "Roma", "lazio": "Lazio", "atalanta": "Atalanta", "fiorentina": "Fiorentina",
    "nott'm forest": "Forest", "sheffield united": "SheffieldUnited",
    "sheffield weds": "SheffieldWeds", "west brom": "WestBrom", "west ham": "WestHam",
    "crystal palace": "CrystalPalace", "aston villa": "AstonVilla",
    "queens park rangers": "QPR", "real sociedad": "Sociedad",
}


def clubelo_name(fd: str) -> str:
    if fd in FD_TO_CLUBELO:
        return FD_TO_CLUBELO[fd]
    return "".join(w.capitalize() for w in fd.split())


def fetch_history(club: str) -> pd.DataFrame | None:
    cache = TEAMS_DIR / f"{club}.csv"
    if cache.exists() and cache.stat().st_size > 200:
        return pd.read_csv(cache)
    try:
        r = requests.get(f"http://api.clubelo.com/{club}", timeout=30)
        if r.status_code != 200 or len(r.text) < 100:
            return None
        df = pd.read_csv(StringIO(r.text))
        if df.empty or "Elo" not in df.columns:
            return None
        TEAMS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        time.sleep(0.4)
        return df
    except Exception:
        return None


def main() -> None:
    found = pd.read_csv(ROOT / "data/processed/foundation_big5_multi_season.csv", low_memory=False)
    found = found[found["season"] >= "2016-2017"].copy()
    found["date"] = pd.to_datetime(found["date"], errors="coerce")
    teams = sorted(set(found.home_team) | set(found.away_team))
    print(f"{len(teams)} big-5 teams to resolve", flush=True)

    histories: dict[str, pd.DataFrame] = {}
    missing = []
    for t in teams:
        h = fetch_history(clubelo_name(t))
        if h is None:
            missing.append(t)
            continue
        h = h.copy()
        h["From"] = pd.to_datetime(h["From"], errors="coerce")
        h["To"] = pd.to_datetime(h["To"], errors="coerce")
        histories[t] = h[["From", "To", "Elo"]].dropna()
    print(f"histories fetched: {len(histories)} | missing: {len(missing)} -> {missing[:12]}", flush=True)

    def elo_at(team: str, when: pd.Timestamp) -> float | None:
        h = histories.get(team)
        if h is None:
            return None
        m = h[(h.From <= when) & (h.To >= when)]
        return float(m.Elo.iloc[0]) if len(m) else None

    rows = []
    for r in found.itertuples(index=False):
        eh = elo_at(r.home_team, r.date)
        ea = elo_at(r.away_team, r.date)
        if eh is None or ea is None:
            continue
        rows.append((r.season, r.date, r.home_goals, r.away_goals, eh, ea))
    m = pd.DataFrame(rows, columns=["season", "date", "hg", "ag", "eh", "ea"]).dropna()
    m["d400"] = (m.eh - m.ea) / 400.0
    print(f"matches with pre-match Elo: {len(m)} / {len(found)}", flush=True)

    # walk-forward evaluation + final constants on ALL data
    from sklearn.linear_model import PoissonRegressor

    def fit_constants(df: pd.DataFrame) -> tuple[float, float, float]:
        X = np.concatenate([np.column_stack([np.ones(len(df)), df.d400]),
                            np.column_stack([np.zeros(len(df)), -df.d400])])
        y = np.concatenate([df.hg, df.ag])
        reg = PoissonRegressor(alpha=1e-4, max_iter=1000, fit_intercept=True).fit(X, y)
        c = float(reg.intercept_)
        hfa = float(reg.coef_[0])
        b = float(reg.coef_[1])
        return c, hfa, b

    def rps3(y_idx, P):
        Y = np.zeros_like(P)
        Y[np.arange(len(y_idx)), y_idx] = 1.0
        cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
        return float(((cp - cy) ** 2)[:, :2].sum(axis=1).mean() / 2)

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from mundialytics.statistical_core.distributions import outcome_probabilities

    res = []
    for s in TEST_SEASONS:
        tr = m[m.season < s]
        te = m[m.season == s]
        if len(tr) < 3000 or len(te) == 0:
            continue
        c, hfa, b = fit_constants(tr)
        P, y = [], []
        for r in te.itertuples(index=False):
            lh = float(np.exp(c + hfa + b * r.d400))
            la = float(np.exp(c - b * r.d400))
            p = outcome_probabilities(lh, la, dixon_coles_rho=-0.07)
            P.append([p["p_home_win"], p["p_draw"], p["p_away_win"]])
            y.append(0 if r.hg > r.ag else (1 if r.hg == r.ag else 2))
        res.append((rps3(np.array(y), np.array(P)), len(te), s))
    pool = sum(x * n for x, n, _ in res) / sum(n for _, n, _ in res)
    print("\nElo-only match model, walk-forward folds:")
    for x, n, s in res:
        print(f"  {s}: RPS {x:.4f} (n={n})")
    print(f"POOLED RPS {pool:.4f}  (deployed engine ref ~0.2003; Elo-only is the "
          f"cross-league fallback, not a replacement)")

    c, hfa, b = fit_constants(m)
    OUT.write_text(json.dumps({
        "c": c, "hfa": hfa, "b": b,
        "pooled_rps_elo_only": pool,
        "n_matches": len(m), "fitted_through": str(m.date.max())[:10],
        "model": "log lam_h = c + hfa + b*(eloH-eloA)/400 ; log lam_a = c - b*(...)",
    }, indent=2))
    print(f"\nWROTE {OUT}: c={c:.4f} hfa={hfa:.4f} b={b:.4f}")


if __name__ == "__main__":
    main()
