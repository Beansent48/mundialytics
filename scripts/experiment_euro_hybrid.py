from __future__ import annotations

"""Hybrid test: on European matches where BOTH clubs are big-5, is our deployed
engine better than the euro-calibrated Elo mapping?

Engine arm: fit on big-5 foundation before each European season (frozen form —
slightly unfair to the engine vs its live walk-forward deployment, so a win
here is a conservative one). Elo arm: euro constants. Blend arm: 50/50 lambdas.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mundialytics.statistical_core.distributions import outcome_probabilities  # noqa: E402
from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402

from validate_european_mapping import (  # noqa: E402
    CALIB, ALIASES, download_results, fetch_history, norm, resolve_names)

# ClubElo name -> foundation name for big-5 clubs (reverse of the calibration map)
CLUBELO_TO_FD = {
    "Arsenal": "arsenal", "ManCity": "man city", "ManUnited": "man united",
    "Liverpool": "liverpool", "Chelsea": "chelsea", "Tottenham": "tottenham",
    "Newcastle": "newcastle", "AstonVilla": "aston villa", "WestHam": "west ham",
    "Brighton": "brighton", "Bournemouth": "bournemouth", "Everton": "everton",
    "Forest": "nott'm forest", "Real Madrid": "real madrid", "RealMadrid": "real madrid",
    "Barcelona": "barcelona", "Atletico": "ath madrid", "Sevilla": "sevilla",
    "Villarreal": "villarreal", "Sociedad": "sociedad", "Betis": "betis",
    "Bilbao": "ath bilbao", "Girona": "girona", "Valencia": "valencia",
    "Bayern": "bayern munich", "Dortmund": "dortmund", "Leverkusen": "leverkusen",
    "RBLeipzig": "rb leipzig", "Frankfurt": "ein frankfurt", "Stuttgart": "stuttgart",
    "UnionBerlin": "union berlin", "Freiburg": "freiburg", "Hoffenheim": "hoffenheim",
    "Inter": "inter", "Milan": "milan", "Juventus": "juventus", "Napoli": "napoli",
    "Roma": "roma", "Lazio": "lazio", "Atalanta": "atalanta", "Fiorentina": "fiorentina",
    "Bologna": "bologna", "Torino": "torino", "ParisSG": "paris sg", "Marseille": "marseille",
    "Monaco": "monaco", "Lille": "lille", "Lyon": "lyon", "Nice": "nice", "Lens": "lens",
    "Brest": "brest", "Rennes": "rennes", "Strasbourg": "strasbourg",
}


def rps3(y_idx, P):
    Y = np.zeros_like(P)
    Y[np.arange(len(y_idx)), y_idx] = 1.0
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return float(((cp - cy) ** 2)[:, :2].sum(axis=1).mean() / 2)


def main() -> None:
    from mundialytics.statistical_core.competition.european import fetch_current_elo
    euro = pd.read_json(ROOT / "data/processed/elo_lambda_calibration_euro.json", typ="series")
    cE, hfaE, bE = euro["c"], euro["hfa"], euro["b"]

    elo_now = fetch_current_elo(ROOT)
    m = download_results()
    m, _ = resolve_names(m, list(elo_now))
    m["home_fd"] = m["home"].map(CLUBELO_TO_FD)
    m["away_fd"] = m["away"].map(CLUBELO_TO_FD)
    m = m.dropna(subset=["home_fd", "away_fd"])
    print(f"partidos europeos big5-vs-big5: {len(m)}", flush=True)

    # pre-match Elo
    hist = {c: fetch_history(c) for c in sorted(set(m.home) | set(m.away))}

    def elo_at(team, when):
        h = hist.get(team)
        if h is None:
            return None
        r = h[(h.From <= when) & (h.To >= when)]
        return float(r.Elo.iloc[0]) if len(r) else None

    found = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv",
                        low_memory=False)
    found = found[found["xg_available"] == True].copy()  # noqa: E712
    for c in ["home_goals", "away_goals", "home_xg", "away_xg"]:
        found[c] = pd.to_numeric(found[c], errors="coerce")
    found["date"] = pd.to_datetime(found["date"], errors="coerce")
    found = found.dropna(subset=["home_goals", "away_goals", "home_xg", "away_xg", "date"])

    res = {arm: [] for arm in ["ELO", "ENGINE", "BLEND"]}
    for edition, g in sorted(m.groupby("edition")):
        season_start = pd.Timestamp(f"{edition[:4]}-08-01")
        tr = found[found["date"] < season_start]
        if len(tr) < 3000:
            continue
        eng = PredictionEngine(blend_weight_gl=0.30, sharpen_gamma_1x2=1.3,
                               rescale_lambda_to_goals=True, outcome_rho=-0.17,
                               xg_rate_kwargs={"use_ewma": True}).fit(tr)
        rows = {arm: [] for arm in res}
        y = []
        for r in g.itertuples(index=False):
            eh, ea = elo_at(r.home, r.date), elo_at(r.away, r.date)
            if eh is None or ea is None:
                continue
            d400 = (eh - ea) / 400.0
            lh_e = float(np.exp(cE + hfaE + bE * d400))
            la_e = float(np.exp(cE - bE * d400))
            try:
                p_eng = eng.predict_match(r.home_fd, r.away_fd, competition=None, neutral=False)
                lh_g, la_g = p_eng.lambda_home, p_eng.lambda_away
            except Exception:
                continue
            y.append(0 if r.hg > r.ag else (1 if r.hg == r.ag else 2))
            for arm, (lh, la) in [("ELO", (lh_e, la_e)), ("ENGINE", (lh_g, la_g)),
                                  ("BLEND", ((lh_e + lh_g) / 2, (la_e + la_g) / 2))]:
                p = outcome_probabilities(lh, la, dixon_coles_rho=-0.07)
                rows[arm].append([p["p_home_win"], p["p_draw"], p["p_away_win"]])
        if not y:
            continue
        y = np.array(y)
        for arm in res:
            res[arm].append((rps3(y, np.array(rows[arm])), len(y), edition))
        print(f"  {edition} (n={len(y)}): " + " | ".join(
            f"{a} {rps3(y, np.array(rows[a])):.4f}" for a in res), flush=True)

    pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
    print("\nPOOLED big5-vs-big5 europeo:")
    for arm in res:
        n = sum(n_ for _, n_, _ in res[arm])
        print(f"  {arm:7s} RPS {pool(res[arm]):.4f} (n={n})")


if __name__ == "__main__":
    main()
