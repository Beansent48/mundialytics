"""La Champions histórica: sides from any era, in the modern format.

HOW STRONG WAS MILAN 1989? The first version of this mode (the Streamlit one,
retired with it) answered through SquadLab's player bridge, from whatever
StatsBomb happened to have released of each side. It ranked Leicester 2016
above Bayern 2013 and Mourinho's Porto near the bottom: it measured how much
of a team had been filmed, not how good it was.

This one reads ClubElo instead: each club's real rating on 1 June of the year
the season ended — the same scale, the same Elo->goals calibration the rest of
the European layer is validated on (~1,000 real UCL/UEL/UECL matches). ClubElo
is zero-sum, so a 2000 in 1989 means what a 2000 means now: that far above the
European field of its day. That is the honest comparison across eras, and the
only one on offer — nobody measured Gullit and Rijkaard in duels won.

Three fields:
  campeones  European champions, 1956-2025 (data/curated/european_champions.csv,
             a list of historical facts); the 36 best of them by rating
  variado    one side per club, each at one of its ten best seasons, at random
  cualquiera any club-season since 1956 in the top 400 by rating, at random

Nottingham Forest (1979, 1980) and Steaua (1986) are missing: ClubElo's files
here do not carry them, and a stand-in would be a made-up strength.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
TEAMS_DIR = ROOT / "data/external/clubelo/teams"
CHAMPIONS = ROOT / "data/curated/european_champions.csv"
FIRST_YEAR, LAST_YEAR = 1956, 2025

POOLS = ("campeones", "variado", "cualquiera")

# ClubElo's file names are run together ("RealMadrid"); the reader wants them spaced
_DISPLAY = {
    "RealMadrid": "Real Madrid", "ManUnited": "Manchester United", "ManCity": "Manchester City",
    "ParisSG": "PSG", "AstonVilla": "Aston Villa", "CrvenaZvezda": "Estrella Roja",
    "Hamburg": "Hamburgo", "Marseille": "Marsella", "Atletico": "Atlético de Madrid",
    "Bilbao": "Athletic", "RBLeipzig": "RB Leipzig", "WestHam": "West Ham",
    "DynamoKyiv": "Dinamo de Kiev", "SpartakMoskva": "Spartak de Moscú",
    "Sociedad": "Real Sociedad", "Leverkusen": "Bayer Leverkusen", "Koeln": "Colonia",
    "Gladbach": "Borussia M'gladbach", "Werder": "Werder Bremen", "Sporting": "Sporting CP",
    "CrystalPalace": "Crystal Palace", "Tottenham": "Tottenham", "Fenerbahce": "Fenerbahçe",
    "Besiktas": "Beşiktaş", "Galatasaray": "Galatasaray", "SpartaPraha": "Sparta de Praga",
    "DinamoZagreb": "Dinamo de Zagreb", "Anderlecht": "Anderlecht", "Brugge": "Brujas",
}


def display_club(club: str) -> str:
    if club in _DISPLAY:
        return _DISPLAY[club]
    out = []
    for ch in club:
        if ch.isupper() and out and out[-1] != " ":
            out.append(" ")
        out.append(ch)
    return "".join(out)


def season_label(year: int) -> str:
    """1989 -> '1988/89' (the season that ended that summer)."""
    return f"{year - 1}/{str(year)[-2:]}"


@lru_cache(maxsize=1)
def _histories() -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(TEAMS_DIR.glob("*.csv")):
        h = pd.read_csv(f, usecols=["Elo", "From", "To"])
        h["From"] = pd.to_datetime(h["From"], errors="coerce")
        h["To"] = pd.to_datetime(h["To"], errors="coerce")
        out[f.stem] = h.dropna().sort_values("From").reset_index(drop=True)
    return out


def elo_on(club: str, when: pd.Timestamp) -> float | None:
    h = _histories().get(club)
    if h is None or h.empty:
        return None
    i = int(np.searchsorted(h["From"].to_numpy(), np.datetime64(when), side="right")) - 1
    if i < 0 or h["To"].iloc[i] < when:
        return None
    return float(h["Elo"].iloc[i])


@lru_cache(maxsize=1)
def team_seasons() -> pd.DataFrame:
    """Every club's rating on 1 June of every year it has one, 1956-2025."""
    rows = []
    for club in _histories():
        for year in range(FIRST_YEAR, LAST_YEAR + 1):
            e = elo_on(club, pd.Timestamp(f"{year}-06-01"))
            if e is not None:
                rows.append({"club": club, "year": year, "elo": e})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["label"] = [f"{display_club(c)} {season_label(y)}" for c, y in zip(df["club"], df["year"])]
    return df


@lru_cache(maxsize=1)
def champions() -> pd.DataFrame:
    if not CHAMPIONS.exists():
        return pd.DataFrame(columns=["club", "year", "elo", "label"])
    c = pd.read_csv(CHAMPIONS)
    c["elo"] = [elo_on(cl, pd.Timestamp(f"{y}-06-01")) for cl, y in zip(c["club"], c["year"])]
    c = c.dropna(subset=["elo"])
    c["label"] = [f"{n} {season_label(y)}" for n, y in zip(c["name"], c["year"])]
    c["champion"] = True
    return c[["club", "year", "elo", "label", "champion"]]


def entrants(pool: str, seed: int, n: int = 36) -> pd.DataFrame:
    """The 36 sides of one historic Champions League."""
    rng = np.random.default_rng(int(seed))
    if pool == "campeones":
        return champions().nlargest(n, "elo").reset_index(drop=True)
    ts = team_seasons()
    champ = champions()
    won = set(zip(champ["club"], champ["year"]))
    if pool == "variado":
        best = (ts.sort_values("elo", ascending=False).groupby("club").head(10))
        # the clubs with the highest peaks, then one of each club's best ten seasons
        peaks = best.groupby("club")["elo"].max().nlargest(60).index
        clubs = rng.choice(np.array(peaks), size=min(n, len(peaks)), replace=False)
        rows = []
        for c in clubs:
            opts = best[best["club"] == c]
            rows.append(opts.iloc[int(rng.integers(len(opts)))])
        out = pd.DataFrame(rows)
    else:
        top = ts.nlargest(400, "elo")
        out = top.iloc[rng.choice(len(top), size=min(n, len(top)), replace=False)]
    out = out.copy()
    out["champion"] = [(c, y) in won for c, y in zip(out["club"], out["year"])]
    return out.sort_values("elo", ascending=False).reset_index(drop=True)


@dataclass
class HistoricResult:
    entrants: pd.DataFrame
    table: pd.DataFrame
    rounds: dict
    champion: str
    runner_up: str
    odds: pd.DataFrame


def play(pool: str = "campeones", seed: int = 7, n_sims: int = 600) -> HistoricResult:
    """One historic Champions: a random draw, the league phase, the bracket.

    Played on the validated European Elo layer (EuropeanTournament), then the
    same field simulated `n_sims` times so the reader can see whether the
    champion was the favourite or a surprise.
    """
    from mundialytics.statistical_core.competition.european import (
        EuropeanTournament, load_calibration,
    )

    if pool not in POOLS:
        raise ValueError(f"unknown pool {pool!r}")
    ent = entrants(pool, seed)
    if len(ent) < 36:
        raise ValueError(f"only {len(ent)} sides available for {pool!r}")
    elo = {r.label: float(r.elo) for r in ent.itertuples()}
    calib = load_calibration(ROOT)
    tour = EuropeanTournament("champions", elo, calib, rng=np.random.default_rng(int(seed) * 1000 + 7))
    res = tour.play_single()
    odds_tour = EuropeanTournament("champions", elo, calib, rng=np.random.default_rng(int(seed) * 1000 + 8))
    odds = odds_tour.simulate(n_sims)
    return HistoricResult(ent, res["table"], res["rounds"], res["champion"], res["runner_up"], odds)
