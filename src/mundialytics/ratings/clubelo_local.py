from __future__ import annotations

"""Advance a ClubElo snapshot locally, so the European layer stops depending on
an API that goes down.

WHY. The European simulator needs a pan-European rating. Its source, the ClubElo
API, returned HTTP 502 for three consecutive days on 2026-09-03 while the
Champions League was five days away, and the newest cached snapshot was six
weeks old. A rating frozen before the transfer window is not an honest input.

WHY WE STILL SEED FROM CLUBELO rather than using our own EloRater outright: the
foundation is Big5-only, and just 18 of the 36 Champions clubs appear in it.
Bodø/Glimt, Club Brugge, Feyenoord, Galatasaray, Slavia Praha and the rest of the
field have no matches we hold. ClubElo's 597-club snapshot is the only
pan-European scale available, so it is the seed — but only once, after which we
carry it forward ourselves from results we do have.

WHY THE CONSTANTS ARE FITTED, NOT COPIED. ClubElo's published constants are not
something to reproduce from memory, and they cannot be reverse-engineered here:
the eleven `*_clubelo/clubelo_match_features.csv` files record snapshot dates but
their value columns are 0% populated, so there is no ClubElo history to fit
against. Instead this uses a standard goal-difference-weighted Elo whose two free
parameters are fitted on our own matches by out-of-sample predictive accuracy
(scripts/fit_local_elo.py). The result is not a ClubElo replica and is not
claimed to be one; it is a rating on the same scale, advanced by real results.

Ratings for clubs with no matches since the seed are carried forward unchanged
and flagged stale, never presented as if freshly computed.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEED_DIR = "data/external/clubelo/daily"
DEFAULT_K = 32.0
DEFAULT_HFA = 55.0


@dataclass
class EloParams:
    """Free parameters of the update rule, fitted by scripts/fit_local_elo.py."""
    k: float = DEFAULT_K
    hfa: float = DEFAULT_HFA


def expected_home(elo_h: float, elo_a: float, hfa: float, neutral: bool = False) -> float:
    """Logistic expectation on the standard 400-point scale."""
    diff = (elo_h + (0.0 if neutral else hfa)) - elo_a
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def goal_diff_multiplier(margin: int) -> float:
    """Weight a result by its margin — the usual World-Football-Elo ladder.

    1 goal x1, 2 goals x1.5, 3 goals x1.75, then +1/8 per extra goal. Without
    this a 5-0 and a 1-0 move a rating identically, which no rating system that
    tracks strength should do.
    """
    m = abs(int(margin))
    if m <= 1:
        return 1.0
    if m == 2:
        return 1.5
    if m == 3:
        return 1.75
    return 1.75 + (m - 3) / 8.0


def load_seed(root: str | Path) -> tuple[dict[str, float], str]:
    """Newest cached ClubElo snapshot -> {club: elo}, plus its date."""
    d = Path(root) / SEED_DIR
    snaps = sorted(d.glob("*.csv")) if d.exists() else []
    if not snaps:
        raise FileNotFoundError(f"no ClubElo snapshot under {d}")
    latest = snaps[-1]
    df = pd.read_csv(latest)
    df = df.dropna(subset=["Club", "Elo"])
    return (dict(zip(df["Club"].astype(str), df["Elo"].astype(float))), latest.stem)


def roll_forward(seed: dict[str, float], matches: pd.DataFrame,
                 params: EloParams | None = None,
                 resolver=None) -> tuple[dict[str, float], dict[str, str]]:
    """Apply matches chronologically to a seed rating.

    `matches` needs date, home_team, away_team, home_goals, away_goals and may
    carry `neutral`. `resolver` maps a match-file club name onto a seed key;
    unmapped clubs are skipped rather than silently invented.

    Returns the updated ratings and, per club, the date it was last updated.
    """
    p = params or EloParams()
    elo = dict(seed)
    last: dict[str, str] = {}
    if matches.empty:
        return elo, last

    m = matches.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"])
    m = m.sort_values("date")
    for r in m.itertuples(index=False):
        h = resolver(r.home_team) if resolver else r.home_team
        a = resolver(r.away_team) if resolver else r.away_team
        if not h or not a or h not in elo or a not in elo:
            continue
        neutral = bool(getattr(r, "neutral", 0))
        exp_h = expected_home(elo[h], elo[a], p.hfa, neutral)
        hg, ag = float(r.home_goals), float(r.away_goals)
        score_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        k = p.k * goal_diff_multiplier(hg - ag)
        delta = k * (score_h - exp_h)
        elo[h] += delta
        elo[a] -= delta
        stamp = str(getattr(r, "date", ""))[:10]
        last[h] = last[a] = stamp
    return elo, last


def to_frame(elo: dict[str, float], last: dict[str, str], seed_date: str) -> pd.DataFrame:
    """Ratings with provenance: when each was last moved, and whether it is stale."""
    rows = [{"club": c, "elo": round(v, 2),
             "last_updated": last.get(c, seed_date),
             "stale": c not in last}
            for c, v in sorted(elo.items(), key=lambda kv: -kv[1])]
    out = pd.DataFrame(rows)
    out["seed_date"] = seed_date
    return out
