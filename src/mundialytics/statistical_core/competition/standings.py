"""
League standings table computed from played matches.

Produces the classic table (played, W/D/L, GF/GA/GD, points) and a rank, applying
tiebreakers. The default tiebreak order is points -> goal difference -> goals for,
which matches the Premier League, Bundesliga and Ligue 1. La Liga and Serie A break
ties on head-to-head first; that refinement is supported via ``tiebreak`` and
applied when the tied set is fully decided by their mutual results.

This module is deliberately pure/stateless: it takes a DataFrame of played matches
and returns a table. Both the browsing UI and the Monte Carlo simulator call it
(the simulator re-ranks a full played+simulated set with the exact same logic, so a
live table and a simulated final table are always consistent).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mundialytics.statistical_core.schemas import canonical_name


# Default competition points scheme (3 for a win, 1 for a draw).
WIN_POINTS = 3
DRAW_POINTS = 1

# Leagues whose first tiebreak is head-to-head (not goal difference).
HEAD_TO_HEAD_LEAGUES = {"laliga", "la liga", "serie a"}


@dataclass
class StandingsRow:
    team: str
    played: int
    won: int
    drawn: int
    lost: int
    goals_for: int
    goals_against: int
    goal_diff: int
    points: int
    rank: int


def _empty_record() -> dict:
    return {
        "played": 0, "won": 0, "drawn": 0, "lost": 0,
        "goals_for": 0, "goals_against": 0, "points": 0,
    }


def _head_to_head_points(team: str, tied: list[str], h2h: dict) -> int:
    """Points ``team`` earned in matches against the other tied teams only."""
    pts = 0
    for other in tied:
        if other == team:
            continue
        for (h, a), (hg, ag) in h2h.items():
            if h == team and a == other:
                pts += WIN_POINTS if hg > ag else (DRAW_POINTS if hg == ag else 0)
            elif h == other and a == team:
                pts += WIN_POINTS if ag > hg else (DRAW_POINTS if ag == hg else 0)
    return pts


def compute_standings(
    played: pd.DataFrame,
    teams: list[str] | None = None,
    competition: str | None = None,
) -> pd.DataFrame:
    """Build a league table from played matches.

    Parameters
    ----------
    played
        Match rows with ``home_team``, ``away_team``, ``home_goals``, ``away_goals``.
        Rows with a missing score are ignored (treated as not-yet-played).
    teams
        Full team list for the competition. Teams with zero played matches still
        appear (0 points). If ``None``, inferred from the matches present.
    competition
        Used to pick the tiebreak rule. ``None`` -> goal-difference tiebreak.

    Returns
    -------
    DataFrame sorted by final rank, one row per team, columns matching
    ``StandingsRow`` fields.
    """
    df = played.copy()
    for col in ("home_goals", "away_goals"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["home_team"] = df["home_team"].map(canonical_name)
    df["away_team"] = df["away_team"].map(canonical_name)

    if teams is None:
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    else:
        teams = [canonical_name(t) for t in teams]

    records = {t: _empty_record() for t in teams}
    h2h: dict[tuple[str, str], tuple[int, int]] = {}

    for r in df.itertuples(index=False):
        h, a = r.home_team, r.away_team
        hg, ag = int(r.home_goals), int(r.away_goals)
        if h not in records or a not in records:
            # A team not in the declared list (e.g. mid-season data glitch) —
            # add it rather than silently drop the match.
            records.setdefault(h, _empty_record())
            records.setdefault(a, _empty_record())
        h2h[(h, a)] = (hg, ag)
        for team, gf, ga in ((h, hg, ag), (a, ag, hg)):
            rec = records[team]
            rec["played"] += 1
            rec["goals_for"] += gf
            rec["goals_against"] += ga
            if gf > ga:
                rec["won"] += 1
                rec["points"] += WIN_POINTS
            elif gf == ga:
                rec["drawn"] += 1
                rec["points"] += DRAW_POINTS
            else:
                rec["lost"] += 1

    use_h2h = competition is not None and canonical_name(competition).lower() in HEAD_TO_HEAD_LEAGUES

    def sort_key(team: str) -> tuple:
        rec = records[team]
        gd = rec["goals_for"] - rec["goals_against"]
        tied = [t for t in records if records[t]["points"] == rec["points"]]
        h2h_pts = _head_to_head_points(team, tied, h2h) if (use_h2h and len(tied) > 1) else 0
        # Higher is better for every component -> sort descending.
        return (rec["points"], h2h_pts, gd, rec["goals_for"])

    ranked = sorted(records.keys(), key=sort_key, reverse=True)

    rows = []
    for rank, team in enumerate(ranked, start=1):
        rec = records[team]
        gd = rec["goals_for"] - rec["goals_against"]
        rows.append({
            "team": team,
            "played": rec["played"],
            "won": rec["won"],
            "drawn": rec["drawn"],
            "lost": rec["lost"],
            "goals_for": rec["goals_for"],
            "goals_against": rec["goals_against"],
            "goal_diff": gd,
            "points": rec["points"],
            "rank": rank,
        })
    return pd.DataFrame(rows)
