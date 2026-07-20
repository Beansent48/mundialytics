"""
LeagueState — the current state of a league competition.

This is the single source of truth for a competition at a point in time:
  - played    : matches already played (with real results)
  - remaining : fixtures not yet played (home/away/date only, no result)
  - standings : the live table derived from ``played``

The same object serves two consumers:
  1. Browsing UI  — render the table, past results, today's and future fixtures.
  2. Simulator    — resume Monte Carlo from ``standings`` over ``remaining``.

The data SOURCE is decoupled: today a CSV+cutoff loader (see ``cutoff.py``) builds
this; later a live-feed loader can build the exact same object, so nothing above
LeagueState changes when real-time data is wired in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from mundialytics.statistical_core.competition.standings import compute_standings
from mundialytics.statistical_core.schemas import canonical_name

# Columns a remaining-fixture frame is guaranteed to expose (results stripped).
FIXTURE_COLUMNS = ["date", "home_team", "away_team", "competition", "season", "match_id"]


@dataclass
class LeagueState:
    """Immutable snapshot of a league at a cutoff.

    Attributes
    ----------
    competition, season
        Identifiers, e.g. "LaLiga" / "2024-2025".
    cutoff_date
        Everything strictly before this counts as played; on/after it is remaining.
        ``None`` means "use whatever split the loader produced" (e.g. a live feed).
    teams
        Canonical team list for the season (all 20, even if a team has 0 remaining).
    played, remaining
        Match DataFrames. ``remaining`` carries no goals columns.
    """
    competition: str
    season: str
    cutoff_date: datetime | None
    teams: list[str]
    played: pd.DataFrame
    remaining: pd.DataFrame
    _standings: pd.DataFrame | None = field(default=None, repr=False)

    # ── Derived views ──────────────────────────────────────────────────────────

    @property
    def standings(self) -> pd.DataFrame:
        """Live table from played matches (cached)."""
        if self._standings is None:
            self._standings = compute_standings(
                self.played, teams=self.teams, competition=self.competition
            )
        return self._standings

    @property
    def n_played(self) -> int:
        return len(self.played)

    @property
    def n_remaining(self) -> int:
        return len(self.remaining)

    @property
    def is_complete(self) -> bool:
        return self.n_remaining == 0

    def remaining_for(self, team: str) -> pd.DataFrame:
        """Fixtures the given team still has to play (home or away)."""
        t = canonical_name(team)
        r = self.remaining
        mask = (r["home_team"].map(canonical_name) == t) | (r["away_team"].map(canonical_name) == t)
        return r[mask]

    def current_points(self) -> dict[str, int]:
        """team -> points already earned (the anchor the simulator resumes from)."""
        s = self.standings
        return dict(zip(s["team"], s["points"]))

    def summary(self) -> str:
        leader = self.standings.iloc[0] if len(self.standings) else None
        head = (f"{leader['team']} {leader['points']}pts" if leader is not None else "—")
        return (
            f"{self.competition} {self.season} @ "
            f"{self.cutoff_date.date() if self.cutoff_date else 'live'} | "
            f"played={self.n_played} remaining={self.n_remaining} | leader: {head}"
        )
