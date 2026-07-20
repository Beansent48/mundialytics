"""
Build a LeagueState from the foundation match CSV at any cutoff.

This is the "from the current point" mechanism for Phase 1. Because the newest
season in the foundation data is already complete, "simulate from now" is realised
by splitting a season at a cutoff: everything before the cutoff is treated as
played (real results kept), everything on/after it as remaining (results stripped
-> home/away/date only). For a completed season this split doubles as a
leakage-free backtest: the real outcome of the remaining fixtures is known, so
predicted probabilities can be scored against it.

When a live data feed is wired in later, a sibling loader will produce the same
LeagueState from the live schedule + results, and nothing downstream changes.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from mundialytics.statistical_core.competition.state import FIXTURE_COLUMNS, LeagueState
from mundialytics.statistical_core.schemas import canonical_name

_DEFAULT_FOUNDATION = Path("data/processed/foundation_big5_multi_season.csv")


def _resolve_cutoff_date(
    season_matches: pd.DataFrame,
    cutoff_date: str | datetime | None,
    cutoff_matchday: int | None,
) -> datetime | None:
    """Turn a date or a matchday index into a concrete cutoff datetime.

    A "matchday" here is approximated by round number: teams are typically all
    playing once per calendar round, so the Nth matchday boundary is taken as the
    date just after the Nth-round block of fixtures. We use the (matches_per_round)
    heuristic from the team count rather than assuming a fixed 10.
    """
    if cutoff_date is not None:
        return pd.to_datetime(cutoff_date)
    if cutoff_matchday is not None:
        n_teams = pd.concat([season_matches["home_team"], season_matches["away_team"]]).nunique()
        per_round = max(n_teams // 2, 1)
        dates_sorted = season_matches.sort_values("date")["date"].reset_index(drop=True)
        idx = cutoff_matchday * per_round
        if idx >= len(dates_sorted):
            # Cutoff beyond the season -> everything played.
            return pd.to_datetime(dates_sorted.iloc[-1]) + pd.Timedelta(days=1)
        # Cut just before the fixture that opens matchday (cutoff_matchday+1).
        return pd.to_datetime(dates_sorted.iloc[idx])
    return None


def load_league_state_from_foundation(
    competition: str,
    season: str,
    cutoff_date: str | datetime | None = None,
    cutoff_matchday: int | None = None,
    foundation: pd.DataFrame | str | Path | None = None,
) -> LeagueState:
    """Construct a LeagueState for one league-season at a cutoff.

    Parameters
    ----------
    competition, season
        Must match the foundation values, e.g. "LaLiga" / "2024-2025".
    cutoff_date
        Matches strictly before this are played; on/after are remaining.
    cutoff_matchday
        Alternative to ``cutoff_date``: cut after this many rounds. Ignored if
        ``cutoff_date`` is given. If neither is given, the whole season is played.
    foundation
        A preloaded DataFrame, a path, or ``None`` to read the default CSV.

    Returns
    -------
    LeagueState with played / remaining / standings populated.
    """
    if foundation is None:
        foundation = _DEFAULT_FOUNDATION
    if isinstance(foundation, (str, Path)):
        foundation = pd.read_csv(foundation, low_memory=False)

    df = foundation[
        (foundation["competition"] == competition) & (foundation["season"] == season)
    ].copy()
    if df.empty:
        raise ValueError(
            f"No matches found for competition={competition!r} season={season!r}. "
            f"Check the exact spelling against the foundation file."
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    teams = sorted(set(df["home_team"].map(canonical_name)) | set(df["away_team"].map(canonical_name)))

    cutoff = _resolve_cutoff_date(df, cutoff_date, cutoff_matchday)
    if cutoff is None:
        played = df
        remaining = df.iloc[0:0]
    else:
        played = df[df["date"] < cutoff]
        remaining = df[df["date"] >= cutoff]

    # Strip results from remaining fixtures — the state must not leak the future.
    keep = [c for c in FIXTURE_COLUMNS if c in remaining.columns]
    extra = [c for c in ("neutral", "stage") if c in remaining.columns]
    remaining = remaining[keep + extra].copy()

    return LeagueState(
        competition=competition,
        season=season,
        cutoff_date=cutoff,
        teams=teams,
        played=played.copy(),
        remaining=remaining,
    )
