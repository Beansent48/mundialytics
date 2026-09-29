"""How many of a team's usual starters are missing from today's XI.

The one team-news signal that survived the backtest: with the share of the regular XI not
starting, a total-goals-preserving tilt improved 1X2 RPS in 6 of 6 seasons
(scripts/lineup_pass/backtest_absence.py). Lineups come out about an hour before
kick-off, so this feeds the pre-kickoff lineup pass (scripts/log_lineup_pass.py), not the
morning prediction.

Regulars = the 11 players with the most starts in the team's last 10 league matches of
the current season, with at least 5 played. The ESPN history the pass reads holds the
current season only, and the backtest uses the same rule, so the deployed coefficient
matches what is served.
"""
from __future__ import annotations

import pandas as pd

WINDOW = 10
MIN_MATCHES = 5
XI = 11


def regular_xi(history: pd.DataFrame, before) -> list | None:
    """The team's usual starters before `before`, or None when there is too little history.

    `history`: one team's rows for one season with columns date, player, starter (bool).
    Ties in starts go to the player who started more recently.
    """
    h = history[pd.to_datetime(history["date"]) < pd.Timestamp(before)]
    dates = sorted(pd.to_datetime(h["date"]).unique())
    if len(dates) < MIN_MATCHES:
        return None
    recent = h[pd.to_datetime(h["date"]).isin(dates[-WINDOW:])]
    st = recent[recent["starter"].astype(bool)]
    if st.empty:
        return None
    agg = (st.assign(date=pd.to_datetime(st["date"]))
             .groupby("player").agg(starts=("date", "size"), last=("date", "max"))
             .sort_values(["starts", "last"], ascending=False))
    return list(agg.index[:XI])


def absent_share(regulars: list, starters) -> float:
    """Share of `regulars` not in `starters` (0.0 = full-strength XI)."""
    if not regulars:
        return 0.0
    s = set(starters)
    return sum(p not in s for p in regulars) / len(regulars)
