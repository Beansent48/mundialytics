"""
The top-scorer race.

Who ends the season as the league's leading scorer, simulated from the fixtures
still to be played. Shared by the Streamlit app and the HTTP API for the same
reason the settlement code is: two front ends running two copies of a Monte
Carlo will quote two different favourites, and a product that disagrees with
itself about who is winning the Golden Boot has no business publishing either
number.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd


def norm_player(name: str) -> str:
    """Join key for a player across providers.

    The prediction source names players the Understat way and the settlement
    source the ESPN way, and they disagree: "Kylian Mbappe-Lottin" against
    "Kylian Mbappé". Folding accents is not enough — matched on the full string
    he arrived with zero goals next to a 60% chance of finishing top scorer.
    First name plus the first part of the surname makes the two meet, and within
    one league it cannot collide with anyone else.
    """
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    parts = " ".join(s.lower().replace(".", " ").split()).split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} {parts[-1].split('-')[0]}"


def goals_so_far(actuals: pd.DataFrame, teams: set[str], since: str,
                 until: str) -> pd.DataFrame:
    """Goals scored this season per player, from the settled player-match data."""
    cols = ["key", "player", "team", "goals"]
    if actuals is None or actuals.empty:
        return pd.DataFrame(columns=cols)
    a = actuals[
        actuals["equipo"].isin(teams)
        & (actuals["fecha"] >= since)
        & (actuals["fecha"] <= until)
    ].copy()
    if a.empty:
        return pd.DataFrame(columns=cols)
    a["goals"] = pd.to_numeric(a["goals"], errors="coerce").fillna(0.0)
    a["key"] = a["player"].map(norm_player)
    return (
        a.groupby("key")
        .agg(player=("player", "first"), team=("equipo", "first"),
             goals=("goals", "sum"))
        .reset_index()
    )


def remaining_rates(remaining: pd.DataFrame, predict, players_for_lambda,
                    progress=None) -> pd.DataFrame:
    """Expected goals each player still has to come, over the fixtures left.

    Walks the real remaining calendar so a striker with eight games left is not
    compared with one who has two. The per-team model call is memoised on the
    rounded match lambda — most fixtures share one to two decimals, which turns
    roughly seven hundred model calls into a few dozen.
    """
    cols = ["key", "player", "team", "mu", "matches"]
    if remaining is None or remaining.empty:
        return pd.DataFrame(columns=cols)

    memo: dict[tuple[str, float], pd.DataFrame] = {}

    def team_players(team: str, lam: float) -> pd.DataFrame:
        k = (team, round(float(lam), 2))
        if k not in memo:
            try:
                memo[k] = players_for_lambda(team, k[1])
            except Exception:
                memo[k] = pd.DataFrame()
        return memo[k]

    acc: dict[str, dict] = {}
    total = max(len(remaining), 1)
    for done, r in enumerate(remaining.itertuples(), start=1):
        if progress is not None:
            progress(done, total, r.home_team, r.away_team)
        pred = predict(r.home_team, r.away_team)
        if pred is None:
            continue
        for team, lam in ((r.home_team, pred.lambda_home),
                          (r.away_team, pred.lambda_away)):
            pl = team_players(team, lam)
            if pl is None or pl.empty or "mu_goals" not in pl.columns:
                continue
            for q in pl.itertuples():
                k = norm_player(q.player)
                e = acc.setdefault(
                    k, {"player": q.player, "team": team, "mu": 0.0, "matches": 0}
                )
                e["mu"] += float(q.mu_goals)
                e["matches"] += 1

    if not acc:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([{"key": k, **v} for k, v in acc.items()])


def simulate(goals: pd.DataFrame, rates: pd.DataFrame, n_sims: int = 20_000,
             field: int = 80, seed: int = 20260905) -> pd.DataFrame:
    """P(finish top scorer) for every contender, plus goals now and expected."""
    if goals.empty and rates.empty:
        return pd.DataFrame()

    tbl = rates.merge(goals[["key", "goals"]], on="key", how="outer")
    tbl["mu"] = pd.to_numeric(tbl.get("mu"), errors="coerce").fillna(0.0)
    tbl["goals"] = pd.to_numeric(tbl.get("goals"), errors="coerce").fillna(0.0)

    names = dict(zip(goals["key"], goals["player"])) if not goals.empty else {}
    teams = dict(zip(goals["key"], goals["team"])) if not goals.empty else {}
    if "player" not in tbl.columns:
        tbl["player"] = tbl["key"]
    tbl["player"] = tbl["player"].fillna(tbl["key"].map(names)).fillna(tbl["key"])
    if "team" not in tbl.columns:
        tbl["team"] = None
    tbl["team"] = tbl["team"].fillna(tbl["key"].map(teams))
    tbl["expected"] = tbl["goals"] + tbl["mu"]

    # Only contenders go into the draw. A defender on two goals cannot win the
    # boot, and five hundred players across twenty thousand seasons is memory
    # spent on nobody.
    top = tbl.nlargest(min(field, len(tbl)), "expected").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    draws = rng.poisson(top["mu"].to_numpy()[None, :], size=(n_sims, len(top)))
    totals = draws + top["goals"].to_numpy()[None, :]
    # A tiny jitter breaks ties uniformly instead of always crediting whichever
    # player happens to sort first.
    winners = np.argmax(totals + rng.random(totals.shape) * 1e-6, axis=1)
    p_win = np.bincount(winners, minlength=len(top)) / n_sims

    out = pd.DataFrame({
        "player": top["player"],
        "team": top["team"].fillna(""),
        "goals": top["goals"].astype(int),
        "expected": top["expected"].round(1),
        "pTopScorer": p_win.round(4),
        "matchesLeft": top.get("matches", pd.Series(0, index=top.index))
        .fillna(0).astype(int),
    })
    return out.sort_values("pTopScorer", ascending=False).reset_index(drop=True)
