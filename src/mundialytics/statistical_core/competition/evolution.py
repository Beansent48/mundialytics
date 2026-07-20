"""
Probability evolution across a season.

Runs the from-current-point forecast at a sequence of cutoffs and stitches the
per-team probabilities into a tidy timeline. This powers the "how do the odds move
as the season goes on?" view (a daily-return visitor's main hook) and is also the
cleanest behavioural check on the model: a good forecaster's probabilities should
sharpen toward the eventual outcome as matches are played.

Each cutoff re-trains the engine on data strictly before it (option A), so the
timeline is leakage-free at every point — the matchday-N row only ever knew what
was true at matchday N.
"""
from __future__ import annotations

import pandas as pd

from mundialytics.statistical_core.competition.cutoff import load_league_state_from_foundation
from mundialytics.statistical_core.competition.engine_provider import (
    fixture_lambdas,
    train_engine_before_cutoff,
)
from mundialytics.statistical_core.competition.resume_simulator import simulate_rest_of_season


def forecast_timeline(
    competition: str,
    season: str,
    matchdays: list[int],
    foundation: pd.DataFrame,
    n_sims: int = 5_000,
    top_places: tuple[int, ...] = (4,),
    relegation_places: int = 3,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Per-team forecast at each requested matchday.

    Returns a long DataFrame with one row per (matchday, team):
    matchday, team, p_champion, p_top{k}..., p_relegation, exp_points, exp_rank,
    plus current_points/current_rank at that cutoff and the team's actual final
    rank (from the completed season) for convergence checks.
    """
    # Actual final table (for convergence reference).
    full = load_league_state_from_foundation(competition, season, foundation=foundation)
    final_rank = dict(zip(full.standings["team"], full.standings["rank"]))

    frames = []
    for md in matchdays:
        state = load_league_state_from_foundation(
            competition, season, cutoff_matchday=md, foundation=foundation
        )
        if state.is_complete:
            continue
        kwargs = {} if lookback_days is None else {"lookback_days": lookback_days}
        engine = train_engine_before_cutoff(state, foundation, **kwargs)
        lam = fixture_lambdas(engine, state)
        fc = simulate_rest_of_season(
            lam, state, n_sims=n_sims, top_places=top_places, relegation_places=relegation_places
        )
        cur_rank = dict(zip(state.standings["team"], state.standings["rank"]))
        cur_pts = dict(zip(state.standings["team"], state.standings["points"]))
        tp = fc.team_probs.copy()
        tp.insert(0, "matchday", md)
        tp["current_points"] = tp["team"].map(cur_pts)
        tp["current_rank"] = tp["team"].map(cur_rank)
        tp["actual_final_rank"] = tp["team"].map(final_rank)
        frames.append(tp)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
