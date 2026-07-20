"""
Analytic expected points (xPoints) over the remaining fixtures.

For each remaining fixture we turn the goal lambdas into 1X2 probabilities (using
the same Dixon-Coles correction the engine uses) and credit each team its expected
points: 3*P(win) + 1*P(draw). Summed over a team's remaining games and added to the
points it already has, this gives a projected final table — instantly, with no
Monte Carlo. It answers "where does the table land if every game plays to
expectation?", and is the fast companion to the simulator's probability spread.
"""
from __future__ import annotations

import pandas as pd

from mundialytics.statistical_core.distributions import outcome_probabilities
from mundialytics.statistical_core.schemas import canonical_name

WIN_POINTS = 3
DRAW_POINTS = 1


def expected_points_table(
    lambdas: pd.DataFrame,
    state,
    dixon_coles_rho: float = -0.07,
    max_goals: int = 10,
) -> pd.DataFrame:
    """Projected final table from current points + expected points remaining.

    Parameters
    ----------
    lambdas
        Output of ``fixture_lambdas`` (home_team, away_team, lambda_home,
        lambda_away per remaining fixture).
    state
        The LeagueState (for current standings and the full team list).
    dixon_coles_rho, max_goals
        Match the engine's scoreline settings so xPoints is consistent with the
        simulator's draws.

    Returns
    -------
    DataFrame sorted by projected points, with columns: team, current_points,
    matches_remaining, xpts_remaining, projected_points, current_gd,
    projected_gd, projected_rank.
    """
    standings = state.standings
    current_points = dict(zip(standings["team"], standings["points"]))
    current_gd = dict(zip(standings["team"], standings["goal_diff"]))

    # Canonicalise so keys match standings / lambdas (don't assume caller did).
    teams = [canonical_name(t) for t in state.teams]
    xpts = {t: 0.0 for t in teams}
    xgd = {t: 0.0 for t in teams}
    n_rem = {t: 0 for t in teams}

    for r in lambdas.itertuples(index=False):
        h, a = canonical_name(r.home_team), canonical_name(r.away_team)
        probs = outcome_probabilities(
            r.lambda_home, r.lambda_away, max_goals=max_goals, dixon_coles_rho=dixon_coles_rho
        )
        p_h, p_d, p_a = probs["p_home_win"], probs["p_draw"], probs["p_away_win"]
        xpts.setdefault(h, 0.0); xpts.setdefault(a, 0.0)
        xgd.setdefault(h, 0.0); xgd.setdefault(a, 0.0)
        n_rem.setdefault(h, 0); n_rem.setdefault(a, 0)
        xpts[h] += WIN_POINTS * p_h + DRAW_POINTS * p_d
        xpts[a] += WIN_POINTS * p_a + DRAW_POINTS * p_d
        # Expected goal difference contribution (mean of Poisson diff).
        xgd[h] += r.lambda_home - r.lambda_away
        xgd[a] += r.lambda_away - r.lambda_home
        n_rem[h] += 1
        n_rem[a] += 1

    rows = []
    for t in teams:
        cp = current_points.get(t, 0)
        cgd = current_gd.get(t, 0)
        rows.append({
            "team": t,
            "current_points": cp,
            "matches_remaining": n_rem.get(t, 0),
            "xpts_remaining": round(xpts.get(t, 0.0), 2),
            "projected_points": round(cp + xpts.get(t, 0.0), 2),
            "current_gd": cgd,
            "projected_gd": round(cgd + xgd.get(t, 0.0), 2),
        })
    table = pd.DataFrame(rows).sort_values(
        ["projected_points", "projected_gd"], ascending=False
    ).reset_index(drop=True)
    table["projected_rank"] = table.index + 1
    return table
