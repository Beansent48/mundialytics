"""
Resume Monte Carlo — simulate a league to the end from the current point.

Unlike PredictionEngine.simulate_league (which plays a full season from scratch and
is kept for SquadLab / pre-season), this starts from the REAL current standings and
only samples the remaining fixtures. That is what makes "who wins the league from
matchday 25?" meaningful: the points already earned are locked in, only the future
is random.

Everything is vectorised across both fixtures and simulations (Poisson draws +
incidence-matrix accumulation), so a full 20-team run is fast. Ranking uses the
points -> goal-difference -> goals-for order (head-to-head tiebreaks, which La Liga
and Serie A use, are omitted inside the sim for speed — a documented simplification
that barely moves probabilities).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mundialytics.statistical_core.schemas import canonical_name


@dataclass
class LeagueForecast:
    n_sims: int
    team_probs: pd.DataFrame          # team, p_champion, p_top2, p_top4, p_relegation, exp_points, exp_rank
    position_matrix: pd.DataFrame     # rows=team, cols=final position 1..N, values=probability
    metadata: dict = field(default_factory=dict)

    def summary(self, top: int = 6) -> str:
        head = self.team_probs.head(top)
        lines = [f"League forecast ({self.n_sims:,} sims):"]
        for _, r in head.iterrows():
            lines.append(
                f"  {r['team']:<16} champ:{r['p_champion']:5.1%}  top4:{r['p_top4']:5.1%}  "
                f"xPts:{r['exp_points']:5.1f}"
            )
        return "\n".join(lines)


def simulate_rest_of_season(
    lambdas: pd.DataFrame,
    state,
    n_sims: int = 10_000,
    relegation_places: int = 3,
    top_places: tuple[int, ...] = (2, 4),
    random_seed: int = 42,
) -> LeagueForecast:
    """Monte Carlo the remaining fixtures, resuming from current standings.

    Parameters
    ----------
    lambdas
        ``fixture_lambdas`` output: remaining fixtures with lambda_home/away.
    state
        LeagueState — supplies the team list and the current points/goals anchor.
    n_sims
        Monte Carlo iterations.
    relegation_places
        Number of bottom places counted as relegation (3 in most leagues).
    top_places
        Extra "top-N" cutoffs to report probabilities for (e.g. top2, top4).

    Returns
    -------
    LeagueForecast with per-team probabilities and the full position matrix.
    """
    # Canonicalise team labels so lookups match compute_standings / canonical_name
    # (the loader already canonicalises, but don't assume the caller did).
    teams = [canonical_name(t) for t in state.teams]
    n_teams = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    rng = np.random.default_rng(random_seed)

    standings = state.standings
    start_points = np.zeros(n_teams)
    start_gf = np.zeros(n_teams)
    start_ga = np.zeros(n_teams)
    for r in standings.itertuples(index=False):
        i = idx.get(r.team)
        if i is None:
            continue
        start_points[i] = r.points
        start_gf[i] = r.goals_for
        start_ga[i] = r.goals_against

    n_fix = len(lambdas)
    if n_fix == 0:
        # Season already complete — the current table is the outcome.
        final_points = start_points
        final_gd = start_gf - start_ga
        order = np.lexsort((start_gf, final_gd, final_points))[::-1]
        ranks = np.empty(n_teams, int)
        ranks[order] = np.arange(n_teams)
        pos_matrix = np.zeros((n_teams, n_teams))
        for t in range(n_teams):
            pos_matrix[t, ranks[t]] = 1.0
        return _assemble(teams, pos_matrix, final_points[None, :],
                         n_sims=1, relegation_places=relegation_places,
                         top_places=top_places, state=state)

    home_idx = np.array([idx[canonical_name(h)] for h in lambdas["home_team"]], dtype=int)
    away_idx = np.array([idx[canonical_name(a)] for a in lambdas["away_team"]], dtype=int)
    lam_h = lambdas["lambda_home"].to_numpy(dtype=float)
    lam_a = lambdas["lambda_away"].to_numpy(dtype=float)

    # One-hot incidence matrices: (n_fix, n_teams).
    H = np.zeros((n_fix, n_teams)); H[np.arange(n_fix), home_idx] = 1.0
    A = np.zeros((n_fix, n_teams)); A[np.arange(n_fix), away_idx] = 1.0

    # Draw all sims at once: (n_sims, n_fix).
    hg = rng.poisson(lam_h, size=(n_sims, n_fix))
    ag = rng.poisson(lam_a, size=(n_sims, n_fix))

    home_pts = np.where(hg > ag, 3, np.where(hg == ag, 1, 0)).astype(float)
    away_pts = np.where(ag > hg, 3, np.where(hg == ag, 1, 0)).astype(float)

    # Accumulate to teams via incidence matrices.
    pts = home_pts @ H + away_pts @ A + start_points          # (n_sims, n_teams)
    gf = hg @ H + ag @ A + start_gf
    ga = ag @ H + hg @ A + start_ga
    gd = gf - ga

    # Composite score for ranking: points dominate, then GD, then GF.
    # Offsets keep it strictly ordered and non-negative for the integer packing.
    score = pts * 1_000_000.0 + (gd + 10_000.0) * 100.0 + gf
    order = np.argsort(-score, axis=1)                        # positions -> team idx
    ranks = np.empty((n_sims, n_teams), dtype=int)
    np.put_along_axis(ranks, order, np.tile(np.arange(n_teams), (n_sims, 1)), axis=1)

    # Position distribution: pos_matrix[team, rank] = probability.
    pos_matrix = np.zeros((n_teams, n_teams))
    for r in range(n_teams):
        pos_matrix[:, r] = (ranks == r).sum(axis=0) / n_sims

    return _assemble(teams, pos_matrix, pts, n_sims, relegation_places, top_places, state)


def _assemble(teams, pos_matrix, pts, n_sims, relegation_places, top_places, state):
    n_teams = len(teams)
    exp_points = pts.mean(axis=0)
    positions = np.arange(1, n_teams + 1)
    exp_rank = (pos_matrix * positions).sum(axis=1)

    p_champion = pos_matrix[:, 0]
    p_releg = pos_matrix[:, n_teams - relegation_places:].sum(axis=1)
    top_cols = {f"p_top{k}": pos_matrix[:, :k].sum(axis=1) for k in top_places}

    team_probs = pd.DataFrame({
        "team": teams,
        "p_champion": p_champion,
        **top_cols,
        "p_relegation": p_releg,
        "exp_points": np.round(exp_points, 1),
        "exp_rank": np.round(exp_rank, 2),
    }).sort_values("exp_points", ascending=False).reset_index(drop=True)

    pos_df = pd.DataFrame(pos_matrix, index=teams, columns=positions)
    pos_df = pos_df.loc[team_probs["team"]]

    return LeagueForecast(
        n_sims=n_sims,
        team_probs=team_probs,
        position_matrix=pos_df,
        metadata={
            "competition": state.competition,
            "season": state.season,
            "cutoff": str(state.cutoff_date.date()) if state.cutoff_date is not None else "complete",
            "n_played": state.n_played,
            "n_remaining": state.n_remaining,
            "relegation_places": relegation_places,
        },
    )
