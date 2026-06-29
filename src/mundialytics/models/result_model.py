from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam

from mundialytics.utils import clip_lambda


@dataclass
class MatchProbabilityResult:
    lambda_home: float
    lambda_away: float
    p_home_win: float
    p_draw: float
    p_away_win: float
    p_over_25: float
    p_btts: float
    most_likely_score: str
    score_matrix: pd.DataFrame


def score_probability_matrix(lambda_home: float, lambda_away: float, max_goals: int = 8) -> pd.DataFrame:
    lh, la = clip_lambda([lambda_home, lambda_away])
    home_goals = np.arange(max_goals + 1)
    away_goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(home_goals, lh), poisson.pmf(away_goals, la))
    # We do not renormalize by default: the omitted tail probability is informative.
    return pd.DataFrame(matrix, index=home_goals, columns=away_goals)


def match_probabilities(lambda_home: float, lambda_away: float, max_goals: int = 8) -> MatchProbabilityResult:
    lh, la = clip_lambda([lambda_home, lambda_away])
    mat = score_probability_matrix(lh, la, max_goals=max_goals)
    p_home_win = float(1 - skellam.cdf(0, lh, la))
    p_draw = float(skellam.pmf(0, lh, la))
    p_away_win = float(skellam.cdf(-1, lh, la))
    p_over_25 = float(1 - poisson.cdf(2, lh + la))
    p_btts = float(1 - poisson.pmf(0, lh) - poisson.pmf(0, la) + poisson.pmf(0, lh) * poisson.pmf(0, la))
    max_idx = np.unravel_index(np.asarray(mat).argmax(), mat.shape)
    most_likely = f"{mat.index[max_idx[0]]}-{mat.columns[max_idx[1]]}"
    return MatchProbabilityResult(lh, la, p_home_win, p_draw, p_away_win, p_over_25, p_btts, most_likely, mat)


def summarize_score_matrix(matrix: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    rows = []
    for h in matrix.index:
        for a in matrix.columns:
            rows.append({"score": f"{h}-{a}", "probability": matrix.loc[h, a]})
    return pd.DataFrame(rows).sort_values("probability", ascending=False).head(top_n).reset_index(drop=True)
