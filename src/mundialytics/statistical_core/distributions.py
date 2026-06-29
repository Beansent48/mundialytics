from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import poisson


def clip_expected(value: float | int | None, floor: float = 0.001, cap: float = 12.0) -> float:
    """Clip count intensities to operationally safe ranges."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = floor
    if not math.isfinite(v):
        v = floor
    return float(min(max(v, floor), cap))


def poisson_prob_at_least(lam: float, threshold: int) -> float:
    lam = clip_expected(lam, floor=0.0, cap=100.0)
    threshold = int(max(threshold, 0))
    if threshold <= 0:
        return 1.0
    return float(1.0 - poisson.cdf(threshold - 1, lam))


def poisson_prob_over(lam: float, line: float | str) -> float:
    """Return P(X > line) for a Poisson count."""
    lam = clip_expected(lam, floor=0.0, cap=100.0)
    line_f = parse_numeric_line(line, default=0.5)
    return float(1.0 - poisson.cdf(math.floor(line_f), lam))


def poisson_prob_under(lam: float, line: float | str) -> float:
    lam = clip_expected(lam, floor=0.0, cap=100.0)
    line_f = parse_numeric_line(line, default=0.5)
    return float(poisson.cdf(math.floor(line_f), lam))


def probability_for_count_line(lam: float, line: str | float | int, selection: str = "over") -> float:
    """Parse common prop lines like '1+', '2+', 0.5, 1.5 and return a probability."""
    selection_norm = str(selection or "over").strip().lower()
    text = str(line).strip().lower()
    if text.endswith("+"):
        threshold = int(float(text[:-1]))
        prob_over = poisson_prob_at_least(lam, threshold)
    else:
        prob_over = poisson_prob_over(lam, parse_numeric_line(text, default=0.5))
    if selection_norm in {"under", "u", "no", "not"}:
        return float(1.0 - prob_over)
    return float(prob_over)


def parse_numeric_line(value: str | float | int | None, default: float = 0.5) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().replace(",", ".")
    if text in {"", "nan", "none"}:
        return float(default)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return float(default)
    return float(match.group(0))


@dataclass(frozen=True)
class ScoreDistribution:
    lambda_home: float
    lambda_away: float
    matrix: pd.DataFrame

    @property
    def p_home_win(self) -> float:
        values = self.matrix.to_numpy(dtype=float)
        return float(np.tril(values, k=-1).sum())

    @property
    def p_draw(self) -> float:
        return float(np.trace(self.matrix.to_numpy(dtype=float)))

    @property
    def p_away_win(self) -> float:
        values = self.matrix.to_numpy(dtype=float)
        return float(np.triu(values, k=1).sum())

    @property
    def p_btts(self) -> float:
        idx = self.matrix.index.to_numpy(dtype=int)
        cols = self.matrix.columns.to_numpy(dtype=int)
        values = self.matrix.to_numpy(dtype=float)
        mask = (idx[:, None] > 0) & (cols[None, :] > 0)
        return float(values[mask].sum())

    def total_goals_probability(self, line: float | str, side: str = "over") -> float:
        line_f = parse_numeric_line(line, default=2.5)
        values = self.matrix.to_numpy(dtype=float)
        home = self.matrix.index.to_numpy(dtype=int)[:, None]
        away = self.matrix.columns.to_numpy(dtype=int)[None, :]
        total = home + away
        if str(side).strip().lower().startswith("u"):
            return float(values[total < line_f].sum())
        return float(values[total > line_f].sum())

    def top_scorelines(self, n: int = 5) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for h in self.matrix.index:
            for a in self.matrix.columns:
                rows.append({"score": f"{h}-{a}", "probability": float(self.matrix.loc[h, a])})
        return sorted(rows, key=lambda x: float(x["probability"]), reverse=True)[:n]

    def to_long_frame(self, match_id: str | None = None) -> pd.DataFrame:
        rows = []
        for h in self.matrix.index:
            for a in self.matrix.columns:
                row = {"home_goals": int(h), "away_goals": int(a), "score": f"{h}-{a}", "probability": float(self.matrix.loc[h, a])}
                if match_id is not None:
                    row["match_id"] = match_id
                rows.append(row)
        return pd.DataFrame(rows)


def _dixon_coles_tau(home_goals: int, away_goals: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """Low-score Dixon-Coles adjustment.

    Negative rho boosts 0-0 / 1-1 and dampens 1-0 / 0-1. This is useful when
    an independent Poisson profile under-prices draws. The return value is
    clipped defensively so the probability matrix stays valid for model-lab
    experiments.
    """
    r = float(rho or 0.0)
    if abs(r) < 1e-12:
        return 1.0
    if home_goals == 0 and away_goals == 0:
        tau = 1.0 - r * lambda_home * lambda_away
    elif home_goals == 0 and away_goals == 1:
        tau = 1.0 + r * lambda_home
    elif home_goals == 1 and away_goals == 0:
        tau = 1.0 + r * lambda_away
    elif home_goals == 1 and away_goals == 1:
        tau = 1.0 - r
    else:
        tau = 1.0
    return float(np.clip(tau, 0.05, 3.0))


def scoreline_distribution(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 10,
    normalize: bool = True,
    dixon_coles_rho: float = 0.0,
) -> ScoreDistribution:
    """Poisson scoreline distribution with optional Dixon-Coles low-score correction.

    With ``dixon_coles_rho=0`` this is the original independent Poisson model.
    Non-zero rho only changes the 0-0, 1-0, 0-1 and 1-1 cells, then the finite
    matrix is re-normalized for auditable probabilities.
    """
    lh = clip_expected(lambda_home, floor=0.01, cap=8.0)
    la = clip_expected(lambda_away, floor=0.01, cap=8.0)
    goals = np.arange(max_goals + 1)
    values = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))
    rho = float(dixon_coles_rho or 0.0)
    if abs(rho) > 1e-12:
        for i, h in enumerate(goals[:2]):
            for j, a in enumerate(goals[:2]):
                values[i, j] *= _dixon_coles_tau(int(h), int(a), lh, la, rho)
    if normalize:
        total = values.sum()
        if total > 0:
            values = values / total
    matrix = pd.DataFrame(values, index=goals, columns=goals)
    return ScoreDistribution(lambda_home=lh, lambda_away=la, matrix=matrix)


def outcome_probabilities(lambda_home: float, lambda_away: float, max_goals: int = 10, dixon_coles_rho: float = 0.0) -> dict[str, float | str]:
    dist = scoreline_distribution(lambda_home, lambda_away, max_goals=max_goals, normalize=True, dixon_coles_rho=dixon_coles_rho)
    top = dist.top_scorelines(1)[0]
    return {
        "p_home_win": dist.p_home_win,
        "p_draw": dist.p_draw,
        "p_away_win": dist.p_away_win,
        "p_btts": dist.p_btts,
        "p_over_05": dist.total_goals_probability(0.5, "over"),
        "p_over_15": dist.total_goals_probability(1.5, "over"),
        "p_over_25": dist.total_goals_probability(2.5, "over"),
        "p_over_35": dist.total_goals_probability(3.5, "over"),
        "p_under_25": dist.total_goals_probability(2.5, "under"),
        "most_likely_score": str(top["score"]),
        "most_likely_score_probability": float(top["probability"]),
    }


def softmax(values: Iterable[float], temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return arr
    temp = max(float(temperature), 1e-9)
    arr = arr / temp
    arr = arr - np.nanmax(arr)
    exp = np.exp(arr)
    total = exp.sum()
    if not math.isfinite(total) or total <= 0:
        return np.ones(len(arr), dtype=float) / len(arr)
    return exp / total
