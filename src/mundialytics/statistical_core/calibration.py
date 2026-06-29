from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


COUNT_CAPS = {
    "goals": (0.05, 6.0),
    "shots": (0.2, 30.0),
    "shots_on_target": (0.05, 15.0),
    "fouls": (0.2, 35.0),
    "yellow_cards": (0.01, 8.0),
    "corners": (0.0, 18.0),
}

MARKET_PROB_CAPS = {
    "player_shots": (0.02, 0.95),
    "player_shots_on_target": (0.01, 0.75),
    "player_fouls_committed": (0.02, 0.95),
    "player_yellow_card": (0.005, 0.45),
}


def safe_count(value: float, market: str, default: float = 0.0) -> float:
    floor, cap = COUNT_CAPS.get(market, (0.0, 50.0))
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = default
    if not math.isfinite(v):
        v = default
    return float(min(max(v, floor), cap))


def safe_probability(value: float, market: str | None = None, sample_size: float | None = None) -> tuple[float, list[str]]:
    warnings: list[str] = []
    lo, hi = MARKET_PROB_CAPS.get(str(market), (0.001, 0.999))
    try:
        p = float(value)
    except (TypeError, ValueError):
        p = lo
        warnings.append("invalid_probability_defaulted")
    if not math.isfinite(p):
        p = lo
        warnings.append("non_finite_probability_defaulted")
    cap = hi
    if sample_size is not None:
        try:
            s = float(sample_size)
        except (TypeError, ValueError):
            s = 0.0
        if s < 90:
            cap = min(cap, 0.50)
            warnings.append("very_low_sample_cap_applied")
        elif s < 270:
            cap = min(cap, 0.65)
            warnings.append("low_sample_cap_applied")
    clipped = float(min(max(p, lo), cap))
    if abs(clipped - p) > 1e-9:
        warnings.append(f"probability_capped_from_{p:.3f}_to_{clipped:.3f}")
    return clipped, warnings


def recency_weights(dates: pd.Series, half_life_days: float = 365.0) -> np.ndarray:
    parsed = pd.to_datetime(dates, errors="coerce")
    if parsed.notna().sum() == 0:
        return np.ones(len(dates), dtype=float)
    max_date = parsed.max()
    age = (max_date - parsed).dt.days.fillna(half_life_days).clip(lower=0).astype(float)
    weights = np.power(0.5, age / max(half_life_days, 1.0)).to_numpy(dtype=float)
    return weights / np.nanmean(weights)


class IsotonicCalibrator:
    """Post-hoc probability calibration via isotonic regression.

    Fit on historical predictions vs outcomes, then apply to new predictions to
    correct systematic over/under-confidence. Preserves monotonicity (higher
    raw prob → higher calibrated prob).

    Usage:
        cal = IsotonicCalibrator()
        cal.fit(pred_probs, actual_binary_outcomes)  # 1 = event occurred
        calibrated = cal.transform(new_probs)
    """

    def __init__(self, out_of_bounds: str = "clip"):
        self._iso = IsotonicRegression(out_of_bounds=out_of_bounds)
        self._fitted = False

    def fit(self, predicted: Sequence[float], actual: Sequence[float]) -> "IsotonicCalibrator":
        p = np.asarray(predicted, dtype=float)
        y = np.asarray(actual, dtype=float)
        mask = np.isfinite(p) & np.isfinite(y)
        if mask.sum() < 5:
            return self
        self._iso.fit(p[mask], y[mask])
        self._fitted = True
        return self

    def transform(self, predicted: Sequence[float]) -> np.ndarray:
        p = np.asarray(predicted, dtype=float)
        if not self._fitted:
            return np.clip(p, 0.0, 1.0)
        return np.clip(self._iso.predict(p), 0.0, 1.0)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class MatchProbabilityCalibrator:
    """Calibrates 1X2 match outcome probabilities.

    Fits one isotonic calibrator per outcome (home/draw/away), applies them
    independently, then renormalizes to enforce sum-to-one. Designed to correct
    the systematic biases of an independent-Poisson model (draws typically
    under-priced, away wins sometimes over-priced).

    Usage:
        cal = MatchProbabilityCalibrator()
        cal.fit(predictions_df, outcomes_series)
        calibrated_df = cal.transform(predictions_df)

    ``predictions_df`` must have columns ``p_home_win``, ``p_draw``, ``p_away_win``.
    ``outcomes_series`` must be string with values "H", "D", "A".
    """

    OUTCOMES = ["p_home_win", "p_draw", "p_away_win"]
    OUTCOME_MAP = {"H": "p_home_win", "D": "p_draw", "A": "p_away_win"}

    def __init__(self):
        self._calibrators: dict[str, IsotonicCalibrator] = {k: IsotonicCalibrator() for k in self.OUTCOMES}
        self._fitted = False

    def fit(self, predictions: pd.DataFrame, outcomes: pd.Series) -> "MatchProbabilityCalibrator":
        """
        Parameters
        ----------
        predictions : DataFrame with columns p_home_win, p_draw, p_away_win
        outcomes    : Series of "H", "D", "A" strings, same index as predictions
        """
        df = predictions[self.OUTCOMES].copy()
        for col, event in [("p_home_win", "H"), ("p_draw", "D"), ("p_away_win", "A")]:
            binary = (outcomes == event).astype(float)
            self._calibrators[col].fit(df[col].values, binary.values)
        self._fitted = True
        return self

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        out = predictions.copy()
        for col in self.OUTCOMES:
            raw = out[col].values
            out[col] = self._calibrators[col].transform(raw)
        total = out[self.OUTCOMES].sum(axis=1).replace(0, 1.0)
        for col in self.OUTCOMES:
            out[col] = out[col] / total
        return out

    @property
    def is_fitted(self) -> bool:
        return self._fitted


def weighted_mean(values: pd.Series, weights: np.ndarray | None = None, default: float = 0.0) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v)
    if not mask.any():
        return float(default)
    if weights is None:
        return float(np.nanmean(v[mask]))
    w = np.asarray(weights, dtype=float)
    w = w[mask]
    v = v[mask]
    total = np.nansum(w)
    if total <= 0 or not np.isfinite(total):
        return float(np.nanmean(v))
    return float(np.nansum(v * w) / total)
