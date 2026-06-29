from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error


def rank_probability_score(pred_matrix: np.ndarray | pd.DataFrame, obs_matrix: np.ndarray | pd.DataFrame) -> float:
    """Rank Probability Score for ordered outcomes.

    For 1X2 we use the order [home_win, draw, away_win] as in the CLADAG lab.
    """
    P = np.asarray(pred_matrix, dtype=float)
    O = np.asarray(obs_matrix, dtype=float)
    if P.shape != O.shape:
        raise ValueError(f"Shape mismatch: {P.shape} vs {O.shape}")
    C = P.shape[1]
    return float(np.mean(np.sum((np.cumsum(P, axis=1) - np.cumsum(O, axis=1)) ** 2, axis=1) / (C - 1)))


def safe_log_loss(y_true, y_pred_proba, labels=None) -> float:
    """Log loss that preserves the supplied probability-column order.

    sklearn sorts string labels lexicographically, which is easy to misuse for
    football 1X2 matrices ordered as [H, D, A]. This helper maps labels by the
    provided order explicitly.
    """
    P = np.asarray(y_pred_proba, dtype=float)
    eps = 1e-15
    P = np.clip(P, eps, 1 - eps)
    P = P / P.sum(axis=1, keepdims=True)
    if labels is None:
        return float(log_loss(y_true, P))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx = np.array([label_to_idx[y] for y in y_true])
    return float(-np.mean(np.log(P[np.arange(len(idx)), idx])))


def brier_multiclass(y_true_onehot: np.ndarray, pred_proba: np.ndarray) -> float:
    return float(np.mean(np.sum((y_true_onehot - pred_proba) ** 2, axis=1)))


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def calibration_table(df: pd.DataFrame, prob_col: str, outcome_col: str, bins: int = 10) -> pd.DataFrame:
    tmp = df[[prob_col, outcome_col]].dropna().copy()
    tmp["bin"] = pd.cut(tmp[prob_col], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    return tmp.groupby("bin", observed=False).agg(
        n=(outcome_col, "size"),
        avg_pred=(prob_col, "mean"),
        observed_rate=(outcome_col, "mean"),
    ).reset_index()
