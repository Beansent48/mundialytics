from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1 / (1 + np.exp(-x))


def clip_lambda(values, floor: float = 0.05, cap: float = 6.0):
    return np.clip(values, floor, cap)


def normalize_probabilities(probs: Iterable[float]) -> np.ndarray:
    arr = np.array(list(probs), dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    total = arr.sum()
    if total <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / total


def decimal_to_implied_probability(odds: float) -> float:
    if odds <= 1:
        raise ValueError("Decimal odds must be > 1.0")
    return 1.0 / odds


def poisson_probability_at_least_one(lam: float) -> float:
    return 1 - math.exp(-max(lam, 0.0))


def atomic_to_csv(df: pd.DataFrame, path, **kwargs) -> None:
    """Write a CSV so readers only ever see the old file or the new one.

    Written beside the target, then swapped in with os.replace (atomic on the
    same volume, Windows included). A crash mid-write leaves the old file intact,
    and a running API reloading on the file's mtime can never read half of it.
    """
    import os
    from pathlib import Path

    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    kwargs.setdefault("index", False)
    df.to_csv(tmp, **kwargs)
    os.replace(tmp, path)
