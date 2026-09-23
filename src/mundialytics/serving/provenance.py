"""What produced a number: the code, the parameters and the data behind a fit.

One fingerprint shared by the API and the pre-kickoff logger, so a logged
prediction can be tied to the exact engine that made it, and a page can say
whether the model it shows is the one that was logged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

# The sources a fitted club engine depends on. Change one and every fit (and
# every fingerprint) built on the old one stops matching.
ENGINE_SOURCES = (
    "src/mundialytics/statistical_core/prediction_engine.py",
    "src/mundialytics/statistical_core/attack_defense_model.py",
    "src/mundialytics/statistical_core/distributions.py",
    "src/mundialytics/models/goal_model.py",
    "src/mundialytics/models/xg_rate_model.py",
    "src/mundialytics/ratings/elo.py",
)


def code_fingerprint(*rel_paths: str, root: Path = ROOT) -> str:
    """8-char hash of the given source files (missing files hash as such)."""
    h = hashlib.sha256()
    for rel in sorted(rel_paths):
        try:
            h.update((root / rel).read_bytes())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()[:8]


def engine_code_fingerprint(root: Path = ROOT) -> str:
    return code_fingerprint(*ENGINE_SOURCES, root=root)


def train_cutoff(df: pd.DataFrame) -> str:
    """The newest match a fit could have seen, as YYYY-MM-DD."""
    return str(pd.to_datetime(df["date"], errors="coerce").max())[:10]


def model_fingerprint(df: pd.DataFrame, kwargs: dict, root: Path = ROOT) -> str:
    """Code + parameters + training data, in 12 characters.

    Two predictions with the same fingerprint came from the same fitted model.
    """
    payload = json.dumps({
        "code": engine_code_fingerprint(root),
        "kwargs": kwargs,
        "rows": int(len(df)),
        "cutoff": train_cutoff(df),
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
