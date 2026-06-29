from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mundialytics.models.goal_model import DEFAULT_NUMERIC_FEATURES
from mundialytics.utils import clip_lambda


@dataclass
class TeamEventConfig:
    random_state: int = 42
    n_estimators: int = 400
    min_samples_leaf: int = 5
    lambda_floor: float = 0.01
    lambda_cap: float = 40.0


class TeamEventModel:
    """Predict expected team-level event counts: shots, SOT, corners, fouls, cards."""

    DEFAULT_TARGETS = ["shots_for", "sot_for", "corners_for", "fouls_for", "yellow_cards_for"]

    def __init__(self, config: TeamEventConfig | None = None, feature_cols: list[str] | None = None):
        self.config = config or TeamEventConfig()
        self.feature_cols = feature_cols or [c for c in DEFAULT_NUMERIC_FEATURES if c not in {"team_elo", "opponent_elo"}] + ["team_elo", "opponent_elo"]
        self.models: Dict[str, Pipeline] = {}

    def fit(self, frame: pd.DataFrame, targets: list[str] | None = None) -> "TeamEventModel":
        targets = targets or self.DEFAULT_TARGETS
        features = [c for c in self.feature_cols if c in frame.columns]
        for target in targets:
            if target not in frame.columns:
                continue
            train = frame.dropna(subset=[target]).copy()
            if len(train) < 5:
                continue
            pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", RandomForestRegressor(
                    n_estimators=self.config.n_estimators,
                    min_samples_leaf=self.config.min_samples_leaf,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )),
            ])
            pipe.fit(train[features], train[target].astype(float).clip(lower=0))
            self.models[target] = pipe
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[["team", "opponent"]].copy() if {"team", "opponent"}.issubset(frame.columns) else pd.DataFrame(index=frame.index)
        features = [c for c in self.feature_cols if c in frame.columns]
        for target, model in self.models.items():
            pred = model.predict(frame[features])
            out[f"expected_{target}"] = clip_lambda(pred, self.config.lambda_floor, self.config.lambda_cap)
        return out
