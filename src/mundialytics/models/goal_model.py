from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mundialytics.utils import clip_lambda


DEFAULT_NUMERIC_FEATURES = [
    "team_elo", "opponent_elo", "elo_diff", "external_team_elo", "external_opponent_elo", "external_elo_diff",
    "team_match_count_pre", "is_home_non_neutral",
    "goals_for_last5", "goals_against_last5", "shots_for_last5", "shots_against_last5",
    "sot_for_last5", "sot_against_last5", "corners_for_last5", "corners_against_last5",
    "fouls_for_last5", "fouls_against_last5", "yellow_cards_for_last5", "yellow_cards_against_last5",
    "goal_diff_last5", "shot_diff_last5",
]
DEFAULT_CATEGORICAL_FEATURES = ["competition", "stage"]


@dataclass
class GoalModelConfig:
    model_type: str = "poisson"  # poisson or random_forest_lambda
    poisson_alpha: float = 0.1
    rf_n_estimators: int = 500
    rf_min_samples_leaf: int = 6
    random_state: int = 42
    lambda_floor: float = 0.05
    lambda_cap: float = 6.0
    time_decay_half_life_days: float | None = 365.0


class GoalLambdaModel:
    """Predict expected goals (lambda) for each team-row.

    `model_type='poisson'` is closest to the CLADAG GLM idea.
    `model_type='random_forest_lambda'` follows the slides' idea: RF predicts
    expected goals, then the prediction is used as Poisson lambda.
    """

    def __init__(
        self,
        config: GoalModelConfig | None = None,
        numeric_features: Sequence[str] | None = None,
        categorical_features: Sequence[str] | None = None,
    ):
        self.config = config or GoalModelConfig()
        self.numeric_features = list(numeric_features or DEFAULT_NUMERIC_FEATURES)
        self.categorical_features = list(categorical_features or DEFAULT_CATEGORICAL_FEATURES)
        self.pipeline: Pipeline | None = None

    def _available_features(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        # Keep only columns that can actually be learnt from. Some sources
        # (for example international_results) do not include event statistics
        # such as shots/corners/cards, so rolling event features can be 100%
        # missing. Passing all-null columns into SimpleImputer creates noisy
        # warnings and, more importantly, suggests the model is using signals it
        # does not really have. We drop them at fit time.
        nums = []
        for c in self.numeric_features:
            if c in X.columns and pd.to_numeric(X[c], errors="coerce").notna().any():
                nums.append(c)
        cats = []
        for c in self.categorical_features:
            if c in X.columns and X[c].notna().any():
                cats.append(c)
        return nums, cats

    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        nums, cats = self._available_features(X)
        preprocess = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cats),
            ],
            remainder="drop",
        )
        if self.config.model_type == "poisson":
            reg = PoissonRegressor(alpha=self.config.poisson_alpha, max_iter=1000)
        elif self.config.model_type == "random_forest_lambda":
            reg = RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators,
                min_samples_leaf=self.config.rf_min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")
        return Pipeline([("preprocess", preprocess), ("model", reg)])

    def _time_decay_sample_weight(self, train: pd.DataFrame) -> np.ndarray | None:
        """Return leakage-safe recency weights for historical team rows.

        Recent rows matter more for current team strength, but all weights are
        computed only from dates already present in the training window.
        """
        half_life = self.config.time_decay_half_life_days
        if half_life is None or half_life <= 0 or "date" not in train.columns:
            return None
        dates = pd.to_datetime(train["date"], errors="coerce")
        if dates.notna().sum() == 0:
            return None
        max_date = dates.max()
        age_days = (max_date - dates).dt.days.clip(lower=0).fillna(0).astype(float)
        weights = np.power(0.5, age_days / float(half_life))
        return np.asarray(weights, dtype=float)

    def fit(self, frame: pd.DataFrame, target: str = "goals_for") -> "GoalLambdaModel":
        train = frame.dropna(subset=[target]).copy()
        X = train.drop(columns=[target], errors="ignore")
        y = train[target].astype(float).clip(lower=0)
        self.pipeline = self._build_pipeline(X)
        sample_weight = self._time_decay_sample_weight(train)
        if sample_weight is not None:
            self.pipeline.fit(X, y, model__sample_weight=sample_weight)
        else:
            self.pipeline.fit(X, y)
        return self

    def predict_lambda(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("GoalLambdaModel must be fitted before prediction.")
        raw = self.pipeline.predict(frame)
        return clip_lambda(raw, self.config.lambda_floor, self.config.lambda_cap)
