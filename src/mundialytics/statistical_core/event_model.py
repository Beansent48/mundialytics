"""
Poisson regression models for team-level count events beyond goals.

One ``EventLambdaModel`` instance per market (shots, SOT, corners, fouls,
yellow_cards).  Each market gets a tailored feature set so that the predictors
are actually informative for that event type.

Architecture:
    build_event_training_frame(team_rows, elo_history)   <- feature engineering
    EventLambdaModel(market).fit(frame)                  <- per-market Poisson
    EventLambdaModel.predict_lambda(frame)               <- lambda for test row

The ``MarketPredictor`` bundles all fitted models and produces a full slate of
predictions for a fixture (expected counts + over/under probabilities).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mundialytics.utils import clip_lambda


# ── Market feature templates ───────────────────────────────────────────────────

MARKET_FEATURES: dict[str, list[str]] = {
    "shots_for": [
        "shots_for_last3", "shots_for_last5", "shots_for_last10",
        "shots_against_last3", "shots_against_last5",
        "goals_for_last5", "sot_for_last5",
        "elo_diff", "team_elo", "opponent_elo",
        "is_home_non_neutral", "team_match_count_pre",
    ],
    "sot_for": [
        "sot_for_last3", "sot_for_last5", "sot_for_last10",
        "shots_for_last5", "sot_against_last5",
        "goals_for_last5", "elo_diff", "team_elo",
        "is_home_non_neutral", "team_match_count_pre",
    ],
    "corners_for": [
        "corners_for_last3", "corners_for_last5", "corners_for_last10",
        "corners_against_last5",
        "shots_for_last5", "sot_for_last5",
        "elo_diff", "is_home_non_neutral", "team_match_count_pre",
    ],
    "fouls_for": [
        "fouls_for_last3", "fouls_for_last5", "fouls_for_last10",
        "fouls_against_last5",
        "yellow_cards_for_last5",
        "elo_diff", "opponent_elo",
        "is_home_non_neutral", "team_match_count_pre",
    ],
    "yellow_cards_for": [
        "yellow_cards_for_last3", "yellow_cards_for_last5", "yellow_cards_for_last10",
        "fouls_for_last5",
        "elo_diff", "opponent_elo",
        "is_home_non_neutral", "team_match_count_pre",
    ],
}

MARKET_CATEGORICAL: dict[str, list[str]] = {
    k: ["competition", "stage"] for k in MARKET_FEATURES
}

MARKET_BOUNDS: dict[str, tuple[float, float]] = {
    "shots_for":        (1.0, 35.0),
    "sot_for":          (0.3, 15.0),
    "corners_for":      (0.5, 18.0),
    "fouls_for":        (2.0, 30.0),
    "yellow_cards_for": (0.05, 8.0),
}


# ── Model ──────────────────────────────────────────────────────────────────────

class EventLambdaModel:
    """Poisson GLM for a single team-level count event (shots, corners, etc.).

    Usage mirrors GoalLambdaModel:
        model = EventLambdaModel(market="shots_for")
        model.fit(train_frame)
        lambdas = model.predict_lambda(test_frame)
        prob_over = poisson_prob_over(lambdas, line=12.5)
    """

    def __init__(
        self,
        market: str,
        model_type: str = "poisson",
        numeric_features: Sequence[str] | None = None,
        categorical_features: Sequence[str] | None = None,
        alpha: float = 0.1,
        time_decay_half_life_days: float | None = None,
    ):
        if market not in MARKET_FEATURES and numeric_features is None:
            raise ValueError(f"Unknown market '{market}'. Known: {list(MARKET_FEATURES)}")
        self.market = market
        self.model_type = model_type
        self.alpha = float(alpha)
        self.time_decay_half_life_days = time_decay_half_life_days
        self.numeric_features = list(numeric_features or MARKET_FEATURES.get(market, []))
        self.categorical_features = list(categorical_features or MARKET_CATEGORICAL.get(market, []))
        self.floor, self.cap = MARKET_BOUNDS.get(market, (0.05, 40.0))
        self.pipeline_: Pipeline | None = None
        self.mean_: float = 0.0

    def _available_features(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        nums = [c for c in self.numeric_features
                if c in X.columns and pd.to_numeric(X[c], errors="coerce").notna().any()]
        cats = [c for c in self.categorical_features
                if c in X.columns and X[c].notna().any()]
        return nums, cats

    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        nums, cats = self._available_features(X)
        preprocess = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                ]), nums),
                ("cat", Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]), cats),
            ],
            remainder="drop",
        )
        if self.model_type == "poisson":
            reg = PoissonRegressor(alpha=self.alpha, max_iter=1000)
        elif self.model_type == "gbm":
            reg = GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                min_samples_leaf=8, random_state=42,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        return Pipeline([("prep", preprocess), ("model", reg)])

    def _sample_weights(self, train: pd.DataFrame) -> np.ndarray | None:
        hl = self.time_decay_half_life_days
        if not hl or "date" not in train.columns:
            return None
        dates = pd.to_datetime(train["date"], errors="coerce")
        if dates.notna().sum() == 0:
            return None
        age = (dates.max() - dates).dt.days.clip(lower=0).fillna(float(hl)).astype(float)
        w = np.power(0.5, age / float(hl))
        return np.asarray(w, dtype=float)

    def fit(self, frame: pd.DataFrame) -> "EventLambdaModel":
        train = frame.dropna(subset=[self.market]).copy()
        if len(train) == 0:
            return self
        X = train.drop(columns=[self.market], errors="ignore")
        y = train[self.market].astype(float).clip(lower=0)
        self.mean_ = float(y.mean())
        self.pipeline_ = self._build_pipeline(X)
        w = self._sample_weights(train)
        if w is not None and self.model_type == "poisson":
            self.pipeline_.fit(X, y, model__sample_weight=w)
        else:
            self.pipeline_.fit(X, y)
        return self

    def predict_lambda(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            return np.full(len(frame), self.mean_ or self.floor)
        raw = self.pipeline_.predict(frame)
        return clip_lambda(raw, self.floor, self.cap)

    def prob_over(self, frame: pd.DataFrame, line: float) -> np.ndarray:
        lam = self.predict_lambda(frame)
        return 1.0 - poisson.cdf(int(np.floor(line)), lam)

    def prob_under(self, frame: pd.DataFrame, line: float) -> np.ndarray:
        lam = self.predict_lambda(frame)
        return poisson.cdf(int(np.floor(line)), lam)

