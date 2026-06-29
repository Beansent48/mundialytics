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

from dataclasses import dataclass, field
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


# ── Player Profile ─────────────────────────────────────────────────────────────

@dataclass
class PlayerEventRate:
    """Per-match rate estimate for a single event type."""
    event: str
    rate_per_match: float
    sample_matches: int
    confidence: float       # 0-1; increases with sample size
    source: str             # "player_history", "position_median", "generic_prior"

    def prob_over(self, line: float) -> float:
        """P(player achieves > line events in a match) under Poisson."""
        return float(1.0 - poisson.cdf(int(np.floor(line)), max(self.rate_per_match, 1e-6)))

    def prob_at_least(self, k: int) -> float:
        if k <= 0:
            return 1.0
        return float(1.0 - poisson.cdf(k - 1, max(self.rate_per_match, 1e-6)))


@dataclass
class PlayerProfile:
    """Comprehensive per-player event rate profile.

    Rates are per match (not per 90) because the StatsBomb pre-aggregated
    CSV does not include minutes.  When lineup data arrives with expected
    minutes, multiply rate by (expected_minutes / avg_minutes_in_sample).
    """
    player: str
    player_id: str
    team: str
    competition: str
    position: str
    matches: int
    rates: dict[str, PlayerEventRate] = field(default_factory=dict)

    def get(self, event: str) -> PlayerEventRate | None:
        return self.rates.get(event)

    def prob_over(self, event: str, line: float) -> float:
        r = self.rates.get(event)
        if r is None:
            return float("nan")
        return r.prob_over(line)

    def summary(self) -> pd.Series:
        d = {"player": self.player, "team": self.team,
             "competition": self.competition, "position": self.position,
             "matches": self.matches}
        for evt, rate in self.rates.items():
            d[f"{evt}_per_match"] = round(rate.rate_per_match, 3)
            d[f"{evt}_sample"] = rate.sample_matches
        return pd.Series(d)


GENERIC_PRIORS: dict[str, dict[str, float]] = {
    "Forward": {"shots": 2.0, "sot": 0.8, "goals": 0.30, "assists": 0.12,
                "fouls": 0.9, "yellow_cards": 0.08, "tackles": 0.6, "pressures": 6.0},
    "Midfielder": {"shots": 1.0, "sot": 0.35, "goals": 0.10, "assists": 0.15,
                   "fouls": 1.2, "yellow_cards": 0.12, "tackles": 1.8, "pressures": 12.0},
    "Defender": {"shots": 0.4, "sot": 0.12, "goals": 0.04, "assists": 0.06,
                 "fouls": 1.4, "yellow_cards": 0.15, "tackles": 2.5, "pressures": 10.0},
    "Goalkeeper": {"shots": 0.02, "sot": 0.01, "goals": 0.001, "assists": 0.01,
                   "fouls": 0.2, "yellow_cards": 0.04, "tackles": 0.1, "pressures": 1.0},
    "Unknown": {"shots": 1.0, "sot": 0.35, "goals": 0.12, "assists": 0.08,
                "fouls": 1.1, "yellow_cards": 0.10, "tackles": 1.5, "pressures": 8.0},
}

EVENTS = ["shots", "sot", "goals", "assists", "fouls", "yellow_cards", "tackles", "pressures"]


class PlayerProfileModel:
    """Builds and stores per-player event rate profiles from StatsBomb data.

    Usage:
        model = PlayerProfileModel()
        model.fit(statsbomb_player_df)
        profile = model.get_profile("Lionel Messi")
        p = profile.prob_over("shots", 2.5)   # P(Messi > 2.5 shots)
    """

    def __init__(self, min_matches: int = 5, prior_weight: float = 10.0):
        self.min_matches = int(min_matches)
        self.prior_weight = float(prior_weight)   # empirical-Bayes shrinkage strength
        self.profiles_: dict[str, PlayerProfile] = {}
        self.position_medians_: dict[str, dict[str, float]] = {}
        self.competition_position_medians_: dict[tuple[str, str], dict[str, float]] = {}

    def _resolve_position(self, pos: str | None) -> str:
        p = str(pos or "").strip()
        if not p or p.lower() in {"nan", "none", "unknown", ""}:
            return "Unknown"
        p_lower = p.lower()
        if any(k in p_lower for k in ["forward", "attacker", "striker", "winger", "fw", "cf", "lw", "rw", "st"]):
            return "Forward"
        if any(k in p_lower for k in ["midfield", "mid", "cm", "cam", "cdm", "lm", "rm", "dm"]):
            return "Midfielder"
        if any(k in p_lower for k in ["defend", "back", "cb", "lb", "rb", "def", "sw"]):
            return "Defender"
        if any(k in p_lower for k in ["keeper", "goal", "gk"]):
            return "Goalkeeper"
        return "Unknown"

    def _shrink(self, raw_rate: float, n: float, prior: float) -> float:
        """Empirical-Bayes shrinkage toward position prior."""
        w = self.prior_weight
        return (raw_rate * n + prior * w) / (n + w)

    def fit(
        self,
        player_df: pd.DataFrame,
        positions_df: pd.DataFrame | None = None,
    ) -> "PlayerProfileModel":
        """Fit player profiles.

        Parameters
        ----------
        player_df : StatsBomb player match stats DataFrame.
        positions_df : Optional DataFrame with columns [player, position_group]
            pre-fetched from StatsBomb lineups.  When supplied, player
            positions override the raw ``position`` column (which is often
            all-NaN in the pre-aggregated CSV).
        """
        df = player_df.copy()
        for c in EVENTS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # Merge pre-fetched positions (StatsBomb lineups → GitHub)
        if positions_df is not None and not positions_df.empty:
            pos_map = positions_df.set_index("player")["position_group"].to_dict()
            df["position_group"] = df["player"].map(pos_map).fillna(
                df.get("position", pd.Series(dtype=str)).apply(self._resolve_position)
            )
        else:
            df["position_group"] = df.get("position", pd.Series(dtype=str)).apply(self._resolve_position)

        df["competition_c"] = df.get("competition", pd.Series(dtype=str)).fillna("unknown")

        # Compute position medians (for prior)
        for pg, sub in df.groupby("position_group"):
            self.position_medians_[str(pg)] = {
                evt: float(sub[evt].mean()) if evt in sub.columns else GENERIC_PRIORS.get(str(pg), {}).get(evt, 0.5)
                for evt in EVENTS
            }

        # Compute competition × position medians
        for (comp, pg), sub in df.groupby(["competition_c", "position_group"]):
            self.competition_position_medians_[(str(comp), str(pg))] = {
                evt: float(sub[evt].mean()) if evt in sub.columns and sub[evt].notna().any() else 0.0
                for evt in EVENTS
            }

        # Per-player profiles
        id_col = "player_id" if "player_id" in df.columns else "player"
        for player_key, grp in df.groupby("player"):
            n = len(grp)
            pos_group = grp["position_group"].mode()[0] if "position_group" in grp.columns else "Unknown"
            team = str(grp.get("team", pd.Series(["unknown"])).mode()[0]) if "team" in grp.columns else "unknown"
            comp = str(grp["competition_c"].mode()[0]) if "competition_c" in grp.columns else "unknown"
            pid = str(grp[id_col].iloc[0]) if id_col in grp.columns else player_key

            rates: dict[str, PlayerEventRate] = {}
            pos_priors = self.position_medians_.get(pos_group, GENERIC_PRIORS.get(pos_group, GENERIC_PRIORS["Unknown"]))
            ctx_priors = self.competition_position_medians_.get((comp, pos_group), pos_priors)

            for evt in EVENTS:
                if evt not in grp.columns:
                    continue
                raw = float(grp[evt].mean()) if grp[evt].notna().any() else 0.0
                prior = ctx_priors.get(evt, pos_priors.get(evt, GENERIC_PRIORS["Unknown"].get(evt, 0.5)))
                shrunk = self._shrink(raw, float(n), prior)
                confidence = float(n / (n + self.prior_weight))
                source = "player_history" if n >= self.min_matches else "shrunk_low_sample"
                rates[evt] = PlayerEventRate(
                    event=evt,
                    rate_per_match=round(shrunk, 4),
                    sample_matches=int(n),
                    confidence=round(confidence, 3),
                    source=source,
                )

            self.profiles_[player_key] = PlayerProfile(
                player=player_key,
                player_id=pid,
                team=team,
                competition=comp,
                position=pos_group,
                matches=int(n),
                rates=rates,
            )
        return self

    def get_profile(
        self,
        player: str,
        position: str | None = None,
        competition: str | None = None,
    ) -> PlayerProfile:
        """Return profile or a generic prior if player not in training data."""
        if player in self.profiles_:
            return self.profiles_[player]
        # Fallback: position × competition prior
        pos_group = self._resolve_position(position)
        comp = competition or "unknown"
        ctx = self.competition_position_medians_.get(
            (comp, pos_group),
            self.position_medians_.get(pos_group, GENERIC_PRIORS.get(pos_group, GENERIC_PRIORS["Unknown"])),
        )
        rates: dict[str, PlayerEventRate] = {}
        for evt in EVENTS:
            prior = float(ctx.get(evt, GENERIC_PRIORS["Unknown"].get(evt, 0.5)))
            rates[evt] = PlayerEventRate(
                event=evt,
                rate_per_match=round(prior, 4),
                sample_matches=0,
                confidence=0.0,
                source="position_prior",
            )
        return PlayerProfile(
            player=player, player_id="unknown", team="unknown",
            competition=comp, position=pos_group, matches=0, rates=rates,
        )

    def all_profiles(self) -> pd.DataFrame:
        return pd.DataFrame([p.summary() for p in self.profiles_.values()])
