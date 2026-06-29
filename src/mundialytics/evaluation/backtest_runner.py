from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mundialytics.data.loaders import to_long_team_rows
from mundialytics.data.schema import infer_single_scope
from mundialytics.features.team_features import build_goal_training_frame, fixture_feature_row
from mundialytics.models.goal_model import GoalLambdaModel, GoalModelConfig
from mundialytics.models.result_model import match_probabilities
from mundialytics.ratings.elo import EloRater
from mundialytics.evaluation.metrics import brier_multiclass, rank_probability_score, safe_log_loss


@dataclass
class BacktestConfig:
    min_train_matches: int = 20
    model_type: str = "poisson"
    retrain_every: int = 50
    max_test_matches: int | None = 1200
    rf_n_estimators: int = 250
    rf_min_samples_leaf: int = 6
    poisson_alpha: float = 1.0
    time_decay_half_life_days: float | None = 365.0
    rolling_shrinkage_prior_matches: float = 10.0


def _actual_outcome(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _onehot(outcome: str) -> list[int]:
    return {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}[outcome]


def _prepare_completed(matches: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    infer_single_scope(matches)
    completed = matches.dropna(subset=["home_goals", "away_goals"]).sort_values(["date", "match_id"]).reset_index(drop=True)
    if cfg.max_test_matches is not None and len(completed) > cfg.min_train_matches + cfg.max_test_matches:
        # Keep a recent evaluation window while preserving enough training data.
        # This avoids the useless 1872->today full-history crawl for national teams.
        completed = completed.tail(cfg.min_train_matches + cfg.max_test_matches).reset_index(drop=True)
    if len(completed) <= cfg.min_train_matches:
        raise ValueError(f"Need more than {cfg.min_train_matches} completed matches for backtest; got {len(completed)}.")
    return completed


def _fit_state(train: pd.DataFrame, cfg: BacktestConfig) -> tuple[EloRater, pd.DataFrame, GoalLambdaModel]:
    rater = EloRater()
    elo_hist = rater.fit(train)
    team_rows = to_long_team_rows(train)
    frame = build_goal_training_frame(
        team_rows,
        elo_hist,
        rolling_shrinkage_prior_matches=cfg.rolling_shrinkage_prior_matches,
    )
    model = GoalLambdaModel(
        GoalModelConfig(
            model_type=cfg.model_type,
            poisson_alpha=cfg.poisson_alpha,
            rf_n_estimators=cfg.rf_n_estimators,
            rf_min_samples_leaf=cfg.rf_min_samples_leaf,
            time_decay_half_life_days=cfg.time_decay_half_life_days,
        )
    ).fit(frame)
    return rater, frame, model


def _predict_one(test: pd.Series, rater: EloRater, frame: pd.DataFrame, model: GoalLambdaModel) -> dict:
    elo_ctx = rater.transform_fixture(test["home_team"], test["away_team"], neutral=int(test.get("neutral", 0) or 0))
    ctx = {
        **elo_ctx,
        "neutral": int(test.get("neutral", 0) or 0),
        "competition": test.get("competition", "unknown"),
        "stage": test.get("stage", "unknown"),
        "home_external_elo": test.get("home_external_elo", test.get("home_clubelo", test.get("home_elo"))),
        "away_external_elo": test.get("away_external_elo", test.get("away_clubelo", test.get("away_elo"))),
    }
    X = fixture_feature_row(test["home_team"], test["away_team"], ctx, frame)
    lam_home, lam_away = model.predict_lambda(X)
    probs = match_probabilities(float(lam_home), float(lam_away))
    outcome = _actual_outcome(test["home_goals"], test["away_goals"])
    return {
        "match_id": test["match_id"],
        "date": test["date"],
        "competition": test.get("competition", "unknown"),
        "home_team": test["home_team"],
        "away_team": test["away_team"],
        "home_goals": test["home_goals"],
        "away_goals": test["away_goals"],
        "actual_outcome": outcome,
        "p_home_win": probs.p_home_win,
        "p_draw": probs.p_draw,
        "p_away_win": probs.p_away_win,
        "lambda_home": probs.lambda_home,
        "lambda_away": probs.lambda_away,
        "most_likely_score": probs.most_likely_score,
    }


def _summarise(pred: pd.DataFrame) -> dict:
    if pred.empty:
        return {
            "n_predictions": 0,
            "rps": None,
            "brier_multiclass": None,
            "log_loss": None,
            "accuracy_pick_max": 0.0,
            "avg_picked_probability": 0.0,
            "reliability_by_confidence_bin": [],
        }
    P = pred[["p_home_win", "p_draw", "p_away_win"]].to_numpy()
    O = pd.DataFrame([_onehot(o) for o in pred["actual_outcome"]]).to_numpy()
    pick_cols = ["p_home_win", "p_draw", "p_away_win"]
    picked = pred[pick_cols].idxmax(axis=1).map({"p_home_win": "H", "p_draw": "D", "p_away_win": "A"})
    pred["picked_outcome"] = picked
    pred["picked_probability"] = pred[pick_cols].max(axis=1)
    pred["picked_correct"] = (pred["picked_outcome"] == pred["actual_outcome"]).astype(int)
    pred["confidence_bin"] = pd.cut(pred["picked_probability"], bins=[0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0], include_lowest=True).astype(str)
    reliability = pred.groupby("confidence_bin", observed=False).agg(
        n=("picked_correct", "size"),
        avg_confidence=("picked_probability", "mean"),
        accuracy=("picked_correct", "mean"),
    ).reset_index()
    reliability = reliability[reliability["n"] > 0].to_dict(orient="records")
    return {
        "n_predictions": int(len(pred)),
        "rps": rank_probability_score(P, O),
        "brier_multiclass": brier_multiclass(O, P),
        "log_loss": safe_log_loss(pred["actual_outcome"], P, labels=["H", "D", "A"]),
        "accuracy_pick_max": float((picked == pred["actual_outcome"]).mean()),
        "avg_picked_probability": float(pred["picked_probability"].mean()) if len(pred) else 0.0,
        "reliability_by_confidence_bin": reliability,
    }


def walk_forward_backtest(matches: pd.DataFrame, config: BacktestConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Chunked expanding-window backtest with strict temporal ordering.

    Earlier versions retrained/rebuilt ELO and rolling features before every
    single match. That was theoretically safe but practically unusable on full
    national-team histories. This version retrains every ``retrain_every``
    matches and predicts the next chunk. Within the chunk, Elo is updated after
    each known result, so later predictions in the chunk still use information
    available at that time. Rolling-form features are held constant within the
    chunk; keep ``retrain_every`` moderate for a good speed/accuracy trade-off.
    """
    cfg = config or BacktestConfig()
    completed = _prepare_completed(matches, cfg)
    rows: list[dict] = []

    for block_start in range(cfg.min_train_matches, len(completed), max(1, cfg.retrain_every)):
        train = completed.iloc[:block_start].copy()
        test_block = completed.iloc[block_start:block_start + max(1, cfg.retrain_every)].copy()
        rater, frame, model = _fit_state(train, cfg)
        for _, test in test_block.iterrows():
            rows.append(_predict_one(test, rater, frame, model))
            # Update Elo using the actual result so the next fixture in the same
            # chunk has a current rating without retraining the whole model.
            rater.update_match(test)

    pred = pd.DataFrame(rows)
    summary = _summarise(pred)
    summary["model_config"] = {
        "model_type": cfg.model_type,
        "poisson_alpha": cfg.poisson_alpha,
        "rf_n_estimators": cfg.rf_n_estimators,
        "rf_min_samples_leaf": cfg.rf_min_samples_leaf,
        "time_decay_half_life_days": cfg.time_decay_half_life_days,
        "rolling_shrinkage_prior_matches": cfg.rolling_shrinkage_prior_matches,
        "internal_elo_features": True,
        "external_elo_features_supported": True,
    }
    return pred, summary
