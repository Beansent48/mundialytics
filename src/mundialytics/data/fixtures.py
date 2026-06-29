from __future__ import annotations

from pathlib import Path

import pandas as pd

from mundialytics.data.schema import normalize_fixtures
from mundialytics.features.team_features import fixture_feature_row
from mundialytics.models.result_model import match_probabilities


def load_fixtures(path: str | Path) -> pd.DataFrame:
    """Load future fixtures in the canonical project format."""
    df = pd.read_csv(path)
    return normalize_fixtures(df)


def predict_fixture_probabilities(fixtures: pd.DataFrame, goal_model, elo_rater, historical_frame: pd.DataFrame) -> pd.DataFrame:
    """Predict 1X2 and goal-market probabilities for a slate of fixtures.

    This is source-agnostic: fixtures can come from World Cup, Euros, Premier
    League, LaLiga, Champions League, etc., as long as they use the canonical
    fixture schema.
    """
    rows: list[dict] = []
    fx = normalize_fixtures(fixtures)
    for _, f in fx.iterrows():
        home = f["home_team"]
        away = f["away_team"]
        neutral = int(f.get("neutral", 0) or 0)
        elo_ctx = elo_rater.transform_fixture(home, away, neutral=neutral)
        ctx = {
            **elo_ctx,
            "neutral": neutral,
            "competition": f.get("competition", "unknown"),
            "stage": f.get("stage", "unknown"),
            "team_scope": f.get("team_scope", "unknown"),
        }
        X = fixture_feature_row(home, away, ctx, historical_frame)
        lambdas = goal_model.predict_lambda(X)
        probs = match_probabilities(float(lambdas[0]), float(lambdas[1]))
        rows.append({
            "fixture_id": f.get("fixture_id"),
            "date": f.get("date"),
            "competition": f.get("competition", "unknown"),
            "season": f.get("season", None),
            "stage": f.get("stage", "unknown"),
            "team_scope": f.get("team_scope", "unknown"),
            "home_team": home,
            "away_team": away,
            "neutral": neutral,
            "home_elo": elo_ctx["home_elo"],
            "away_elo": elo_ctx["away_elo"],
            "elo_diff": elo_ctx["elo_diff"],
            "expected_home_score_elo": elo_ctx["expected_home_score_elo"],
            "lambda_home": probs.lambda_home,
            "lambda_away": probs.lambda_away,
            "p_home_win": probs.p_home_win,
            "p_draw": probs.p_draw,
            "p_away_win": probs.p_away_win,
            "p_over_25": probs.p_over_25,
            "p_btts": probs.p_btts,
            "most_likely_score": probs.most_likely_score,
        })
    return pd.DataFrame(rows)
