from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.artifacts.model_bundle import create_model_bundle, save_model_bundle
from mundialytics.data.loaders import load_matches, to_long_team_rows
from mundialytics.data.schema import infer_single_scope
from mundialytics.features.team_features import build_goal_training_frame
from mundialytics.models.goal_model import GoalLambdaModel, GoalModelConfig
from mundialytics.ratings.elo import EloRater


def main() -> None:
    p = argparse.ArgumentParser(description="Train a scoped ELO + goal-lambda model from canonical matches.")
    p.add_argument("--matches", required=True, help="Canonical CSV with match_id,date,home_team,away_team,home_goals,away_goals,neutral,team_scope")
    p.add_argument("--model-out", default="models/goal_model.pkl")
    p.add_argument("--model-type", choices=["poisson", "random_forest_lambda"], default="poisson")
    p.add_argument("--data-source", default="unknown")
    args = p.parse_args()

    path = Path(args.matches)
    matches = load_matches(ROOT / path if not path.is_absolute() else path)
    scope = infer_single_scope(matches)
    completed = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    if len(completed) < 4:
        raise ValueError("Need at least 4 completed matches for a meaningful training run.")

    rater = EloRater()
    elo_hist = rater.fit(completed)
    rows = to_long_team_rows(completed)
    frame = build_goal_training_frame(rows, elo_hist)
    model = GoalLambdaModel(GoalModelConfig(model_type=args.model_type)).fit(frame)
    bundle = create_model_bundle(model, rater, frame, completed, model_type=args.model_type, data_source=args.data_source)

    out = ROOT / args.model_out if not Path(args.model_out).is_absolute() else Path(args.model_out)
    save_model_bundle(bundle, out)
    print(f"Saved {scope} model bundle to {out}")
    print(json.dumps(bundle.metadata, indent=2))


if __name__ == "__main__":
    main()
