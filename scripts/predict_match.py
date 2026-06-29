from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.artifacts.model_bundle import load_model_bundle
from mundialytics.features.team_features import fixture_feature_row
from mundialytics.models.result_model import match_probabilities, summarize_score_matrix


def main() -> None:
    p = argparse.ArgumentParser(description="Predict one match from a trained scoped bundle.")
    p.add_argument("--bundle", default="models/goal_model.pkl")
    p.add_argument("--home", required=True)
    p.add_argument("--away", required=True)
    p.add_argument("--neutral", type=int, default=1)
    p.add_argument("--competition", default="unknown")
    p.add_argument("--stage", default="unknown")
    args = p.parse_args()

    bundle = load_model_bundle(ROOT / args.bundle if not Path(args.bundle).is_absolute() else args.bundle)
    fixture = bundle.elo_rater.transform_fixture(args.home, args.away, neutral=args.neutral)
    ctx = {**fixture, "neutral": args.neutral, "competition": args.competition, "stage": args.stage}
    X = fixture_feature_row(args.home, args.away, ctx, bundle.training_frame)
    lam = bundle.goal_model.predict_lambda(X)
    probs = match_probabilities(lam[0], lam[1])
    print({
        "model_scope": bundle.model_scope,
        "home": args.home,
        "away": args.away,
        "lambda_home": probs.lambda_home,
        "lambda_away": probs.lambda_away,
        "p_home_win": probs.p_home_win,
        "p_draw": probs.p_draw,
        "p_away_win": probs.p_away_win,
        "p_over_25": probs.p_over_25,
        "p_btts": probs.p_btts,
        "most_likely_score": probs.most_likely_score,
    })
    print(summarize_score_matrix(probs.score_matrix).to_string(index=False))


if __name__ == "__main__":
    main()
