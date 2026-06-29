from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.artifacts.model_bundle import load_model_bundle
from mundialytics.data.fixtures import load_fixtures, predict_fixture_probabilities


def main() -> None:
    p = argparse.ArgumentParser(description="Predict future fixtures from a scoped trained bundle.")
    p.add_argument("--bundle", default="models/goal_model.pkl")
    p.add_argument("--fixtures", required=True, help="Canonical fixture CSV")
    p.add_argument("--out", default="outputs/fixture_predictions.csv")
    p.add_argument("--metadata-out", default=None, help="Optional JSON metadata output")
    args = p.parse_args()

    bundle_path = ROOT / args.bundle if not Path(args.bundle).is_absolute() else Path(args.bundle)
    fixture_path = ROOT / args.fixtures if not Path(args.fixtures).is_absolute() else Path(args.fixtures)
    bundle = load_model_bundle(bundle_path)
    fixtures = load_fixtures(fixture_path)
    try:
        bundle.validate_fixtures(fixtures)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    preds = predict_fixture_probabilities(
        fixtures,
        goal_model=bundle.goal_model,
        elo_rater=bundle.elo_rater,
        historical_frame=bundle.training_frame,
    )
    preds["model_scope"] = bundle.model_scope
    preds["model_type"] = bundle.model_type
    preds["model_created_at_utc"] = bundle.created_at_utc

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out, index=False)
    print(f"Saved fixture predictions to {out}")
    print(preds[["date", "competition", "home_team", "away_team", "p_home_win", "p_draw", "p_away_win", "p_over_25", "most_likely_score"]].to_string(index=False))

    if args.metadata_out:
        meta_out = ROOT / args.metadata_out if not Path(args.metadata_out).is_absolute() else Path(args.metadata_out)
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        meta_out.write_text(json.dumps(bundle.metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
