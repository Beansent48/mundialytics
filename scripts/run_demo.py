from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.config import load_config, ensure_dirs
from mundialytics.data.loaders import load_matches, load_player_events, load_odds, load_lineups, to_long_team_rows
from mundialytics.ratings.elo import EloConfig, EloRater
from mundialytics.features.team_features import build_goal_training_frame, fixture_feature_row
from mundialytics.models.goal_model import GoalLambdaModel, GoalModelConfig
from mundialytics.models.result_model import match_probabilities, summarize_score_matrix
from mundialytics.models.team_event_model import TeamEventModel
from mundialytics.models.player_event_model import PlayerEventModel
from mundialytics.models.minutes_model import MinutesModel
from mundialytics.reports.daily_picks import build_daily_player_props, format_pick_text


def main() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    ensure_dirs(ROOT / cfg["reports"]["output_dir"])

    matches = load_matches(ROOT / cfg["data"]["sample_matches_path"])
    events = load_player_events(ROOT / cfg["data"]["sample_player_events_path"])
    odds = load_odds(ROOT / cfg["data"]["sample_odds_path"])
    lineups = load_lineups(ROOT / cfg["data"]["sample_lineups_path"])

    elo_cfg = EloConfig(**cfg["elo"])
    rater = EloRater(elo_cfg)
    elo_history = rater.fit(matches)

    completed = matches.dropna(subset=["home_goals", "away_goals"])
    team_rows = to_long_team_rows(completed)
    goal_frame = build_goal_training_frame(team_rows, elo_history)

    goal_model = GoalLambdaModel(GoalModelConfig(**{
        "model_type": "poisson",
        "poisson_alpha": cfg["models"]["poisson_alpha"],
        "lambda_floor": cfg["models"]["lambda_floor"],
        "lambda_cap": cfg["models"]["lambda_cap"],
    }))
    goal_model.fit(goal_frame)

    # Predict Spain vs Uruguay sample fixture
    fixture = rater.transform_fixture("Uruguay", "Spain", neutral=1)
    match_context = {**fixture, "neutral": 1, "competition": "World Cup", "stage": "Group"}
    fixture_rows = fixture_feature_row("Uruguay", "Spain", match_context, goal_frame)
    lambdas = goal_model.predict_lambda(fixture_rows)
    probs = match_probabilities(lambdas[0], lambdas[1], max_goals=cfg["models"]["max_goals_matrix"])

    print("\n=== Match prediction: Uruguay vs Spain ===")
    print(f"ELO Uruguay={fixture['home_elo']:.0f} | Spain={fixture['away_elo']:.0f} | diff={fixture['elo_diff']:.0f}")
    print(f"lambda Uruguay={probs.lambda_home:.2f}, lambda Spain={probs.lambda_away:.2f}")
    print(f"1X2: Uruguay {probs.p_home_win:.1%}, Draw {probs.p_draw:.1%}, Spain {probs.p_away_win:.1%}")
    print(f"Over 2.5={probs.p_over_25:.1%}, BTTS={probs.p_btts:.1%}, most likely={probs.most_likely_score}")
    print("Top scores:")
    print(summarize_score_matrix(probs.score_matrix).to_string(index=False))

    # Team events + player props
    team_event_model = TeamEventModel().fit(goal_frame)
    team_event_pred = team_event_model.predict(fixture_rows)
    print("\n=== Expected team events ===")
    print(team_event_pred.to_string(index=False))

    player_model = PlayerEventModel(cfg["models"]["min_minutes_for_player_rate"]).fit(events)
    minutes_model = MinutesModel().fit(events, projected_lineups=lineups)

    ctx_by_match_team = {
        (17, "uruguay"): {"elo_diff": fixture["elo_diff"], "expected_possession": 47},
        (17, "spain"): {"elo_diff": -fixture["elo_diff"], "expected_possession": 53},
    }
    picks = build_daily_player_props(
        odds,
        player_model,
        minutes_model,
        lineups=lineups,
        team_context_by_match_team=ctx_by_match_team,
        min_edge=cfg["betting"]["min_edge"],
        min_ev=cfg["betting"]["min_expected_return"],
        commission=cfg["betting"]["exchange_commission"],
    )
    out_path = ROOT / cfg["reports"]["output_dir"] / "demo_daily_picks.csv"
    picks.to_csv(out_path, index=False)

    print("\n=== Top player props / value table ===")
    print(picks[["player", "market_type", "line", "odds", "model_probability", "model_probability_adjusted", "implied_probability", "edge", "expected_return", "value_flag", "sample_size", "replacement"]].to_string(index=False))
    print(f"\nSaved: {out_path}")
    if not picks.empty:
        print("\nText picks:")
        for _, row in picks.head(5).iterrows():
            print("- " + format_pick_text(row))


if __name__ == "__main__":
    main()
