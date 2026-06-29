from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.artifacts.model_bundle import load_model_bundle
from mundialytics.data.fixtures import load_fixtures, predict_fixture_probabilities
from mundialytics.data.identity import canonical_team_name
from mundialytics.data.competition_taxonomy import enrich_competition_metadata
from mundialytics.inference.safe_props import DEFAULT_MARKETS, predict_props_for_lineups, validate_current_lineups
from mundialytics.reports.match_value import build_match_value_picks
from mundialytics.features.team_match_stats import predict_team_props_simple


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _fixture_key(df: pd.DataFrame) -> str:
    if "fixture_id" in df.columns:
        return "fixture_id"
    if "match_id" in df.columns:
        return "match_id"
    raise ValueError("Expected fixture_id or match_id column")


def attach_match_context_to_lineups(lineups: pd.DataFrame, match_predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach ELO/Poisson context to current lineup rows for props.

    The context is team-perspective: if the lineup team is home, elo_diff is the
    home elo_diff; if away, it is negated. Expected possession is a rough bounded
    transform of Elo expected score, used only as a small contextual multiplier.
    """
    lu = enrich_competition_metadata(lineups, overwrite=True)
    mp = enrich_competition_metadata(match_predictions, overwrite=True)
    key_pred = _fixture_key(mp)
    if "match_id" not in lu.columns:
        if "fixture_id" in lu.columns:
            lu = lu.rename(columns={"fixture_id": "match_id"})
        else:
            raise ValueError("lineups must contain match_id or fixture_id")
    if key_pred != "match_id":
        mp = mp.rename(columns={key_pred: "match_id"})
    for col in ["team", "opponent"]:
        if col in lu.columns:
            lu[col] = lu[col].map(canonical_team_name)
    for col in ["home_team", "away_team"]:
        if col in mp.columns:
            mp[col] = mp[col].map(canonical_team_name)
    meta_cols = [
        "match_id", "date", "competition", "team_scope", "team_type", "competition_context", "gender", "home_team", "away_team",
        "home_elo", "away_elo", "elo_diff", "expected_home_score_elo",
        "lambda_home", "lambda_away", "p_home_win", "p_draw", "p_away_win", "p_over_25", "p_btts", "most_likely_score",
    ]
    meta = mp[[c for c in meta_cols if c in mp.columns]].drop_duplicates("match_id")
    out = lu.merge(meta, on="match_id", how="left", suffixes=("", "_match"), validate="many_to_one")
    if out["home_team"].isna().any():
        bad = out.loc[out["home_team"].isna(), "match_id"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Lineup contains match_id not present in fixture predictions: {bad}")
    is_home = out["team"].astype(str) == out["home_team"].astype(str)
    is_away = out["team"].astype(str) == out["away_team"].astype(str)
    if (~(is_home | is_away)).any():
        bad = out.loc[~(is_home | is_away), ["match_id", "team", "home_team", "away_team"]].drop_duplicates().head(10).to_dict(orient="records")
        raise ValueError(f"Lineup team does not match fixture home/away teams. Examples: {bad}")
    out["opponent"] = out["away_team"].where(is_home, out["home_team"])
    out["elo_diff"] = out["elo_diff"].where(is_home, -out["elo_diff"])
    home_exp = pd.to_numeric(out["expected_home_score_elo"], errors="coerce").fillna(0.5)
    team_exp = home_exp.where(is_home, 1 - home_exp)
    out["expected_possession"] = (50 + (team_exp - 0.5) * 34).clip(lower=35, upper=65)
    # Prefer fixture-level metadata over possibly missing lineup metadata.
    for col in ["date", "competition", "team_scope", "team_type", "competition_context", "gender"]:
        match_col = f"{col}_match"
        if match_col in out.columns:
            out[col] = out[col].where(out[col].notna() & (out[col].astype(str).str.strip() != ""), out[match_col])
            out = out.drop(columns=[match_col])
    return enrich_competition_metadata(out, overwrite=True)


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end matchday analysis: ELO+Poisson fixtures + safe lineup props + optional 1X2 value.")
    p.add_argument("--goal-bundle", required=True, help="Trained goal model bundle, e.g. outputs/.../final_national_poisson_model.pkl")
    p.add_argument("--fixtures", required=True, help="Current fixtures CSV")
    p.add_argument("--lineups", required=True, help="Current official/expected lineups CSV. This is the only candidate set for props.")
    p.add_argument("--player-events", required=True, help="Clean historical player-events CSV")
    p.add_argument("--calibration-predictions", default=None)
    p.add_argument("--calibration-results", default=None)
    p.add_argument("--calibration-policy", default=None, help="Optional player_props_policy.json")
    p.add_argument("--identity-map", default=None, help="Optional provider ID ↔ historical player identity map CSV")
    p.add_argument("--match-odds", default=None, help="Optional 1X2 odds CSV")
    p.add_argument("--team-match-stats", default=None, help="Optional clean team_match_stats.csv for team/match event prop predictions")
    p.add_argument("--out-dir", default="outputs/matchday_analysis")
    p.add_argument("--markets", nargs="+", default=list(DEFAULT_MARKETS))
    p.add_argument("--line", default="1+")
    p.add_argument("--min-calibration-rows", type=int, default=500)
    p.add_argument("--min-edge", type=float, default=0.03)
    p.add_argument("--min-ev", type=float, default=0.03)
    p.add_argument("--commission", type=float, default=0.0)
    args = p.parse_args()

    out_dir = _resolve(args.out_dir)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_model_bundle(_resolve(args.goal_bundle))
    fixtures = load_fixtures(_resolve(args.fixtures))
    bundle.validate_fixtures(fixtures)
    match_predictions = predict_fixture_probabilities(fixtures, bundle.goal_model, bundle.elo_rater, bundle.training_frame)
    match_predictions["model_scope"] = bundle.model_scope
    match_predictions["model_type"] = bundle.model_type
    match_predictions.to_csv(out_dir / "match_predictions.csv", index=False)

    lineups = pd.read_csv(_resolve(args.lineups))
    validate_current_lineups(lineups, strict=True)
    lineups_context = attach_match_context_to_lineups(lineups, match_predictions)
    lineups_context.to_csv(out_dir / "lineups_with_match_context.csv", index=False)

    events = pd.read_csv(_resolve(args.player_events))
    cal_pred_path = _resolve(args.calibration_predictions)
    cal_results_path = _resolve(args.calibration_results)
    cal_policy_path = _resolve(args.calibration_policy)
    identity_map_path = _resolve(args.identity_map)
    cal_pred = pd.read_csv(cal_pred_path) if cal_pred_path and cal_pred_path.exists() else None
    cal_results = pd.read_csv(cal_results_path) if cal_results_path and cal_results_path.exists() else None
    market_policy = json.loads(cal_policy_path.read_text(encoding="utf-8")) if cal_policy_path and cal_policy_path.exists() else None
    props = predict_props_for_lineups(
        events,
        lineups_context,
        markets=args.markets,
        line=args.line,
        calibration_predictions=cal_pred,
        calibration_results=cal_results,
        min_calibration_rows=args.min_calibration_rows,
        strict_lineup_contract=True,
        market_calibration_policy=market_policy,
        identity_map=str(identity_map_path) if identity_map_path else None,
    )
    props.to_csv(out_dir / "safe_lineup_props.csv", index=False)

    team_props_path = None
    team_props = pd.DataFrame()
    if args.team_match_stats:
        team_stats = pd.read_csv(_resolve(args.team_match_stats))
        team_props = predict_team_props_simple(team_stats, fixtures)
        team_props_path = out_dir / "team_props_predictions.csv"
        team_props.to_csv(team_props_path, index=False)

    value_path = None
    if args.match_odds:
        odds = pd.read_csv(_resolve(args.match_odds))
        picks = build_match_value_picks(match_predictions, odds, min_edge=args.min_edge, min_ev=args.min_ev, commission=args.commission)
        value_path = out_dir / "match_value_picks.csv"
        picks.to_csv(value_path, index=False)

    report = {
        "status": "MATCHDAY_ANALYSIS_COMPLETE",
        "goal_bundle": str(_resolve(args.goal_bundle)),
        "fixtures": str(_resolve(args.fixtures)),
        "lineups": str(_resolve(args.lineups)),
        "player_events": str(_resolve(args.player_events)),
        "outputs": {
            "match_predictions": str(out_dir / "match_predictions.csv"),
            "lineups_with_match_context": str(out_dir / "lineups_with_match_context.csv"),
            "safe_lineup_props": str(out_dir / "safe_lineup_props.csv"),
            "team_props_predictions": str(team_props_path) if team_props_path else None,
            "identity_map": str(identity_map_path) if identity_map_path else None,
            "calibration_policy": str(cal_policy_path) if cal_policy_path else None,
            "match_value_picks": str(value_path) if value_path else None,
        },
        "n_fixtures": int(len(match_predictions)),
        "n_lineup_players": int(lineups[["match_id", "team", "player"]].drop_duplicates().shape[0]),
        "n_prop_predictions": int(len(props)),
        "n_team_prop_predictions": int(len(team_props)) if args.team_match_stats else 0,
        "prop_markets": sorted(props["market_type"].unique().tolist()) if not props.empty else [],
        "rules": [
            "ELO+Poisson predicts match probabilities and goal markets.",
            "Player props are generated only for supplied current lineup rows.",
            "Historical/retired players can train rates but cannot appear unless present in lineups.",
            "safe_probability applies market caps and low-sample guards.",
        ],
    }
    (out_dir / "matchday_analysis_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
