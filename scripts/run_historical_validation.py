from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.artifacts.model_bundle import create_model_bundle, save_model_bundle
from mundialytics.data.adapters import football_data_uk_to_matches, international_results_to_matches, openfootball_json_to_matches
from mundialytics.data.adapters.football_data_uk import football_data_uk_to_match_odds
from mundialytics.data.loaders import load_matches, to_long_team_rows
from mundialytics.data.quality import data_quality_report
from mundialytics.data.schema import infer_single_scope
from mundialytics.data_quality.match_dataset_foundation import prepare_match_dataset
from mundialytics.evaluation.backtest_runner import BacktestConfig, walk_forward_backtest
from mundialytics.evaluation.readiness import ReadinessThresholds, evaluate_readiness
from mundialytics.evaluation.statistical_engine import evaluate_statistical_engine
from mundialytics.evaluation.value_backtest import run_match_value_backtest
from mundialytics.features.team_features import build_goal_training_frame
from mundialytics.models.goal_model import GoalLambdaModel, GoalModelConfig
from mundialytics.ratings.elo import EloRater
from mundialytics.reports.match_value import build_match_value_picks


def _resolve(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_or_build_matches(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    if args.matches:
        return load_matches(_resolve(args.matches))
    if not args.input or not args.source:
        raise ValueError("Provide either --matches canonical.csv OR --source + --input raw-source-file.")
    inp = _resolve(args.input)
    if args.source == "football-data-uk":
        matches = football_data_uk_to_matches(inp, season=args.season)
    elif args.source == "international-results":
        matches = international_results_to_matches(inp)
    elif args.source == "openfootball":
        matches = openfootball_json_to_matches(inp, competition=args.competition or "openfootball", season=args.season or "unknown", team_scope=args.team_scope)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported source: {args.source}")
    out = out_dir / "canonical_matches.csv"
    matches.to_csv(out, index=False)
    return matches


def _apply_operational_filters(matches: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply sport-aware filters so validation stays useful and finite.

    National-team football from the 19th century is not useful for a modern
    World Cup betting model and makes walk-forward backtests painfully slow.
    Unless --full-history is explicitly requested, we validate on a modern,
    recent window.
    """
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    info: dict[str, Any] = {"input_rows": int(len(df)), "filters": []}

    start_date = args.start_date
    if start_date is None and args.source == "international-results" and not args.full_history:
        start_date = "2010-01-01"
        info["filters"].append("default_start_date_for_international_results=2010-01-01")
    if start_date:
        before = len(df)
        df = df[df["date"] >= pd.to_datetime(start_date)].copy()
        info["filters"].append(f"start_date>={start_date}: {before}->{len(df)}")
    if args.end_date:
        before = len(df)
        df = df[df["date"] <= pd.to_datetime(args.end_date)].copy()
        info["filters"].append(f"end_date<={args.end_date}: {before}->{len(df)}")

    if not args.full_history and args.max_completed_matches:
        completed_mask = df[["home_goals", "away_goals"]].notna().all(axis=1)
        completed = df[completed_mask].sort_values("date")
        incomplete = df[~completed_mask]
        if len(completed) > args.max_completed_matches:
            keep_ids = set(completed.tail(args.max_completed_matches)["match_id"].astype(str))
            before = len(df)
            df = pd.concat([completed[completed["match_id"].astype(str).isin(keep_ids)], incomplete], ignore_index=True).sort_values("date")
            info["filters"].append(f"max_completed_matches={args.max_completed_matches}: {before}->{len(df)}")

    info["output_rows"] = int(len(df))
    if len(df):
        info["date_min"] = str(pd.to_datetime(df["date"], errors="coerce").min().date())
        info["date_max"] = str(pd.to_datetime(df["date"], errors="coerce").max().date())
    return df.reset_index(drop=True), info


def _train_final_bundle(
    matches: pd.DataFrame,
    model_type: str,
    data_source: str,
    out_path: Path,
    *,
    poisson_alpha: float = 1.0,
    time_decay_half_life_days: float | None = 365.0,
    rolling_shrinkage_prior_matches: float = 10.0,
) -> dict[str, Any]:
    completed = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    rater = EloRater()
    elo_hist = rater.fit(completed)
    frame = build_goal_training_frame(
        to_long_team_rows(completed),
        elo_hist,
        rolling_shrinkage_prior_matches=rolling_shrinkage_prior_matches,
    )
    model = GoalLambdaModel(
        GoalModelConfig(
            model_type=model_type,
            poisson_alpha=poisson_alpha,
            time_decay_half_life_days=time_decay_half_life_days,
        )
    ).fit(frame)
    bundle = create_model_bundle(model, rater, frame, completed, model_type=model_type, data_source=data_source)
    save_model_bundle(bundle, out_path)
    return bundle.metadata


def _model_rank_key(summary: dict[str, Any]) -> tuple[float, float, float]:
    # Lower is better. Accuracy is inverted as a final tie-breaker.
    return (
        float(summary.get("log_loss", 999.0) or 999.0),
        float(summary.get("rps", 999.0) or 999.0),
        -float(summary.get("accuracy_pick_max", 0.0) or 0.0),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "One-command operational validation: data quality, walk-forward backtests, "
            "quality gates, optional historical odds value backtest, and final model training."
        )
    )
    source_group = p.add_argument_group("input")
    source_group.add_argument("--matches", default=None, help="Canonical Mundialytics match CSV.")
    source_group.add_argument("--source", choices=["football-data-uk", "international-results", "openfootball"], default=None, help="Raw source format to convert first.")
    source_group.add_argument("--input", default=None, help="Raw source file for --source.")
    source_group.add_argument("--season", default=None)
    source_group.add_argument("--competition", default=None)
    source_group.add_argument("--team-scope", choices=["club", "national"], default="club")

    p.add_argument("--odds", default=None, help="Historical/current odds CSV compatible with match_id/fixture_id. Optional.")
    p.add_argument("--auto-football-data-odds", action="store_true", help="If --source football-data-uk, extract 1X2 odds from the same raw CSV.")
    p.add_argument("--fixtures", default=None, help="Optional upcoming fixtures CSV. If provided, final model predicts it.")
    p.add_argument("--fixture-odds", default=None, help="Optional odds CSV for upcoming fixtures. If provided, creates value picks.")
    p.add_argument("--out-dir", default="outputs/operational_validation")
    p.add_argument("--min-train-matches", type=int, default=50)
    p.add_argument("--retrain-every", type=int, default=10)
    p.add_argument("--model-types", nargs="+", choices=["poisson", "random_forest_lambda"], default=["poisson", "random_forest_lambda"])
    p.add_argument("--start-date", default=None, help="Optional lower date bound. For international-results, defaults to 2010-01-01 unless --full-history is set.")
    p.add_argument("--end-date", default=None, help="Optional upper date bound.")
    p.add_argument("--max-completed-matches", type=int, default=3000, help="Keep only the most recent N completed matches after filtering. Use --full-history to disable.")
    p.add_argument("--max-backtest-predictions", type=int, default=1200, help="Cap the evaluation window for runtime-safe walk-forward validation.")
    p.add_argument("--full-history", action="store_true", help="Disable default modern-window and max-match filters. Can be slow.")
    p.add_argument("--rf-n-estimators", type=int, default=250)
    p.add_argument("--rf-min-samples-leaf", type=int, default=6)
    p.add_argument("--poisson-alpha", type=float, default=1.0, help="Poisson regularization strength for goal-lambda model.")
    p.add_argument("--time-decay-half-life-days", type=float, default=365.0, help="Recency half-life for goal-model sample weights. Use <=0 to disable.")
    p.add_argument("--rolling-shrinkage-prior-matches", type=float, default=10.0, help="Shrink rolling team features toward global medians for low-sample teams. Use <=0 to disable.")
    p.add_argument("--min-matches-ready", type=int, default=200)
    p.add_argument("--min-backtest-predictions-ready", type=int, default=100)
    p.add_argument("--min-edge", type=float, default=0.03)
    p.add_argument("--min-ev", type=float, default=0.03)
    p.add_argument("--stake", type=float, default=1.0)
    args = p.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "started",
        "outputs_dir": str(out_dir),
        "recommendation": "Do not use real money unless readiness and long paper tracking are positive.",
    }

    matches = _load_or_build_matches(args, out_dir)
    matches, filter_info = _apply_operational_filters(matches, args)

    foundation = prepare_match_dataset(
        matches,
        dataset_name=f"{args.source or 'canonical'}_historical_validation",
        drop_incomplete_goals=False,
    )
    matches = foundation.cleaned_matches
    foundation.feature_coverage.to_csv(out_dir / "match_dataset_feature_coverage.csv", index=False)
    foundation.quality_by_competition_season.to_csv(out_dir / "match_dataset_quality_by_competition_season.csv", index=False)
    foundation.anomalies.to_csv(out_dir / "match_dataset_anomalies.csv", index=False)
    foundation.dropped_rows.to_csv(out_dir / "match_dataset_dropped_rows.csv", index=False)
    _write_json(foundation.summary, out_dir / "match_dataset_foundation_report.json")

    (out_dir / "canonical_matches_filtered.csv").parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out_dir / "canonical_matches_filtered.csv", index=False)
    report["operational_filters"] = filter_info
    report["data_foundation"] = {
        "summary": foundation.summary,
        "summary_json": str(out_dir / "match_dataset_foundation_report.json"),
        "feature_coverage_csv": str(out_dir / "match_dataset_feature_coverage.csv"),
        "quality_by_competition_season_csv": str(out_dir / "match_dataset_quality_by_competition_season.csv"),
        "anomalies_csv": str(out_dir / "match_dataset_anomalies.csv"),
        "dropped_rows_csv": str(out_dir / "match_dataset_dropped_rows.csv"),
    }
    scope = infer_single_scope(matches)
    completed = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    report["scope"] = scope
    report["n_completed_matches"] = int(len(completed))

    quality_full = data_quality_report(matches)
    quality_completed = data_quality_report(completed)
    _write_json(quality_full, out_dir / "data_quality_report_full_input.json")
    _write_json(quality_completed, out_dir / "data_quality_report_completed_only.json")
    report["data_quality_full_input"] = quality_full
    report["data_quality_completed_only"] = quality_completed

    if len(completed) <= args.min_train_matches:
        report["status"] = "failed_not_enough_matches"
        report["error"] = f"Need more than --min-train-matches={args.min_train_matches}; got {len(completed)} completed matches."
        _write_json(report, out_dir / "operational_validation_report.json")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        sys.exit(2)

    backtests: dict[str, Any] = {}
    quality_gates: dict[str, Any] = {}
    for model_type in args.model_types:
        pred, summary = walk_forward_backtest(
            completed,
            BacktestConfig(
                min_train_matches=args.min_train_matches,
                model_type=model_type,
                retrain_every=args.retrain_every,
                max_test_matches=args.max_backtest_predictions,
                rf_n_estimators=args.rf_n_estimators,
                rf_min_samples_leaf=args.rf_min_samples_leaf,
                poisson_alpha=args.poisson_alpha,
                time_decay_half_life_days=args.time_decay_half_life_days if args.time_decay_half_life_days > 0 else None,
                rolling_shrinkage_prior_matches=args.rolling_shrinkage_prior_matches,
            ),
        )
        pred_path = out_dir / f"backtest_{model_type}.csv"
        sum_path = out_dir / f"backtest_{model_type}_summary.json"
        pred.to_csv(pred_path, index=False)
        _write_json(summary, sum_path)

        (
            goal_error_metrics,
            goal_line_evaluation,
            goal_line_calibration_bins,
            scoreline_evaluation,
            calibration_detail,
            dixon_coles_scorelines,
            statistical_summary,
        ) = evaluate_statistical_engine(pred)
        goal_error_path = out_dir / f"statistical_engine_goal_errors_{model_type}.csv"
        goal_line_path = out_dir / f"statistical_engine_goal_lines_{model_type}.csv"
        line_calibration_path = out_dir / f"statistical_engine_line_calibration_{model_type}.csv"
        scoreline_path = out_dir / f"statistical_engine_scorelines_{model_type}.csv"
        calibration_path = out_dir / f"statistical_engine_calibration_layer_{model_type}.csv"
        dixon_coles_path = out_dir / f"statistical_engine_dixon_coles_scorelines_{model_type}.csv"
        statistical_summary_path = out_dir / f"statistical_engine_{model_type}_summary.json"
        goal_error_metrics.to_csv(goal_error_path, index=False)
        goal_line_evaluation.to_csv(goal_line_path, index=False)
        goal_line_calibration_bins.to_csv(line_calibration_path, index=False)
        scoreline_evaluation.to_csv(scoreline_path, index=False)
        calibration_detail.to_csv(calibration_path, index=False)
        dixon_coles_scorelines.to_csv(dixon_coles_path, index=False)
        _write_json(statistical_summary, statistical_summary_path)

        backtests[model_type] = {
            "summary": summary,
            "predictions_csv": str(pred_path),
            "summary_json": str(sum_path),
            "statistical_engine_evaluation": {
                "summary": statistical_summary,
                "summary_json": str(statistical_summary_path),
                "goal_errors_csv": str(goal_error_path),
                "goal_lines_csv": str(goal_line_path),
                "line_calibration_csv": str(line_calibration_path),
                "scorelines_csv": str(scoreline_path),
                "calibration_layer_csv": str(calibration_path),
                "dixon_coles_scorelines_csv": str(dixon_coles_path),
            },
        }

        gate = evaluate_readiness(
            quality_completed,
            summary,
            ReadinessThresholds(
                min_matches=args.min_matches_ready,
                min_backtest_predictions=args.min_backtest_predictions_ready,
            ),
        )
        gate_path = out_dir / f"quality_gate_{model_type}.json"
        _write_json(gate, gate_path)
        quality_gates[model_type] = {"result": gate, "json": str(gate_path)}

    best_model_type = min(backtests, key=lambda mt: _model_rank_key(backtests[mt]["summary"]))
    report["best_model_type"] = best_model_type
    report["backtests"] = backtests
    report["quality_gates"] = quality_gates

    model_path = out_dir / f"final_{scope}_{best_model_type}_model.pkl"
    metadata = _train_final_bundle(
        completed,
        best_model_type,
        args.source or "canonical_csv",
        model_path,
        poisson_alpha=args.poisson_alpha,
        time_decay_half_life_days=args.time_decay_half_life_days if args.time_decay_half_life_days > 0 else None,
        rolling_shrinkage_prior_matches=args.rolling_shrinkage_prior_matches,
    )
    _write_json(metadata, out_dir / "final_model_metadata.json")
    report["final_model"] = {"path": str(model_path), "metadata": metadata}

    odds_path = _resolve(args.odds)
    if odds_path is None and args.auto_football_data_odds:
        if args.source != "football-data-uk" or not args.input:
            raise ValueError("--auto-football-data-odds requires --source football-data-uk --input file.csv")
        odds = football_data_uk_to_match_odds(_resolve(args.input))
        odds_path = out_dir / "historical_1x2_odds.csv"
        odds.to_csv(odds_path, index=False)
    elif odds_path is not None:
        odds = pd.read_csv(odds_path)
    else:
        odds = None

    if odds is not None and not odds.empty:
        best_pred = pd.read_csv(backtests[best_model_type]["predictions_csv"])
        value_rows, value_summary = run_match_value_backtest(
            best_pred, odds, min_edge=args.min_edge, min_ev=args.min_ev, stake=args.stake
        )
        value_rows.to_csv(out_dir / "historical_value_backtest.csv", index=False)
        _write_json(value_summary, out_dir / "historical_value_backtest_summary.json")
        report["historical_value_backtest"] = {
            "picks_csv": str(out_dir / "historical_value_backtest.csv"),
            "summary_json": str(out_dir / "historical_value_backtest_summary.json"),
            "summary": value_summary,
        }
    elif odds is not None and odds.empty:
        report["historical_value_backtest"] = {"warning": "Odds file/extraction produced zero usable rows."}

    # Optional upcoming-fixture path is intentionally delegated to existing CLI scripts.
    # This keeps this validation script focused on historical trustworthiness.
    if args.fixtures:
        report["next_step_fixture_prediction"] = (
            f"Run: python scripts/predict_fixtures.py --bundle {model_path} --fixtures {args.fixtures} "
            f"--out {out_dir / 'upcoming_predictions.csv'}"
        )
        if args.fixture_odds:
            report["next_step_fixture_value"] = (
                f"Then run: python scripts/value_from_predictions.py --predictions {out_dir / 'upcoming_predictions.csv'} "
                f"--odds {args.fixture_odds} --out {out_dir / 'upcoming_value_picks.csv'}"
            )

    best_gate = quality_gates[best_model_type]["result"]
    if best_gate["passed"]:
        report["status"] = "ready_for_extended_paper_mode"
        report["recommendation"] = "Use this model for paper tracking on upcoming matches. Keep stakes virtual until long-run paper ROI and calibration are positive."
    else:
        report["status"] = "not_ready_keep_paper_only"
        report["recommendation"] = "The pipeline works, but failed readiness checks. Use outputs to diagnose; do not stake real money."

    _write_json(report, out_dir / "operational_validation_report.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
