from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from mundialytics.statistical_core.evaluation import (
    TemporalEvaluationConfig,
    _evaluation_summary,
    _score_match_predictions,
    build_calibration_bins,
    fit_calibration_summary,
    temporal_train_test_split,
)
from mundialytics.statistical_core.match_model import MatchOutcomeModel, _team_match_goal_frame
from mundialytics.statistical_core.schemas import write_json


def default_experiment_configs() -> list[dict[str, Any]]:
    """Return a small, deliberate model-hardening grid.

    The grid is intentionally compact because each trial performs a real
    temporal backtest over the full event file. The variants target the failure
    modes discovered in v0.22: overconfident 1X2 probabilities, extreme goal
    lambdas, and weak low-sample/fallback handling.
    """

    configs: list[tuple[str, dict[str, Any]]] = [
        ("baseline_v022", {}),
        ("cap4_shrink6_low8_draw005", {"goal_cap": 4.0, "attack_cap": 2.2, "defense_cap": 2.2, "profile_shrinkage_k": 6.0, "low_sample_blend_k": 8.0, "rating_clip": 0.28, "rating_divisor": 1000.0, "draw_lambda_blend": 0.05}),
        ("cap35_shrink10_low12_draw005", {"goal_cap": 3.5, "attack_cap": 2.0, "defense_cap": 2.0, "profile_shrinkage_k": 10.0, "low_sample_blend_k": 12.0, "rating_clip": 0.24, "rating_divisor": 1100.0, "draw_lambda_blend": 0.05}),
        ("cap4_shrink15_low15_draw010", {"goal_cap": 4.0, "attack_cap": 2.0, "defense_cap": 2.0, "profile_shrinkage_k": 15.0, "low_sample_blend_k": 15.0, "rating_clip": 0.22, "rating_divisor": 1200.0, "draw_lambda_blend": 0.10}),
        ("cap45_shrink10_low10_rating_soft", {"goal_cap": 4.5, "attack_cap": 2.15, "defense_cap": 2.15, "profile_shrinkage_k": 10.0, "low_sample_blend_k": 10.0, "rating_clip": 0.22, "rating_divisor": 1300.0, "rating_coefficient": 70.0, "draw_lambda_blend": 0.05}),
        ("cap4_shrink25_low20_draw010", {"goal_cap": 4.0, "attack_cap": 1.9, "defense_cap": 1.9, "profile_shrinkage_k": 25.0, "low_sample_blend_k": 20.0, "rating_clip": 0.20, "rating_divisor": 1300.0, "rating_coefficient": 65.0, "draw_lambda_blend": 0.10}),
        ("recency_fast_cap4", {"goal_cap": 4.0, "attack_cap": 2.1, "defense_cap": 2.1, "profile_shrinkage_k": 8.0, "low_sample_blend_k": 10.0, "recency_half_life_days": 180.0, "rating_clip": 0.24, "rating_divisor": 1150.0, "draw_lambda_blend": 0.05}),
        ("recency_slow_cap4", {"goal_cap": 4.0, "attack_cap": 2.1, "defense_cap": 2.1, "profile_shrinkage_k": 8.0, "low_sample_blend_k": 10.0, "recency_half_life_days": 730.0, "rating_clip": 0.24, "rating_divisor": 1150.0, "draw_lambda_blend": 0.05}),
        ("strong_draw_regularization", {"goal_cap": 4.0, "attack_cap": 2.0, "defense_cap": 2.0, "profile_shrinkage_k": 12.0, "low_sample_blend_k": 12.0, "rating_clip": 0.20, "rating_divisor": 1300.0, "draw_lambda_blend": 0.18}),
        ("light_hardening", {"goal_cap": 5.0, "attack_cap": 2.3, "defense_cap": 2.3, "profile_shrinkage_k": 4.0, "low_sample_blend_k": 5.0, "rating_clip": 0.30, "rating_divisor": 1000.0, "draw_lambda_blend": 0.03}),

        ("dc_draw_light", {"goal_cap": 4.0, "attack_cap": 2.2, "defense_cap": 2.2, "profile_shrinkage_k": 6.0, "low_sample_blend_k": 8.0, "rating_clip": 0.28, "rating_divisor": 1000.0, "draw_lambda_blend": 0.03, "dixon_coles_rho": -0.04}),
        ("dc_draw_medium", {"goal_cap": 4.0, "attack_cap": 2.1, "defense_cap": 2.1, "profile_shrinkage_k": 8.0, "low_sample_blend_k": 10.0, "rating_clip": 0.24, "rating_divisor": 1150.0, "draw_lambda_blend": 0.05, "dixon_coles_rho": -0.08}),
        ("dc_draw_strong", {"goal_cap": 3.8, "attack_cap": 2.0, "defense_cap": 2.0, "profile_shrinkage_k": 12.0, "low_sample_blend_k": 12.0, "rating_clip": 0.22, "rating_divisor": 1250.0, "draw_lambda_blend": 0.08, "dixon_coles_rho": -0.12}),
        ("dc_light_hardening", {"goal_cap": 5.0, "attack_cap": 2.3, "defense_cap": 2.3, "profile_shrinkage_k": 4.0, "low_sample_blend_k": 5.0, "rating_clip": 0.30, "rating_divisor": 1000.0, "draw_lambda_blend": 0.03, "dixon_coles_rho": -0.05}),
    ]
    return [{"trial_name": name, "model_config": cfg} for name, cfg in configs]


def run_model_lab(
    historical_events: pd.DataFrame,
    out_dir: str | Path,
    n_trials: int | None = None,
    test_fraction: float = 0.25,
    min_train_matches: int = 50,
    calibration_bins: int = 10,
    max_test_matches: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    configs = default_experiment_configs()
    if n_trials is not None:
        configs = configs[: max(1, int(n_trials))]

    # Expensive event-to-team aggregation is performed once; every trial then
    # fits from this compact match/team frame. This is the core agent-mode speedup.
    team_goal_frame = _team_match_goal_frame(historical_events)
    match_results = _build_match_results_from_team_goal_frame(team_goal_frame)

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_score = float("inf")

    for i, item in enumerate(configs, start=1):
        trial_id = f"trial_{i:03d}"
        trial_name = str(item["trial_name"])
        model_config = dict(item["model_config"])
        try:
            cfg = TemporalEvaluationConfig(
                test_fraction=test_fraction,
                min_train_matches=min_train_matches,
                calibration_bins=calibration_bins,
                model_config=model_config,
            )
            predictions, bins, summary, calibration = _evaluate_from_precomputed_frames(match_results, team_goal_frame, cfg, max_test_matches=max_test_matches)
            metrics = summary.get("metrics", {}) or {}
            diagnostics = summary.get("diagnostics", {}) or {}
            one = calibration.get("1x2", {}) or {}
            over = calibration.get("over_25", {}) or {}
            btts = calibration.get("btts", {}) or {}
            calibrated_1x2 = _finite(one.get("calibrated_log_loss"), default=metrics.get("log_loss_1x2", float("inf")))
            calibrated_over = _finite(over.get("calibrated_log_loss"), default=metrics.get("log_loss_over25", float("inf")))
            calibrated_btts = _finite(btts.get("calibrated_log_loss"), default=metrics.get("log_loss_btts", float("inf")))
            high_conf_rows = int(diagnostics.get("high_confidence_rows_p_ge_0_80") or 0)
            high_conf_acc = diagnostics.get("high_confidence_accuracy")
            high_conf_penalty = 0.0
            if high_conf_rows >= 25 and high_conf_acc is not None and float(high_conf_acc) < 0.65:
                high_conf_penalty = 0.03
            objective = calibrated_1x2 + 0.08 * calibrated_over + 0.08 * calibrated_btts + high_conf_penalty
            row = {
                "trial_id": trial_id,
                "trial_name": trial_name,
                "objective": objective,
                "raw_log_loss_1x2": metrics.get("log_loss_1x2"),
                "calibrated_log_loss_1x2": calibrated_1x2,
                "accuracy_pick_max": metrics.get("accuracy_pick_max"),
                "raw_log_loss_over25": metrics.get("log_loss_over25"),
                "calibrated_log_loss_over25": calibrated_over,
                "raw_log_loss_btts": metrics.get("log_loss_btts"),
                "calibrated_log_loss_btts": calibrated_btts,
                "mean_actual_outcome_probability": metrics.get("mean_actual_outcome_probability"),
                "alpha_1x2": one.get("alpha"),
                "alpha_over25": over.get("alpha"),
                "alpha_btts": btts.get("alpha"),
                "high_confidence_rows_p_ge_0_80": diagnostics.get("high_confidence_rows_p_ge_0_80"),
                "high_confidence_accuracy": diagnostics.get("high_confidence_accuracy"),
                "extreme_confidence_rows_p_ge_0_90": diagnostics.get("extreme_confidence_rows_p_ge_0_90"),
                "extreme_confidence_accuracy": diagnostics.get("extreme_confidence_accuracy"),
                "lambda_home_max": diagnostics.get("lambda_home_max"),
                "lambda_away_max": diagnostics.get("lambda_away_max"),
                "model_config_json": json.dumps(model_config, sort_keys=True),
                "status": summary.get("status"),
            }
            rows.append(row)
            trial_dir = out / "trials" / trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            summary_path = trial_dir / "match_evaluation_summary.json"
            calibration_path = trial_dir / "match_calibration_model.json"
            write_json(summary_path, summary)
            write_json(calibration_path, calibration)
            if not bins.empty:
                bins.to_csv(trial_dir / "match_calibration_bins.csv", index=False)
            if objective < best_score:
                best_score = objective
                best_payload = {
                    "trial_id": trial_id,
                    "trial_name": trial_name,
                    "objective": objective,
                    "model_config": model_config,
                    "calibration_model": calibration,
                    "summary": summary,
                    "artifact_paths": {"summary": str(summary_path), "calibration_model": str(calibration_path)},
                }
        except Exception as exc:  # pragma: no cover - written for robustness in long lab runs
            failed.append({"trial_id": trial_id, "trial_name": trial_name, "error": str(exc), "traceback": traceback.format_exc(), "model_config": model_config})

    leaderboard = pd.DataFrame(rows).sort_values("objective", ascending=True).reset_index(drop=True) if rows else pd.DataFrame()
    if not leaderboard.empty:
        leaderboard.to_csv(out / "experiment_leaderboard.csv", index=False)
    write_json(out / "failed_experiments.json", failed)
    if best_payload is None:
        best_payload = {"status": "no_successful_trials", "failed_experiments": failed}
    else:
        best_payload["status"] = "completed"
        best_payload["version"] = "v0.26_statistical_upgrade_model_lab"
    write_json(out / "best_model_config.json", best_payload)
    if best_payload.get("calibration_model"):
        write_json(out / "best_calibration_model.json", best_payload["calibration_model"])
    report = build_model_lab_report(out / "model_lab_report.html", leaderboard, best_payload, failed)
    best_payload["report"] = str(report)
    write_json(out / "model_lab_audit.json", {"status": "completed" if rows else "failed", "trials_run": len(rows), "trials_failed": len(failed), "best_trial": best_payload.get("trial_id"), "report": str(report)})
    return leaderboard, best_payload


def build_model_lab_report(path: str | Path, leaderboard: pd.DataFrame, best_payload: dict[str, Any], failed: list[dict[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = {k: v for k, v in best_payload.items() if k not in {"summary", "calibration_model"}}
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>Mundialytics Model Lab v0.26</title>"]
    html.append("<style>body{font-family:Arial,sans-serif;margin:28px} table{border-collapse:collapse;width:100%;font-size:13px} th,td{border:1px solid #ddd;padding:6px} th{background:#f4f4f4}.warn{background:#fff4d6;border:1px solid #d7aa28;padding:10px}</style></head><body>")
    html.append("<h1>Mundialytics Model Lab v0.26</h1>")
    html.append("<p>Automatic hardening loop for match model calibration, lambda caps, shrinkage, Dixon-Coles draw correction and overconfidence diagnostics.</p>")
    html.append("<h2>Best trial</h2><pre>" + _escape(json.dumps(best, indent=2, ensure_ascii=False)) + "</pre>")
    if not leaderboard.empty:
        cols = [c for c in ["trial_id", "trial_name", "objective", "raw_log_loss_1x2", "calibrated_log_loss_1x2", "accuracy_pick_max", "calibrated_log_loss_over25", "calibrated_log_loss_btts", "high_confidence_rows_p_ge_0_80", "high_confidence_accuracy", "lambda_home_max", "lambda_away_max"] if c in leaderboard.columns]
        html.append("<h2>Leaderboard</h2>")
        html.append(leaderboard[cols].to_html(index=False, classes="data-table", float_format=lambda x: f"{x:.6f}"))
    if failed:
        html.append("<h2>Failed experiments</h2><div class='warn'><pre>" + _escape(json.dumps(failed, indent=2, ensure_ascii=False)[:10000]) + "</pre></div>")
    html.append("</body></html>")
    out.write_text("\n".join(html), encoding="utf-8")
    return out


def _build_match_results_from_team_goal_frame(team_rows: pd.DataFrame) -> pd.DataFrame:
    if team_rows is None or team_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for match_id, g in team_rows.groupby("match_id", dropna=False):
        g = g.dropna(subset=["team"]).copy()
        g = g.sort_values(["team", "opponent"]).drop_duplicates(subset=["team"], keep="first")
        if len(g) < 2:
            continue
        teams = sorted(g["team"].astype(str).unique().tolist())[:2]
        a_row = g[g["team"].astype(str).eq(teams[0])].iloc[0]
        b_row = g[g["team"].astype(str).eq(teams[1])].iloc[0]
        date = pd.to_datetime(a_row.get("date"), errors="coerce")
        if pd.isna(date):
            date = pd.to_datetime(b_row.get("date"), errors="coerce")
        rows.append({
            "match_id": str(match_id),
            "date": date,
            "home_team": str(teams[0]),
            "away_team": str(teams[1]),
            "actual_home_goals": int(round(float(a_row.get("goals_for", 0)))),
            "actual_away_goals": int(round(float(b_row.get("goals_for", 0)))),
            "neutral": 1,
            "competition": "historical_eval",
            "stage": "historical_eval",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values(["date", "match_id"]).reset_index(drop=True)
    return out


def _evaluate_from_precomputed_frames(
    match_results: pd.DataFrame,
    team_goal_frame: pd.DataFrame,
    cfg: TemporalEvaluationConfig,
    max_test_matches: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    cutoff, train_matches, test_matches = temporal_train_test_split(match_results, cfg)
    if cutoff is None or train_matches.empty or test_matches.empty:
        summary = {
            "status": "not_enough_historical_matches_for_temporal_evaluation",
            "matches_available": int(len(match_results)),
            "min_train_matches": int(cfg.min_train_matches),
        }
        return pd.DataFrame(), pd.DataFrame(), summary, {}
    if max_test_matches is not None and int(max_test_matches) > 0 and len(test_matches) > int(max_test_matches):
        test_matches = test_matches.tail(int(max_test_matches)).copy()
    train_goal_frame = team_goal_frame[pd.to_datetime(team_goal_frame["date"], errors="coerce") < cutoff].copy()
    model_kwargs = dict(cfg.model_config or {})
    model_kwargs.setdefault("max_goals", cfg.max_goals)
    model = MatchOutcomeModel(**model_kwargs).fit_team_goal_frame(train_goal_frame)
    pred, _ = model.predict_fixtures(test_matches[["match_id", "date", "home_team", "away_team", "neutral", "competition", "stage"]])
    scored = test_matches.merge(pred, on=["match_id", "date", "home_team", "away_team", "neutral", "competition", "stage"], how="left")
    scored = _score_match_predictions(scored)
    bins = build_calibration_bins(scored, n_bins=cfg.calibration_bins)
    calibration = fit_calibration_summary(scored)
    summary = _evaluation_summary(scored, train_matches, test_matches, cutoff, model.audit, calibration)
    return scored, bins, summary, calibration


def _finite(value: Any, default: Any = float("inf")) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float(default)
    return v if math.isfinite(v) else float(default)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
