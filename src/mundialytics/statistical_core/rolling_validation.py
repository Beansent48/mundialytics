from __future__ import annotations

import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mundialytics.statistical_core.evaluation import (
    _evaluation_summary,
    _score_match_predictions,
    apply_match_calibration,
    build_historical_match_results,
    fit_calibration_summary,
)
from mundialytics.statistical_core.match_model import MatchOutcomeModel
from mundialytics.statistical_core.model_lab import default_experiment_configs
from mundialytics.statistical_core.schemas import write_json


@dataclass(frozen=True)
class RollingMatchConfig:
    """Rolling-origin validation for the statistical match model.

    Each fold uses only past data: train period -> calibration period -> test
    period. Calibration is fitted on the calibration period and then applied to
    the future test fold, so the reported calibrated scores are closer to how
    the model would behave in production than a single holdout optimized on one
    split.
    """

    min_train_matches: int = 300
    calibration_matches: int = 400
    test_matches: int = 250
    step_matches: int = 250
    max_folds: int | None = 6
    max_goals: int = 10
    calibration_bins: int = 10
    model_config: dict[str, Any] | None = None


def rolling_match_backtest(historical_events: pd.DataFrame, cfg: RollingMatchConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cfg = cfg or RollingMatchConfig()
    events = historical_events.copy() if historical_events is not None else pd.DataFrame()
    if "date" in events.columns:
        events["date"] = pd.to_datetime(events["date"], errors="coerce")
    match_results = build_historical_match_results(events)
    if match_results.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "no_match_results"}
    match_results = match_results.dropna(subset=["date"]).sort_values(["date", "match_id"]).reset_index(drop=True)
    folds = _build_match_count_folds(match_results, cfg)
    if not folds:
        return pd.DataFrame(), pd.DataFrame(), {"status": "not_enough_matches_for_rolling", "matches": int(len(match_results))}

    prediction_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    model_kwargs = dict(cfg.model_config or {})
    model_kwargs.setdefault("max_goals", cfg.max_goals)

    for i, fold in enumerate(folds, start=1):
        train = match_results.iloc[: fold["train_end"]].copy()
        cal = match_results.iloc[fold["cal_start"] : fold["cal_end"]].copy()
        test = match_results.iloc[fold["test_start"] : fold["test_end"]].copy()
        if train.empty or cal.empty or test.empty:
            continue
        train_cutoff = pd.Timestamp(cal.iloc[0]["date"])
        cal_cutoff = pd.Timestamp(test.iloc[0]["date"])
        if "date" in events.columns:
            train_events = events[events["date"] < train_cutoff].copy()
            train_cal_events = events[events["date"] < cal_cutoff].copy()
        else:
            train_events = events.copy()
            train_cal_events = events.copy()

        model = MatchOutcomeModel(**model_kwargs).fit(train_events)
        cal_pred, _ = model.predict_fixtures(cal[["match_id", "date", "home_team", "away_team", "neutral", "competition", "stage"]])
        cal_scored = cal.merge(cal_pred, on=["match_id", "date", "home_team", "away_team", "neutral", "competition", "stage"], how="left")
        cal_scored = _score_match_predictions(cal_scored)
        calibration = fit_calibration_summary(cal_scored)

        # Refit using train+cal before predicting the future test fold.
        model2 = MatchOutcomeModel(**model_kwargs).fit(train_cal_events)
        test_pred_raw, _ = model2.predict_fixtures(test[["match_id", "date", "home_team", "away_team", "neutral", "competition", "stage"]])
        raw_scored = test.merge(test_pred_raw, on=["match_id", "date", "home_team", "away_team", "neutral", "competition", "stage"], how="left")
        raw_scored = _score_match_predictions(raw_scored)
        test_pred_cal = apply_match_calibration(test_pred_raw, {**calibration, "status": "loaded"})
        cal_scored_test = test.merge(test_pred_cal, on=["match_id", "date", "home_team", "away_team", "neutral", "competition", "stage"], how="left")
        cal_scored_test = _score_match_predictions(cal_scored_test)
        cal_scored_test["fold"] = i
        cal_scored_test["fold_train_matches"] = len(train)
        cal_scored_test["fold_calibration_matches"] = len(cal)
        cal_scored_test["fold_test_matches"] = len(test)
        cal_scored_test["fold_train_end_date"] = str(pd.Timestamp(train.iloc[-1]["date"]).date())
        cal_scored_test["fold_test_start_date"] = str(pd.Timestamp(test.iloc[0]["date"]).date())
        cal_scored_test["fold_test_end_date"] = str(pd.Timestamp(test.iloc[-1]["date"]).date())
        prediction_parts.append(cal_scored_test)
        fold_rows.append(_fold_metric_row(i, train, cal, test, raw_scored, cal_scored_test, calibration))

    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    folds_df = pd.DataFrame(fold_rows)
    summary = _rolling_summary(predictions, folds_df, cfg)
    return predictions, folds_df, summary


def run_rolling_model_lab(
    historical_events: pd.DataFrame,
    out_dir: str | Path,
    n_trials: int | None = None,
    cfg: RollingMatchConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_cfg = cfg or RollingMatchConfig()
    configs = default_experiment_configs()
    if n_trials is not None:
        configs = configs[: max(1, int(n_trials))]
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_score = float("inf")
    for i, item in enumerate(configs, start=1):
        trial_id = f"trial_{i:03d}"
        trial_name = str(item.get("trial_name"))
        model_config = dict(item.get("model_config") or {})
        try:
            trial_cfg = RollingMatchConfig(
                min_train_matches=base_cfg.min_train_matches,
                calibration_matches=base_cfg.calibration_matches,
                test_matches=base_cfg.test_matches,
                step_matches=base_cfg.step_matches,
                max_folds=base_cfg.max_folds,
                max_goals=base_cfg.max_goals,
                calibration_bins=base_cfg.calibration_bins,
                model_config=model_config,
            )
            preds, folds, summary = rolling_match_backtest(historical_events, trial_cfg)
            metrics = summary.get("metrics", {}) or {}
            objective = _rolling_objective(metrics, summary.get("diagnostics", {}) or {})
            row = {
                "trial_id": trial_id,
                "trial_name": trial_name,
                "objective": objective,
                "folds": summary.get("folds"),
                "test_matches": summary.get("test_matches"),
                "calibrated_log_loss_1x2": metrics.get("calibrated_log_loss_1x2"),
                "raw_log_loss_1x2": metrics.get("raw_log_loss_1x2"),
                "accuracy_pick_max": metrics.get("accuracy_pick_max"),
                "calibrated_log_loss_over25": metrics.get("calibrated_log_loss_over25"),
                "calibrated_log_loss_btts": metrics.get("calibrated_log_loss_btts"),
                "high_confidence_accuracy": (summary.get("diagnostics", {}) or {}).get("high_confidence_accuracy"),
                "model_config_json": json.dumps(model_config, sort_keys=True),
                "status": summary.get("status", "completed"),
            }
            rows.append(row)
            trial_dir = out / "trials" / trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            preds.to_csv(trial_dir / "rolling_predictions.csv", index=False)
            folds.to_csv(trial_dir / "rolling_fold_metrics.csv", index=False)
            write_json(trial_dir / "rolling_summary.json", {**summary, "trial_id": trial_id, "trial_name": trial_name, "model_config": model_config})
            if objective < best_score:
                best_score = objective
                best_payload = {"trial_id": trial_id, "trial_name": trial_name, "objective": objective, "model_config": model_config, "summary": summary, "artifact_paths": {"predictions": str(trial_dir / "rolling_predictions.csv"), "folds": str(trial_dir / "rolling_fold_metrics.csv"), "summary": str(trial_dir / "rolling_summary.json")}}
        except Exception as exc:  # pragma: no cover - safety for long labs
            failed.append({"trial_id": trial_id, "trial_name": trial_name, "error": repr(exc), "traceback": traceback.format_exc()})
    leaderboard = pd.DataFrame(rows).sort_values("objective", na_position="last") if rows else pd.DataFrame()
    if not leaderboard.empty:
        leaderboard.to_csv(out / "rolling_model_leaderboard.csv", index=False)
    if best_payload is None:
        best_payload = {"status": "no_successful_trials", "failed": failed}
    else:
        best_payload["status"] = "completed"
        best_payload["version"] = "v0.27_rolling_model_lab"
    write_json(out / "best_rolling_model_config.json", best_payload)
    write_json(out / "failed_rolling_experiments.json", {"failed": failed})
    build_rolling_model_report(out / "rolling_model_report.html", leaderboard, best_payload, failed)
    best_payload["report"] = str(out / "rolling_model_report.html")
    return leaderboard, best_payload


def build_rolling_model_report(path: str | Path, leaderboard: pd.DataFrame, best_payload: dict[str, Any], failed: list[dict[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    def esc(x: Any) -> str:
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>Mundialytics Rolling Model Lab</title>"]
    html.append("<style>body{font-family:Arial,sans-serif;margin:28px;color:#111} table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:24px} th,td{border:1px solid #ddd;padding:6px} th{background:#f4f4f4} code{background:#eee;padding:2px 4px}</style></head><body>")
    html.append("<h1>Mundialytics v0.27 Rolling Match Model Lab</h1>")
    html.append("<p>Rolling-origin validation: train on past, calibrate on recent past, score future fold. This is stricter than a single holdout.</p>")
    html.append(f"<h2>Best trial</h2><pre>{esc(json.dumps(best_payload, indent=2, ensure_ascii=False, default=str))}</pre>")
    if not leaderboard.empty:
        cols = [c for c in ["trial_id","trial_name","objective","folds","test_matches","calibrated_log_loss_1x2","raw_log_loss_1x2","accuracy_pick_max","calibrated_log_loss_over25","calibrated_log_loss_btts","high_confidence_accuracy"] if c in leaderboard.columns]
        html.append("<h2>Leaderboard</h2>")
        html.append(leaderboard[cols].to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    if failed:
        html.append("<h2>Failed trials</h2><pre>" + esc(json.dumps(failed, indent=2, ensure_ascii=False, default=str)) + "</pre>")
    html.append("</body></html>")
    out.write_text("\n".join(html), encoding="utf-8")
    return out


def _build_match_count_folds(match_results: pd.DataFrame, cfg: RollingMatchConfig) -> list[dict[str, int]]:
    n = len(match_results)
    train = int(cfg.min_train_matches)
    cal = int(cfg.calibration_matches)
    test = int(cfg.test_matches)
    step = max(1, int(cfg.step_matches))
    folds = []
    start = train + cal
    while start + 1 < n:
        end = min(n, start + test)
        if end <= start:
            break
        folds.append({"train_end": start - cal, "cal_start": start - cal, "cal_end": start, "test_start": start, "test_end": end})
        start += step
    if cfg.max_folds is not None and len(folds) > cfg.max_folds:
        folds = folds[-int(cfg.max_folds):]
    return folds


def _fold_metric_row(i: int, train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame, raw_scored: pd.DataFrame, cal_scored: pd.DataFrame, calibration: dict[str, Any]) -> dict[str, Any]:
    return {
        "fold": i,
        "train_matches": int(len(train)),
        "calibration_matches": int(len(cal)),
        "test_matches": int(len(test)),
        "train_end_date": str(pd.Timestamp(train.iloc[-1]["date"]).date()),
        "calibration_start_date": str(pd.Timestamp(cal.iloc[0]["date"]).date()),
        "test_start_date": str(pd.Timestamp(test.iloc[0]["date"]).date()),
        "test_end_date": str(pd.Timestamp(test.iloc[-1]["date"]).date()),
        "raw_log_loss_1x2": float(raw_scored["log_loss_1x2_row"].mean()),
        "calibrated_log_loss_1x2": float(cal_scored["log_loss_1x2_row"].mean()),
        "accuracy_pick_max": float(cal_scored["prediction_correct"].mean()),
        "calibrated_log_loss_over25": float(cal_scored["log_loss_over25_row"].mean()),
        "calibrated_log_loss_btts": float(cal_scored["log_loss_btts_row"].mean()),
        "alpha_1x2": (calibration.get("1x2", {}) or {}).get("alpha"),
    }


def _rolling_summary(predictions: pd.DataFrame, folds: pd.DataFrame, cfg: RollingMatchConfig) -> dict[str, Any]:
    if predictions.empty:
        return {"status": "no_predictions"}
    max_prob = predictions[["p_home_win", "p_draw", "p_away_win"]].astype(float).max(axis=1)
    high = max_prob >= 0.80
    metrics = {
        "calibrated_log_loss_1x2": float(predictions["log_loss_1x2_row"].mean()),
        "brier_1x2": float(predictions["brier_1x2_row"].mean()),
        "accuracy_pick_max": float(predictions["prediction_correct"].mean()),
        "calibrated_log_loss_over25": float(predictions["log_loss_over25_row"].mean()),
        "brier_over25": float(predictions["brier_over25_row"].mean()),
        "calibrated_log_loss_btts": float(predictions["log_loss_btts_row"].mean()),
        "brier_btts": float(predictions["brier_btts_row"].mean()),
        "mean_actual_outcome_probability": float(predictions["actual_outcome_probability"].mean()),
    }
    if "p_home_win_raw" in predictions.columns:
        raw = predictions.copy()
        raw[["p_home_win", "p_draw", "p_away_win"]] = raw[["p_home_win_raw", "p_draw_raw", "p_away_win_raw"]].to_numpy()
        raw = _score_match_predictions(raw)
        metrics["raw_log_loss_1x2"] = float(raw["log_loss_1x2_row"].mean())
    diagnostics = {
        "high_confidence_rows_p_ge_0_80": int(high.sum()),
        "high_confidence_accuracy": float(predictions.loc[high, "prediction_correct"].mean()) if high.any() else None,
        "fold_log_loss_std_1x2": float(folds["calibrated_log_loss_1x2"].std()) if not folds.empty and "calibrated_log_loss_1x2" in folds else None,
    }
    return {
        "status": "completed",
        "version": "v0.27_rolling_match_evaluation",
        "folds": int(predictions["fold"].nunique()) if "fold" in predictions.columns else int(len(folds)),
        "test_matches": int(len(predictions)),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "fold_metrics": folds.to_dict(orient="records") if not folds.empty else [],
        "config": {k: v for k, v in cfg.__dict__.items() if k != "model_config"},
        "model_config": cfg.model_config or {},
        "honest_limitations": [
            "Rolling-origin validation is stricter than a single holdout but still depends on event coverage and deterministic team_a/team_b labels when venue is unknown.",
            "Calibration is fitted on the period immediately before each test fold and applied to future matches only.",
        ],
    }


def _rolling_objective(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> float:
    one = float(metrics.get("calibrated_log_loss_1x2", float("inf")))
    over = float(metrics.get("calibrated_log_loss_over25", 0.70))
    btts = float(metrics.get("calibrated_log_loss_btts", 0.70))
    volatility_penalty = 0.0
    std = diagnostics.get("fold_log_loss_std_1x2")
    if std is not None and np.isfinite(float(std)):
        volatility_penalty = 0.05 * float(std)
    return float(one + 0.08 * over + 0.08 * btts + volatility_penalty)
