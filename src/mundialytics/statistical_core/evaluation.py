from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mundialytics.statistical_core.match_model import MatchOutcomeModel, _team_match_goal_frame
from mundialytics.statistical_core.schemas import canonical_name, standardize_fixtures, write_json


@dataclass(frozen=True)
class TemporalEvaluationConfig:
    """Configuration for auditable temporal holdout evaluation.

    v0.22 deliberately starts with a single chronological train/test split. It
    is less fancy than rolling-origin validation, but it is fast, reproducible,
    and catches the most important failure mode for betting: overconfident
    probabilities that look good only in current examples.
    """

    test_fraction: float = 0.25
    min_train_matches: int = 20
    max_goals: int = 10
    calibration_bins: int = 10
    model_config: dict[str, Any] | None = None


def build_historical_match_results(historical_events: pd.DataFrame | None) -> pd.DataFrame:
    """Build one row per historical match from the processed event dataset.

    The source file is event/player oriented, so home/away labels may not be
    trustworthy. For evaluation we create a deterministic neutral fixture by
    pairing the two teams in each match. The labels are therefore team_a/team_b,
    but the probability scoring is still valid: the model assigns probability
    to the actual team_a win / draw / team_b win outcome.
    """

    team_rows = _team_match_goal_frame(historical_events)
    if team_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for match_id, g in team_rows.groupby("match_id", dropna=False):
        g = g.dropna(subset=["team"]).copy()
        # Keep one row per team inside the match. Some raw event files may have
        # duplicate team/opponent pairs after aggregation quirks; collapse them.
        g = g.sort_values(["team", "opponent"]).drop_duplicates(subset=["team"], keep="first")
        if len(g) < 2:
            continue
        teams = sorted(g["team"].astype(str).unique().tolist())[:2]
        a_row = g[g["team"].astype(str).eq(teams[0])].iloc[0]
        b_row = g[g["team"].astype(str).eq(teams[1])].iloc[0]
        date = pd.to_datetime(a_row.get("date"), errors="coerce")
        if pd.isna(date):
            date = pd.to_datetime(b_row.get("date"), errors="coerce")
        rows.append(
            {
                "match_id": str(match_id),
                "date": date,
                "home_team": canonical_name(teams[0]),
                "away_team": canonical_name(teams[1]),
                "actual_home_goals": int(round(float(a_row.get("goals_for", 0)))),
                "actual_away_goals": int(round(float(b_row.get("goals_for", 0)))),
                "neutral": 1,
                "competition": "historical_eval",
                "stage": "historical_eval",
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values(["date", "match_id"]).reset_index(drop=True)
    return out


def temporal_train_test_split(match_results: pd.DataFrame, cfg: TemporalEvaluationConfig) -> tuple[pd.Timestamp | None, pd.DataFrame, pd.DataFrame]:
    if match_results is None or match_results.empty or "date" not in match_results.columns:
        return None, pd.DataFrame(), pd.DataFrame()
    work = match_results.dropna(subset=["date"]).sort_values(["date", "match_id"]).reset_index(drop=True)
    if len(work) <= max(2, cfg.min_train_matches):
        return None, pd.DataFrame(), pd.DataFrame()
    test_n = max(1, int(round(len(work) * float(cfg.test_fraction))))
    split_idx = max(cfg.min_train_matches, len(work) - test_n)
    if split_idx >= len(work):
        split_idx = len(work) - 1
    cutoff = pd.Timestamp(work.iloc[split_idx]["date"])
    train_matches = work.iloc[:split_idx].copy()
    test_matches = work.iloc[split_idx:].copy()
    return cutoff, train_matches, test_matches


def evaluate_match_model_temporal(
    historical_events: pd.DataFrame | None,
    cfg: TemporalEvaluationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Fit on past data, predict future holdout matches, score probabilities."""

    cfg = cfg or TemporalEvaluationConfig()
    match_results = build_historical_match_results(historical_events)
    cutoff, train_matches, test_matches = temporal_train_test_split(match_results, cfg)
    if cutoff is None or train_matches.empty or test_matches.empty:
        summary = {
            "status": "not_enough_historical_matches_for_temporal_evaluation",
            "matches_available": int(len(match_results)),
            "min_train_matches": int(cfg.min_train_matches),
        }
        return pd.DataFrame(), pd.DataFrame(), summary, {}

    historical_events = historical_events.copy() if historical_events is not None else pd.DataFrame()
    if "date" in historical_events.columns:
        event_dates = pd.to_datetime(historical_events["date"], errors="coerce")
        train_events = historical_events[event_dates < cutoff].copy()
    else:
        train_events = historical_events.copy()
    model_kwargs = dict(cfg.model_config or {})
    model_kwargs.setdefault("max_goals", cfg.max_goals)
    model = MatchOutcomeModel(**model_kwargs).fit(train_events)
    pred, _ = model.predict_fixtures(test_matches[["match_id", "date", "home_team", "away_team", "neutral", "competition", "stage"]])
    scored = test_matches.merge(pred, on=["match_id", "date", "home_team", "away_team", "neutral", "competition", "stage"], how="left")
    scored = _score_match_predictions(scored)
    bins = build_calibration_bins(scored, n_bins=cfg.calibration_bins)
    calibration = fit_calibration_summary(scored)
    summary = _evaluation_summary(scored, train_matches, test_matches, cutoff, model.audit, calibration)
    return scored, bins, summary, calibration


def _score_match_predictions(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored
    out = scored.copy()
    hg = pd.to_numeric(out["actual_home_goals"], errors="coerce").fillna(0).astype(int)
    ag = pd.to_numeric(out["actual_away_goals"], errors="coerce").fillna(0).astype(int)
    out["actual_outcome"] = np.where(hg > ag, "home", np.where(hg < ag, "away", "draw"))
    out["actual_over_25"] = ((hg + ag) > 2.5).astype(int)
    out["actual_btts"] = ((hg > 0) & (ag > 0)).astype(int)
    probs = out[["p_home_win", "p_draw", "p_away_win"]].astype(float).clip(1e-9, 1 - 1e-9)
    actual_idx = out["actual_outcome"].map({"home": "p_home_win", "draw": "p_draw", "away": "p_away_win"})
    out["actual_outcome_probability"] = [float(probs.loc[i, c]) for i, c in actual_idx.items()]
    out["log_loss_1x2_row"] = -np.log(out["actual_outcome_probability"].clip(1e-9, 1.0))
    y_home = out["actual_outcome"].eq("home").astype(float)
    y_draw = out["actual_outcome"].eq("draw").astype(float)
    y_away = out["actual_outcome"].eq("away").astype(float)
    out["brier_1x2_row"] = (probs["p_home_win"] - y_home) ** 2 + (probs["p_draw"] - y_draw) ** 2 + (probs["p_away_win"] - y_away) ** 2
    out["predicted_outcome"] = probs.idxmax(axis=1).map({"p_home_win": "home", "p_draw": "draw", "p_away_win": "away"})
    out["prediction_correct"] = out["predicted_outcome"].eq(out["actual_outcome"]).astype(int)
    out["log_loss_over25_row"] = _binary_logloss_row(out["actual_over_25"], out["p_over_25"])
    out["brier_over25_row"] = (out["p_over_25"].astype(float) - out["actual_over_25"].astype(float)) ** 2
    out["log_loss_btts_row"] = _binary_logloss_row(out["actual_btts"], out["p_btts"])
    out["brier_btts_row"] = (out["p_btts"].astype(float) - out["actual_btts"].astype(float)) ** 2
    return out


def _binary_logloss_row(y: pd.Series, p: pd.Series) -> pd.Series:
    yy = pd.to_numeric(y, errors="coerce").fillna(0).astype(float)
    pp = pd.to_numeric(p, errors="coerce").fillna(0.5).clip(1e-9, 1 - 1e-9).astype(float)
    return -(yy * np.log(pp) + (1.0 - yy) * np.log(1.0 - pp))


def build_calibration_bins(scored: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    if scored is None or scored.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    # Long 1X2 rows.
    long_rows = []
    for _, r in scored.iterrows():
        for sel, col in [("home", "p_home_win"), ("draw", "p_draw"), ("away", "p_away_win")]:
            long_rows.append({"market": "1x2", "selection": sel, "predicted_probability": float(r[col]), "actual": int(r["actual_outcome"] == sel)})
    long = pd.DataFrame(long_rows)
    rows.append(_bin_frame(long, n_bins))
    rows.append(_bin_frame(pd.DataFrame({"market": "over_25", "selection": "over", "predicted_probability": scored["p_over_25"], "actual": scored["actual_over_25"]}), n_bins))
    rows.append(_bin_frame(pd.DataFrame({"market": "btts", "selection": "yes", "predicted_probability": scored["p_btts"], "actual": scored["actual_btts"]}), n_bins))
    return pd.concat([r for r in rows if not r.empty], ignore_index=True) if rows else pd.DataFrame()


def _bin_frame(frame: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    f = frame.copy()
    f["predicted_probability"] = pd.to_numeric(f["predicted_probability"], errors="coerce").clip(0, 1)
    f["actual"] = pd.to_numeric(f["actual"], errors="coerce").fillna(0).astype(float)
    edges = np.linspace(0, 1, int(n_bins) + 1)
    labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
    f["probability_bin"] = pd.cut(f["predicted_probability"], bins=edges, labels=labels, include_lowest=True)
    out = (
        f.groupby(["market", "selection", "probability_bin"], observed=True)
        .agg(count=("actual", "size"), mean_predicted_probability=("predicted_probability", "mean"), empirical_frequency=("actual", "mean"))
        .reset_index()
    )
    out["calibration_error"] = (out["mean_predicted_probability"] - out["empirical_frequency"]).abs()
    return out


def fit_calibration_summary(scored: pd.DataFrame) -> dict[str, Any]:
    if scored is None or scored.empty:
        return {"status": "not_available"}
    one_x2 = _fit_1x2_shrinkage(scored)
    over25 = _fit_binary_shrinkage(scored["actual_over_25"], scored["p_over_25"], "over_25")
    btts = _fit_binary_shrinkage(scored["actual_btts"], scored["p_btts"], "btts")
    return {"status": "fitted_temporal_holdout", "method": "shrink_to_empirical_base_rate", "1x2": one_x2, "over_25": over25, "btts": btts}


def _fit_1x2_shrinkage(scored: pd.DataFrame) -> dict[str, Any]:
    y = pd.get_dummies(scored["actual_outcome"])[[c for c in ["home", "draw", "away"] if c in pd.get_dummies(scored["actual_outcome"]).columns]]
    for c in ["home", "draw", "away"]:
        if c not in y.columns:
            y[c] = 0
    y = y[["home", "draw", "away"]].to_numpy(dtype=float)
    p = scored[["p_home_win", "p_draw", "p_away_win"]].to_numpy(dtype=float)
    p = np.clip(p, 1e-9, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    base = y.mean(axis=0)
    if base.sum() <= 0:
        base = np.ones(3) / 3
    base = base / base.sum()
    best_alpha, best_loss = 1.0, float("inf")
    for alpha in np.linspace(0.1, 1.0, 19):
        pc = alpha * p + (1.0 - alpha) * base[None, :]
        pc = np.clip(pc, 1e-9, 1.0)
        pc = pc / pc.sum(axis=1, keepdims=True)
        loss = float(-(y * np.log(pc)).sum(axis=1).mean())
        if loss < best_loss:
            best_alpha, best_loss = float(alpha), loss
    raw_loss = float(-(y * np.log(p)).sum(axis=1).mean())
    return {
        "alpha": best_alpha,
        "base_rates": {"home": float(base[0]), "draw": float(base[1]), "away": float(base[2])},
        "raw_log_loss": raw_loss,
        "calibrated_log_loss": best_loss,
        "n": int(len(scored)),
    }


def _fit_binary_shrinkage(actual: pd.Series, pred: pd.Series, market: str) -> dict[str, Any]:
    y = pd.to_numeric(actual, errors="coerce").fillna(0).astype(float).to_numpy()
    p = pd.to_numeric(pred, errors="coerce").fillna(0.5).astype(float).clip(1e-9, 1 - 1e-9).to_numpy()
    base = float(np.mean(y)) if len(y) else 0.5
    best_alpha, best_loss = 1.0, float("inf")
    for alpha in np.linspace(0.1, 1.0, 19):
        pc = np.clip(alpha * p + (1.0 - alpha) * base, 1e-9, 1 - 1e-9)
        loss = float(-(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)).mean())
        if loss < best_loss:
            best_alpha, best_loss = float(alpha), loss
    raw_loss = float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())
    return {"market": market, "alpha": best_alpha, "base_rate": base, "raw_log_loss": raw_loss, "calibrated_log_loss": best_loss, "n": int(len(y))}


def apply_match_calibration(match_predictions: pd.DataFrame, calibration: dict[str, Any] | None) -> pd.DataFrame:
    """Apply fitted shrinkage calibration to current match predictions."""

    if match_predictions is None or match_predictions.empty or not calibration or calibration.get("status") not in {"fitted_temporal_holdout", "loaded"}:
        return match_predictions
    out = match_predictions.copy()
    one = calibration.get("1x2", {}) or {}
    if {"p_home_win", "p_draw", "p_away_win"}.issubset(out.columns) and "alpha" in one:
        alpha = float(one.get("alpha", 1.0))
        base_rates = one.get("base_rates", {}) or {}
        base = np.array([float(base_rates.get("home", 1 / 3)), float(base_rates.get("draw", 1 / 3)), float(base_rates.get("away", 1 / 3))], dtype=float)
        if base.sum() <= 0:
            base = np.ones(3) / 3
        base = base / base.sum()
        raw = out[["p_home_win", "p_draw", "p_away_win"]].astype(float).to_numpy()
        cal = alpha * raw + (1.0 - alpha) * base[None, :]
        cal = np.clip(cal, 1e-9, 1.0)
        cal = cal / cal.sum(axis=1, keepdims=True)
        out["p_home_win_raw"] = raw[:, 0]
        out["p_draw_raw"] = raw[:, 1]
        out["p_away_win_raw"] = raw[:, 2]
        out[["p_home_win", "p_draw", "p_away_win"]] = cal
        out["calibration_applied"] = "1x2_shrinkage"
    for col, key in [("p_over_25", "over_25"), ("p_btts", "btts")]:
        params = calibration.get(key, {}) or {}
        if col in out.columns and "alpha" in params:
            alpha = float(params.get("alpha", 1.0))
            base = float(params.get("base_rate", 0.5))
            raw = out[col].astype(float).clip(1e-9, 1 - 1e-9)
            out[f"{col}_raw"] = raw
            out[col] = (alpha * raw + (1.0 - alpha) * base).clip(1e-9, 1 - 1e-9)
            if col == "p_over_25":
                out["p_under_25"] = 1.0 - out[col]
    return out


def write_evaluation_artifacts(
    out_dir: str | Path,
    predictions: pd.DataFrame,
    bins: pd.DataFrame,
    summary: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    pred_path = out / "match_backtest_predictions.csv"
    bins_path = out / "match_calibration_bins.csv"
    summary_path = out / "match_evaluation_summary.json"
    calibration_path = out / "match_calibration_model.json"
    predictions.to_csv(pred_path, index=False)
    bins.to_csv(bins_path, index=False)
    write_json(summary_path, summary)
    write_json(calibration_path, calibration)
    files["match_backtest_predictions"] = str(pred_path)
    files["match_calibration_bins"] = str(bins_path)
    files["match_evaluation_summary"] = str(summary_path)
    files["match_calibration_model"] = str(calibration_path)
    return files


def load_calibration(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("status") == "fitted_temporal_holdout":
        data = data.copy()
        data["status"] = "loaded"
    return data if isinstance(data, dict) else {}


def _evaluation_summary(
    scored: pd.DataFrame,
    train_matches: pd.DataFrame,
    test_matches: pd.DataFrame,
    cutoff: pd.Timestamp,
    model_audit: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    if scored.empty:
        return {"status": "no_scored_predictions"}
    max_prob = scored[["p_home_win", "p_draw", "p_away_win"]].astype(float).max(axis=1)
    high_conf = max_prob >= 0.80
    extreme_conf = max_prob >= 0.90
    fallback_rows = scored.get("warnings", pd.Series("", index=scored.index)).astype(str).str.contains("fallback|low_team_match_sample|lambda_shrunk", regex=True, na=False)
    diagnostics = {
        "high_confidence_rows_p_ge_0_80": int(high_conf.sum()),
        "high_confidence_accuracy": float(scored.loc[high_conf, "prediction_correct"].mean()) if high_conf.any() else None,
        "extreme_confidence_rows_p_ge_0_90": int(extreme_conf.sum()),
        "extreme_confidence_accuracy": float(scored.loc[extreme_conf, "prediction_correct"].mean()) if extreme_conf.any() else None,
        "fallback_or_low_sample_rows": int(fallback_rows.sum()),
        "fallback_or_low_sample_mean_log_loss_1x2": float(scored.loc[fallback_rows, "log_loss_1x2_row"].mean()) if fallback_rows.any() else None,
        "lambda_home_max": float(pd.to_numeric(scored.get("lambda_home"), errors="coerce").max()),
        "lambda_away_max": float(pd.to_numeric(scored.get("lambda_away"), errors="coerce").max()),
    }
    return {
        "status": "completed",
        "version": "v0.26_evaluation_calibration",
        "cutoff_date": str(pd.Timestamp(cutoff).date()),
        "train_matches": int(len(train_matches)),
        "test_matches": int(len(test_matches)),
        "model_audit": model_audit,
        "metrics": {
            "log_loss_1x2": float(scored["log_loss_1x2_row"].mean()),
            "brier_1x2": float(scored["brier_1x2_row"].mean()),
            "accuracy_pick_max": float(scored["prediction_correct"].mean()),
            "log_loss_over25": float(scored["log_loss_over25_row"].mean()),
            "brier_over25": float(scored["brier_over25_row"].mean()),
            "log_loss_btts": float(scored["log_loss_btts_row"].mean()),
            "brier_btts": float(scored["brier_btts_row"].mean()),
            "mean_actual_outcome_probability": float(scored["actual_outcome_probability"].mean()),
        },
        "diagnostics": diagnostics,
        "calibration": calibration,
        "honest_limitations": [
            "Temporal holdout is useful but should be complemented with rolling-origin validation.",
            "Home/away labels may be deterministic team_a/team_b when source events lack venue metadata.",
            "Evaluation scores model probability quality; betting value still requires real odds and closing-line tracking.",
        ],
    }
