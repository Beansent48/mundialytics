from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import math
import html
import pandas as pd


SIMULATION_EVALUATION_VERSION = "v0.48.4_simulation_evaluation_report"

OUTCOME_CLASSES: tuple[str, ...] = ("home_win", "draw", "away_win")

SIMULATION_EVALUATION_COLUMNS: tuple[str, ...] = (
    "match_id",
    "date",
    "competition",
    "home_team",
    "away_team",
    "actual_home_goals",
    "actual_away_goals",
    "actual_outcome",
    "predicted_outcome",
    "predicted_outcome_probability",
    "prediction_correct",
    "p_home_win",
    "p_draw",
    "p_away_win",
    "expected_home_goals",
    "expected_away_goals",
    "actual_total_goals",
    "expected_total_goals",
    "home_goal_error",
    "away_goal_error",
    "total_goal_error",
    "actual_scoreline",
    "actual_scoreline_probability",
    "actual_scoreline_rank",
    "actual_scoreline_top1",
    "actual_scoreline_top3",
    "actual_scoreline_top5",
    "data_quality_flag",
    "evaluation_note",
)

CALIBRATION_COLUMNS: tuple[str, ...] = (
    "outcome_class",
    "bin_lower",
    "bin_upper",
    "row_count",
    "avg_predicted_probability",
    "observed_frequency",
    "calibration_error",
    "status",
)

GOAL_ERROR_COLUMNS: tuple[str, ...] = (
    "metric_name",
    "value",
    "sample_size",
    "status",
)

BASELINE_COMPARISON_COLUMNS: tuple[str, ...] = (
    "metric_name",
    "model_value",
    "uniform_baseline_value",
    "empirical_frequency_baseline_value",
    "lower_is_better",
    "best_label",
    "note",
)

SCORELINE_EVALUATION_COLUMNS: tuple[str, ...] = (
    "match_id",
    "actual_scoreline",
    "actual_scoreline_probability",
    "actual_scoreline_rank",
    "top1_scoreline",
    "top1_probability",
    "top3_coverage",
    "top5_coverage",
    "status",
)

LINE_EVALUATION_COLUMNS: tuple[str, ...] = (
    "market",
    "scope",
    "line",
    "over_under",
    "rows",
    "wins",
    "losses",
    "pushes",
    "accuracy_excluding_pushes",
    "avg_model_probability",
    "observed_rate_excluding_pushes",
    "calibration_gap",
    "status",
)


@dataclass(frozen=True)
class SimulationEvaluationOutputs:
    evaluation: pd.DataFrame
    calibration: pd.DataFrame
    goal_errors: pd.DataFrame
    scorelines: pd.DataFrame
    baselines: pd.DataFrame
    line_evaluation: pd.DataFrame
    metrics: dict[str, Any]


def evaluate_simulation_predictions(
    *,
    predictions: pd.DataFrame,
    actual_results: pd.DataFrame | None = None,
    scoreline_distribution: pd.DataFrame | None = None,
    dynamic_market_lines: pd.DataFrame | None = None,
    evaluation_mode: str = "retrospective_backtest",
    min_matches_for_calibration: int = 30,
) -> SimulationEvaluationOutputs:
    """Evaluate simulator outputs against known results.

    This module is deliberately offline and read-only with respect to model logic.
    It evaluates already-generated predictions, compares them with simple
    diagnostic baselines and clearly marks unavailable metrics when actual
    results are missing.
    """
    predictions = _ensure_frame(predictions)
    actual_results = _ensure_frame(actual_results)
    scoreline_distribution = _ensure_frame(scoreline_distribution)
    dynamic_market_lines = _ensure_frame(dynamic_market_lines)

    warnings: list[str] = []
    if predictions.empty:
        warnings.append("predictions_not_available")
        return _empty_outputs(
            status="not_available",
            evaluation_mode=evaluation_mode,
            warnings=warnings,
            reason="predictions input is empty",
        )

    predictions_std = _standardize_predictions(predictions)
    actuals_std = _standardize_actual_results(actual_results)
    if actuals_std.empty:
        warnings.append("actual_results_not_available")
        return _empty_outputs(
            status="not_available",
            evaluation_mode=evaluation_mode,
            warnings=warnings,
            reason="actual results input is empty or missing required goal columns",
            predictions_rows=len(predictions_std),
        )

    merged = predictions_std.merge(
        actuals_std,
        on="match_id",
        how="inner",
        suffixes=("", "_actual"),
    )
    if merged.empty:
        warnings.append("no_matching_match_ids_between_predictions_and_actual_results")
        return _empty_outputs(
            status="not_available",
            evaluation_mode=evaluation_mode,
            warnings=warnings,
            reason="no matching match_id values between predictions and actual results",
            predictions_rows=len(predictions_std),
            actual_rows=len(actuals_std),
        )

    dropped_predictions = int(len(predictions_std) - len(merged))
    dropped_actuals = int(len(actuals_std) - len(merged))
    if dropped_predictions:
        warnings.append(f"predictions_without_actual_results={dropped_predictions}")
    if dropped_actuals:
        warnings.append(f"actual_results_without_predictions={dropped_actuals}")

    merged = _compute_match_level_evaluation(merged, scoreline_distribution)
    calibration = _build_calibration_bins(merged)
    goal_errors = _build_goal_error_metrics(merged)
    scorelines = _build_scoreline_evaluation(merged, scoreline_distribution)
    baselines = _build_baseline_comparison(merged)
    line_evaluation = _build_line_evaluation(dynamic_market_lines, actuals_std)

    metrics = _build_metrics_payload(
        merged=merged,
        calibration=calibration,
        goal_errors=goal_errors,
        scorelines=scorelines,
        baselines=baselines,
        line_evaluation=line_evaluation,
        evaluation_mode=evaluation_mode,
        warnings=warnings,
        min_matches_for_calibration=min_matches_for_calibration,
        predictions_rows=len(predictions_std),
        actual_rows=len(actuals_std),
    )
    return SimulationEvaluationOutputs(
        evaluation=merged[list(SIMULATION_EVALUATION_COLUMNS)].copy(),
        calibration=calibration,
        goal_errors=goal_errors,
        scorelines=scorelines,
        baselines=baselines,
        line_evaluation=line_evaluation,
        metrics=metrics,
    )


def write_simulation_evaluation_outputs(
    *,
    out_dir: str | Path,
    outputs: SimulationEvaluationOutputs,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "simulation_evaluation.csv": outputs.evaluation,
        "calibration_1x2.csv": outputs.calibration,
        "goal_error_metrics.csv": outputs.goal_errors,
        "scoreline_evaluation.csv": outputs.scorelines,
        "baseline_comparison.csv": outputs.baselines,
        "line_evaluation.csv": outputs.line_evaluation,
    }
    written: dict[str, str] = {}
    for name, frame in files.items():
        path = out / name
        frame.to_csv(path, index=False)
        written[name] = str(path)

    metrics_path = out / "simulation_metrics.json"
    _write_json(metrics_path, outputs.metrics)
    written["simulation_metrics.json"] = str(metrics_path)

    report_path = build_simulation_evaluation_html_report(
        out / "simulation_evaluation_report.html",
        outputs=outputs,
    )
    written["simulation_evaluation_report.html"] = str(report_path)
    return written


def build_simulation_evaluation_html_report(
    out_path: str | Path,
    *,
    outputs: SimulationEvaluationOutputs,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics = outputs.metrics
    summary = metrics.get("metrics", {})
    baseline_summary = metrics.get("baselines", {})
    warnings = metrics.get("warnings", [])

    css = """
    body { font-family: Arial, sans-serif; margin: 28px; color: #111827; }
    h1 { margin-bottom: 6px; }
    h2 { margin-top: 28px; padding-bottom: 6px; border-bottom: 2px solid #d9dee7; }
    .subtitle { color: #626c7a; margin-top: 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 18px 0; }
    .card { border: 1px solid #d9dee7; border-radius: 10px; padding: 14px; background: #f8fafc; }
    .metric { font-size: 24px; font-weight: 700; }
    .label { color: #626c7a; font-size: 12px; text-transform: uppercase; }
    .warn { background: #fff7df; padding: 10px; border: 1px solid #d7aa28; border-radius: 8px; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 10px 0 20px; }
    th, td { border: 1px solid #d9dee7; padding: 6px 8px; text-align: left; }
    th { background: #eef2f7; }
    code { background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }
    """
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Mundialytics Simulation Evaluation Report</title>
<style>{css}</style>
</head>
<body>
<h1>Mundialytics Simulation Evaluation Report</h1>
<p class="subtitle">{html.escape(SIMULATION_EVALUATION_VERSION)} · evaluation mode: <code>{html.escape(str(metrics.get("evaluation_mode", "unknown")))}</code></p>

<h2>Evaluation Summary</h2>
<div class="grid">
{_metric_card("Status", metrics.get("status", "unknown"))}
{_metric_card("Matches evaluated", metrics.get("matches_evaluated", 0))}
{_metric_card("1X2 accuracy", _fmt(summary.get("accuracy_1x2")))}
{_metric_card("1X2 log loss", _fmt(summary.get("log_loss_1x2")))}
{_metric_card("Brier score", _fmt(summary.get("brier_1x2")))}
{_metric_card("Top-3 scoreline coverage", _fmt(summary.get("scoreline_top3_coverage")))}
</div>

<h2>Warnings and Evaluation Mode</h2>
{_warnings_html(warnings, metrics)}

<h2>Model vs Baselines</h2>
{_table(outputs.baselines)}

<h2>1X2 Calibration Bins</h2>
{_table(outputs.calibration)}

<h2>Goal Error Metrics</h2>
{_table(outputs.goal_errors)}

<h2>Scoreline Evaluation</h2>
{_table(outputs.scorelines)}

<h2>Dynamic Line Evaluation</h2>
<p>This evaluates only statistical line rows that can be matched to known goals. It is not ROI, CLV or betting performance.</p>
{_table(outputs.line_evaluation)}

<h2>Per-Match Evaluation</h2>
{_table(outputs.evaluation)}

<h2>Next Data Foundation Requirements</h2>
<p>The next phase should collect forward-logged predictions and reliable actual results before model changes or betting claims.</p>
<ul>
<li>Historical and forward fixtures with stable <code>match_id</code>.</li>
<li>Actual scores, status and kickoff timestamps.</li>
<li>Pre-match prediction snapshots generated before kickoff.</li>
<li>Competition, team scope, gender, team type and objective context labels.</li>
<li>Player squads/lineups and availability for future player-prop and Golden Boot modelling.</li>
</ul>
</body>
</html>"""
    out.write_text(html_doc, encoding="utf-8")
    return out


def _empty_outputs(
    *,
    status: str,
    evaluation_mode: str,
    warnings: list[str],
    reason: str,
    predictions_rows: int = 0,
    actual_rows: int = 0,
) -> SimulationEvaluationOutputs:
    metrics = {
        "version": SIMULATION_EVALUATION_VERSION,
        "status": status,
        "evaluation_mode": evaluation_mode,
        "matches_evaluated": 0,
        "predictions_rows": int(predictions_rows),
        "actual_rows": int(actual_rows),
        "reason": reason,
        "warnings": warnings,
        "metrics": {},
        "baselines": {},
        "generated_at_utc": _now_utc(),
        "principles": _principles(),
    }
    return SimulationEvaluationOutputs(
        evaluation=pd.DataFrame(columns=SIMULATION_EVALUATION_COLUMNS),
        calibration=pd.DataFrame(columns=CALIBRATION_COLUMNS),
        goal_errors=pd.DataFrame(columns=GOAL_ERROR_COLUMNS),
        scorelines=pd.DataFrame(columns=SCORELINE_EVALUATION_COLUMNS),
        baselines=pd.DataFrame(columns=BASELINE_COMPARISON_COLUMNS),
        line_evaluation=pd.DataFrame(columns=LINE_EVALUATION_COLUMNS),
        metrics=metrics,
    )


def _standardize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    required = ["match_id", "p_home_win", "p_draw", "p_away_win"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"predictions missing required columns: {missing}")
    for c in ["match_id", "home_team", "away_team", "competition", "date"]:
        if c not in df.columns:
            df[c] = ""
    df["match_id"] = df["match_id"].astype(str)
    for c in ["p_home_win", "p_draw", "p_away_win"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    prob_sum = df[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    prob_sum = prob_sum.where(prob_sum > 0, 1.0)
    for c in ["p_home_win", "p_draw", "p_away_win"]:
        df[c] = (df[c] / prob_sum).clip(1e-12, 1.0)
    df["expected_home_goals"] = pd.to_numeric(df.get("expected_home_goals", df.get("lambda_home")), errors="coerce")
    df["expected_away_goals"] = pd.to_numeric(df.get("expected_away_goals", df.get("lambda_away")), errors="coerce")
    return df


def _standardize_actual_results(actual_results: pd.DataFrame) -> pd.DataFrame:
    if actual_results is None or actual_results.empty:
        return pd.DataFrame()
    df = actual_results.copy()
    rename = {}
    if "fixture_id" in df.columns and "match_id" not in df.columns:
        rename["fixture_id"] = "match_id"
    if "home_score" in df.columns and "home_goals" not in df.columns:
        rename["home_score"] = "home_goals"
    if "away_score" in df.columns and "away_goals" not in df.columns:
        rename["away_score"] = "away_goals"
    if "home_goals_actual" in df.columns and "home_goals" not in df.columns:
        rename["home_goals_actual"] = "home_goals"
    if "away_goals_actual" in df.columns and "away_goals" not in df.columns:
        rename["away_goals_actual"] = "away_goals"
    df = df.rename(columns=rename)
    required = ["match_id", "home_goals", "away_goals"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame()
    df["match_id"] = df["match_id"].astype(str)
    df["actual_home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["actual_away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df.dropna(subset=["actual_home_goals", "actual_away_goals"])
    df["actual_home_goals"] = df["actual_home_goals"].astype(int)
    df["actual_away_goals"] = df["actual_away_goals"].astype(int)
    for c in ["date", "competition", "home_team", "away_team", "status", "source"]:
        if c not in df.columns:
            df[c] = ""
    return df[["match_id", "date", "competition", "home_team", "away_team", "actual_home_goals", "actual_away_goals", "status", "source"]]


def _compute_match_level_evaluation(merged: pd.DataFrame, scorelines: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    df["actual_outcome"] = [
        _actual_outcome(h, a) for h, a in zip(df["actual_home_goals"], df["actual_away_goals"])
    ]
    pred_cols = ["p_home_win", "p_draw", "p_away_win"]
    pred_labels = ["home_win", "draw", "away_win"]
    pred_idx = df[pred_cols].astype(float).values.argmax(axis=1)
    df["predicted_outcome"] = [pred_labels[i] for i in pred_idx]
    df["predicted_outcome_probability"] = [float(df.iloc[i][pred_cols[pred_idx[i]]]) for i in range(len(df))]
    df["prediction_correct"] = df["predicted_outcome"].eq(df["actual_outcome"])
    df["actual_total_goals"] = df["actual_home_goals"] + df["actual_away_goals"]
    df["expected_total_goals"] = df["expected_home_goals"] + df["expected_away_goals"]
    df["home_goal_error"] = df["expected_home_goals"] - df["actual_home_goals"]
    df["away_goal_error"] = df["expected_away_goals"] - df["actual_away_goals"]
    df["total_goal_error"] = df["expected_total_goals"] - df["actual_total_goals"]
    df["actual_scoreline"] = df["actual_home_goals"].astype(str) + "-" + df["actual_away_goals"].astype(str)

    scoreline_lookup = _scoreline_lookup(scorelines)
    actual_probs: list[float] = []
    actual_ranks: list[Any] = []
    top1_flags: list[bool] = []
    top3_flags: list[bool] = []
    top5_flags: list[bool] = []
    for _, row in df.iterrows():
        summary = scoreline_lookup.get(str(row["match_id"]), {})
        key = (int(row["actual_home_goals"]), int(row["actual_away_goals"]))
        actual_probs.append(float(summary.get("probabilities", {}).get(key, 0.0)))
        rank = summary.get("ranks", {}).get(key)
        actual_ranks.append(rank if rank is not None else pd.NA)
        top1_flags.append(bool(rank is not None and rank <= 1))
        top3_flags.append(bool(rank is not None and rank <= 3))
        top5_flags.append(bool(rank is not None and rank <= 5))
    df["actual_scoreline_probability"] = actual_probs
    df["actual_scoreline_rank"] = actual_ranks
    df["actual_scoreline_top1"] = top1_flags
    df["actual_scoreline_top3"] = top3_flags
    df["actual_scoreline_top5"] = top5_flags
    df["data_quality_flag"] = "actual_result_available"
    df["evaluation_note"] = "offline_evaluation_no_model_change"
    # Prefer prediction metadata for display, falling back to actual fields.
    for c in ["date", "competition", "home_team", "away_team"]:
        actual_col = f"{c}_actual"
        if actual_col in df.columns:
            df[c] = df[c].where(df[c].astype(str).str.len() > 0, df[actual_col])
    return df


def _scoreline_lookup(scorelines: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if scorelines is None or scorelines.empty:
        return {}
    required = {"match_id", "home_goals", "away_goals", "probability"}
    if not required.issubset(scorelines.columns):
        return {}
    work = scorelines.copy()
    work["match_id"] = work["match_id"].astype(str)
    work["home_goals"] = pd.to_numeric(work["home_goals"], errors="coerce").astype("Int64")
    work["away_goals"] = pd.to_numeric(work["away_goals"], errors="coerce").astype("Int64")
    work["probability"] = pd.to_numeric(work["probability"], errors="coerce").fillna(0.0)
    lookup: dict[str, dict[str, Any]] = {}
    for match_id, frame in work.dropna(subset=["home_goals", "away_goals"]).groupby("match_id", dropna=False):
        ranked = frame.sort_values("probability", ascending=False).reset_index(drop=True)
        probs: dict[tuple[int, int], float] = {}
        ranks: dict[tuple[int, int], int] = {}
        for i, row in ranked.iterrows():
            key = (int(row["home_goals"]), int(row["away_goals"]))
            probs[key] = float(row["probability"])
            ranks[key] = int(i + 1)
        lookup[str(match_id)] = {"probabilities": probs, "ranks": ranks, "ranked": ranked}
    return lookup


def _build_calibration_bins(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=CALIBRATION_COLUMNS)
    rows: list[dict[str, Any]] = []
    class_map = {
        "home_win": ("p_home_win", "home_win"),
        "draw": ("p_draw", "draw"),
        "away_win": ("p_away_win", "away_win"),
    }
    bins = [(i / 10, (i + 1) / 10) for i in range(10)]
    for outcome_class, (prob_col, label) in class_map.items():
        probs = pd.to_numeric(df[prob_col], errors="coerce")
        observed = df["actual_outcome"].eq(label).astype(float)
        for lower, upper in bins:
            if upper >= 1.0:
                mask = (probs >= lower) & (probs <= upper)
            else:
                mask = (probs >= lower) & (probs < upper)
            n = int(mask.sum())
            if n:
                avg_p = float(probs[mask].mean())
                obs = float(observed[mask].mean())
                err = avg_p - obs
                status = "available"
            else:
                avg_p = math.nan
                obs = math.nan
                err = math.nan
                status = "not_available"
            rows.append(
                {
                    "outcome_class": outcome_class,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "row_count": n,
                    "avg_predicted_probability": avg_p,
                    "observed_frequency": obs,
                    "calibration_error": err,
                    "status": status,
                }
            )
    return pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)


def _build_goal_error_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=GOAL_ERROR_COLUMNS)
    metrics = {
        "home_goals_mae": _mae(df["home_goal_error"]),
        "away_goals_mae": _mae(df["away_goal_error"]),
        "total_goals_mae": _mae(df["total_goal_error"]),
        "home_goals_rmse": _rmse(df["home_goal_error"]),
        "away_goals_rmse": _rmse(df["away_goal_error"]),
        "total_goals_rmse": _rmse(df["total_goal_error"]),
        "expected_total_goals_mean": float(pd.to_numeric(df["expected_total_goals"], errors="coerce").mean()),
        "actual_total_goals_mean": float(pd.to_numeric(df["actual_total_goals"], errors="coerce").mean()),
    }
    return pd.DataFrame(
        [
            {
                "metric_name": name,
                "value": value,
                "sample_size": int(len(df)),
                "status": "available" if not pd.isna(value) else "not_available",
            }
            for name, value in metrics.items()
        ],
        columns=GOAL_ERROR_COLUMNS,
    )


def _build_scoreline_evaluation(df: pd.DataFrame, scorelines: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SCORELINE_EVALUATION_COLUMNS)
    lookup = _scoreline_lookup(scorelines)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        match_id = str(row["match_id"])
        summary = lookup.get(match_id, {})
        ranked = summary.get("ranked", pd.DataFrame())
        if isinstance(ranked, pd.DataFrame) and not ranked.empty:
            top = ranked.iloc[0]
            top1_scoreline = f"{int(top['home_goals'])}-{int(top['away_goals'])}"
            top1_probability = float(top["probability"])
            status = "available"
        else:
            top1_scoreline = ""
            top1_probability = math.nan
            status = "scoreline_distribution_not_available"
        rows.append(
            {
                "match_id": match_id,
                "actual_scoreline": row["actual_scoreline"],
                "actual_scoreline_probability": row["actual_scoreline_probability"],
                "actual_scoreline_rank": row["actual_scoreline_rank"],
                "top1_scoreline": top1_scoreline,
                "top1_probability": top1_probability,
                "top3_coverage": bool(row["actual_scoreline_top3"]),
                "top5_coverage": bool(row["actual_scoreline_top5"]),
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=SCORELINE_EVALUATION_COLUMNS)


def _build_baseline_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=BASELINE_COMPARISON_COLUMNS)
    y = _outcome_matrix(df["actual_outcome"])
    model_p = df[["p_home_win", "p_draw", "p_away_win"]].astype(float).values
    uniform_p = [[1 / 3, 1 / 3, 1 / 3] for _ in range(len(df))]
    freqs = y.mean(axis=0)
    empirical_p = [freqs.tolist() for _ in range(len(df))]

    model_pred = model_p.argmax(axis=1)
    empirical_pred = int(freqs.argmax())
    rows = [
        _baseline_row(
            "accuracy_1x2",
            float((model_pred == y.argmax(axis=1)).mean()),
            1 / 3,
            float((empirical_pred == y.argmax(axis=1)).mean()),
            lower_is_better=False,
            note="Empirical frequency baseline is diagnostic/in-sample unless supplied from a train period.",
        ),
        _baseline_row(
            "log_loss_1x2",
            _log_loss(y, model_p),
            _log_loss(y, uniform_p),
            _log_loss(y, empirical_p),
            lower_is_better=True,
            note="Lower is better. Probabilities are clipped for numerical stability.",
        ),
        _baseline_row(
            "brier_1x2",
            _brier(y, model_p),
            _brier(y, uniform_p),
            _brier(y, empirical_p),
            lower_is_better=True,
            note="Multiclass Brier uses mean squared probability error across H/D/A.",
        ),
    ]
    return pd.DataFrame(rows, columns=BASELINE_COMPARISON_COLUMNS)


def _build_line_evaluation(dynamic_lines: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    if dynamic_lines is None or dynamic_lines.empty or actuals.empty:
        return pd.DataFrame(columns=LINE_EVALUATION_COLUMNS)
    required = {"match_id", "market", "scope", "line", "over_under", "model_probability"}
    if not required.issubset(dynamic_lines.columns):
        return pd.DataFrame(columns=LINE_EVALUATION_COLUMNS)
    actual = actuals.copy()
    actual["actual_total_goals"] = actual["actual_home_goals"] + actual["actual_away_goals"]
    work = dynamic_lines.copy()
    work = work[work["market"].astype(str).str.lower().eq("goals")]
    work = work[work["scope"].astype(str).str.lower().eq("match")]
    if "side" in work.columns:
        work = work[work["side"].astype(str).str.lower().isin(["both", "match", ""])]
    work = work.merge(actual[["match_id", "actual_total_goals"]], on="match_id", how="inner")
    if work.empty:
        return pd.DataFrame(columns=LINE_EVALUATION_COLUMNS)
    work["line"] = pd.to_numeric(work["line"], errors="coerce")
    work["model_probability"] = pd.to_numeric(work["model_probability"], errors="coerce")
    work = work.dropna(subset=["line", "model_probability"])
    if work.empty:
        return pd.DataFrame(columns=LINE_EVALUATION_COLUMNS)

    def settle(row: pd.Series) -> str:
        actual_total = float(row["actual_total_goals"])
        line = float(row["line"])
        side = str(row["over_under"]).lower()
        if actual_total == line:
            return "push"
        if side == "over":
            return "win" if actual_total > line else "loss"
        if side == "under":
            return "win" if actual_total < line else "loss"
        return "not_available"

    work["result"] = work.apply(settle, axis=1)
    rows: list[dict[str, Any]] = []
    for keys, frame in work.groupby(["market", "scope", "line", "over_under"], dropna=False):
        market, scope, line, over_under = keys
        wins = int(frame["result"].eq("win").sum())
        losses = int(frame["result"].eq("loss").sum())
        pushes = int(frame["result"].eq("push").sum())
        settled = wins + losses
        observed = wins / settled if settled else math.nan
        avg_p = float(frame["model_probability"].mean()) if len(frame) else math.nan
        rows.append(
            {
                "market": market,
                "scope": scope,
                "line": line,
                "over_under": over_under,
                "rows": int(len(frame)),
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "accuracy_excluding_pushes": observed,
                "avg_model_probability": avg_p,
                "observed_rate_excluding_pushes": observed,
                "calibration_gap": avg_p - observed if not pd.isna(avg_p) and not pd.isna(observed) else math.nan,
                "status": "available" if settled else "not_available_no_settled_rows",
            }
        )
    return pd.DataFrame(rows, columns=LINE_EVALUATION_COLUMNS)


def _build_metrics_payload(
    *,
    merged: pd.DataFrame,
    calibration: pd.DataFrame,
    goal_errors: pd.DataFrame,
    scorelines: pd.DataFrame,
    baselines: pd.DataFrame,
    line_evaluation: pd.DataFrame,
    evaluation_mode: str,
    warnings: list[str],
    min_matches_for_calibration: int,
    predictions_rows: int,
    actual_rows: int,
) -> dict[str, Any]:
    y = _outcome_matrix(merged["actual_outcome"])
    model_p = merged[["p_home_win", "p_draw", "p_away_win"]].astype(float).values
    date_range = _date_range(merged)
    if len(merged) < min_matches_for_calibration:
        warnings = list(warnings) + [f"low_sample_for_calibration_bins={len(merged)}<{min_matches_for_calibration}"]
    metrics = {
        "accuracy_1x2": float(merged["prediction_correct"].mean()) if len(merged) else None,
        "log_loss_1x2": _log_loss(y, model_p),
        "brier_1x2": _brier(y, model_p),
        "home_goals_mae": _metric_value(goal_errors, "home_goals_mae"),
        "away_goals_mae": _metric_value(goal_errors, "away_goals_mae"),
        "total_goals_mae": _metric_value(goal_errors, "total_goals_mae"),
        "total_goals_rmse": _metric_value(goal_errors, "total_goals_rmse"),
        "scoreline_top1_accuracy": float(merged["actual_scoreline_top1"].mean()) if len(merged) else None,
        "scoreline_top3_coverage": float(merged["actual_scoreline_top3"].mean()) if len(merged) else None,
        "scoreline_top5_coverage": float(merged["actual_scoreline_top5"].mean()) if len(merged) else None,
        "actual_scoreline_probability_mean": float(pd.to_numeric(merged["actual_scoreline_probability"], errors="coerce").mean()) if len(merged) else None,
    }
    baseline_map = {
        str(row["metric_name"]): {
            "model_value": _safe_float(row["model_value"]),
            "uniform_baseline_value": _safe_float(row["uniform_baseline_value"]),
            "empirical_frequency_baseline_value": _safe_float(row["empirical_frequency_baseline_value"]),
            "best_label": row["best_label"],
        }
        for _, row in baselines.iterrows()
    }
    return {
        "version": SIMULATION_EVALUATION_VERSION,
        "status": "completed",
        "evaluation_mode": evaluation_mode,
        "matches_evaluated": int(len(merged)),
        "predictions_rows": int(predictions_rows),
        "actual_rows": int(actual_rows),
        "date_range": date_range,
        "metrics": metrics,
        "baselines": baseline_map,
        "line_evaluation_rows": int(len(line_evaluation)),
        "calibration_bins_available": int(calibration["status"].eq("available").sum()) if not calibration.empty else 0,
        "warnings": warnings,
        "generated_at_utc": _now_utc(),
        "principles": _principles(),
        "next_data_phase": _next_data_phase_requirements(),
    }


def _principles() -> dict[str, Any]:
    return {
        "offline_only": True,
        "model_logic_changed": False,
        "odds_required": False,
        "betting_recommendations": False,
        "player_props_deep_evaluation": False,
        "no_live_automation": True,
        "missing_data_policy": "not_available",
    }


def _next_data_phase_requirements() -> dict[str, Any]:
    return {
        "minimum_match_results_columns": [
            "match_id",
            "date",
            "competition",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "status",
        ],
        "forward_prediction_snapshot_columns": [
            "run_id",
            "generated_at_utc",
            "match_id",
            "kickoff_utc",
            "model_version",
            "data_version",
            "p_home_win",
            "p_draw",
            "p_away_win",
            "lambda_home",
            "lambda_away",
        ],
        "future_player_prop_columns": [
            "match_id",
            "team",
            "player",
            "player_id",
            "position",
            "current_squad_flag",
            "expected_minutes",
            "actual_minutes",
            "goals",
            "shots",
            "shots_on_target",
            "fouls_committed",
            "yellow_cards",
        ],
        "golden_boot_data_needed": [
            "player_id",
            "player",
            "team",
            "current_squad",
            "expected_minutes_by_match",
            "scoring_rate_features",
            "team_progression_probabilities",
            "actual_goals_by_match",
        ],
    }


def _baseline_row(metric_name: str, model_value: float, uniform_value: float, empirical_value: float, *, lower_is_better: bool, note: str) -> dict[str, Any]:
    values = {
        "model": model_value,
        "uniform_baseline": uniform_value,
        "empirical_frequency_baseline": empirical_value,
    }
    if lower_is_better:
        best = min(values, key=values.get)
    else:
        best = max(values, key=values.get)
    return {
        "metric_name": metric_name,
        "model_value": model_value,
        "uniform_baseline_value": uniform_value,
        "empirical_frequency_baseline_value": empirical_value,
        "lower_is_better": lower_is_better,
        "best_label": best,
        "note": note,
    }


def _actual_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def _outcome_matrix(outcomes: pd.Series) -> Any:
    rows = []
    for value in outcomes.astype(str):
        rows.append([
            1.0 if value == "home_win" else 0.0,
            1.0 if value == "draw" else 0.0,
            1.0 if value == "away_win" else 0.0,
        ])
    return pd.DataFrame(rows).values


def _log_loss(y: Any, p: Any) -> float:
    y_arr = pd.DataFrame(y).astype(float).values
    p_arr = pd.DataFrame(p).astype(float).clip(1e-12, 1.0).values
    row_sum = p_arr.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    p_arr = p_arr / row_sum
    return float(-(y_arr * pd.DataFrame(p_arr).map(math.log).values).sum(axis=1).mean())


def _brier(y: Any, p: Any) -> float:
    y_arr = pd.DataFrame(y).astype(float).values
    p_arr = pd.DataFrame(p).astype(float).values
    return float(((p_arr - y_arr) ** 2).sum(axis=1).mean())


def _mae(errors: pd.Series) -> float:
    return float(pd.to_numeric(errors, errors="coerce").abs().mean())


def _rmse(errors: pd.Series) -> float:
    values = pd.to_numeric(errors, errors="coerce")
    return float(math.sqrt((values ** 2).mean()))


def _metric_value(frame: pd.DataFrame, metric_name: str) -> float | None:
    if frame.empty:
        return None
    row = frame[frame["metric_name"].astype(str).eq(metric_name)]
    if row.empty:
        return None
    return _safe_float(row.iloc[0]["value"])


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _date_range(df: pd.DataFrame) -> dict[str, str | None]:
    if "date" not in df.columns:
        return {"min": None, "max": None}
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.dropna().empty:
        return {"min": None, "max": None}
    return {
        "min": str(dates.min().date()),
        "max": str(dates.max().date()),
    }


def _ensure_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.NA.__class__,)):
        return None
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return str(value)


def _metric_card(label: str, value: Any) -> str:
    return f'<div class="card"><div class="label">{html.escape(str(label))}</div><div class="metric">{html.escape(str(value))}</div></div>'


def _warnings_html(warnings: list[str], metrics: dict[str, Any]) -> str:
    if not warnings:
        body = "<p>No warnings.</p>"
    else:
        body = "<ul>" + "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings) + "</ul>"
    principles = metrics.get("principles", {})
    return f"""
<div class="warn">
<strong>Policy:</strong> offline evaluation only. No model changes, no betting recommendations and no API calls.
{body}
<p><strong>Missing data policy:</strong> {html.escape(str(principles.get("missing_data_policy", "not_available")))}</p>
</div>
"""


def _table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "<p><em>not_available</em></p>"
    safe = df.head(max_rows).copy()
    for col in safe.columns:
        safe[col] = safe[col].map(_display_value)
    return safe.to_html(index=False, escape=True)


def _display_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return round(value, 6)
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "not_available"
    try:
        if pd.isna(value):
            return "not_available"
        return f"{float(value):.4f}"
    except Exception:
        return str(value)
