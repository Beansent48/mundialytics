from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression

from mundialytics.evaluation.metrics import brier_multiclass, rank_probability_score, safe_log_loss
from mundialytics.utils import clip_lambda


DEFAULT_TOTAL_GOAL_LINES: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5, 4.5)
DEFAULT_CALIBRATION_BINS: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 11))
DEFAULT_DIXON_COLES_RHO_GRID: tuple[float, ...] = tuple(np.round(np.arange(-0.20, 0.201, 0.01), 2))


def _binary_log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-15
    prob = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    obs = np.asarray(y_true, dtype=float)
    return float(-np.mean(obs * np.log(prob) + (1 - obs) * np.log(1 - prob)))


def _brier_binary(y_true: np.ndarray, p: np.ndarray) -> float:
    obs = np.asarray(y_true, dtype=float)
    prob = np.asarray(p, dtype=float)
    return float(np.mean((prob - obs) ** 2))


def _safe_mean(values: pd.Series | np.ndarray | list[float]) -> float | None:
    arr = pd.Series(values).dropna()
    if arr.empty:
        return None
    return float(arr.mean())


def _actual_outcome_label(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _onehot_1x2(outcome: str) -> list[int]:
    return {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}[outcome]


def _line_probability_rows(
    predictions: pd.DataFrame,
    *,
    total_goal_lines: tuple[float, ...] = DEFAULT_TOTAL_GOAL_LINES,
) -> pd.DataFrame:
    """Expand prediction rows into binary statistical market rows.

    These rows are for calibration/statistical evaluation only. They do not
    include odds, staking or ROI.
    """
    df = predictions.copy()
    required = {"match_id", "home_goals", "away_goals", "lambda_home", "lambda_away"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions missing line-calibration columns: {sorted(missing)}")
    df = df.dropna(subset=list(required)).copy()
    if df.empty:
        return pd.DataFrame()

    actual_total = df["home_goals"].astype(float) + df["away_goals"].astype(float)
    expected_total = df["lambda_home"].astype(float) + df["lambda_away"].astype(float)
    rows: list[dict[str, Any]] = []

    for line in total_goal_lines:
        threshold = math.floor(float(line))
        p_over = 1 - poisson.cdf(threshold, expected_total)
        actual_over = (actual_total > float(line)).astype(int).to_numpy()
        for idx, (_, row) in enumerate(df.iterrows()):
            base = {
                "match_id": row.get("match_id"),
                "date": row.get("date"),
                "competition": row.get("competition", "unknown"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "market": "total_goals",
                "line": float(line),
            }
            p = float(np.asarray(p_over, dtype=float)[idx])
            obs = int(actual_over[idx])
            rows.append({**base, "side": "over", "model_probability": p, "observed": obs})
            rows.append({**base, "side": "under", "model_probability": 1.0 - p, "observed": 1 - obs})

    p_btts = (
        1
        - poisson.pmf(0, df["lambda_home"].astype(float))
        - poisson.pmf(0, df["lambda_away"].astype(float))
        + poisson.pmf(0, df["lambda_home"].astype(float)) * poisson.pmf(0, df["lambda_away"].astype(float))
    )
    actual_btts = ((df["home_goals"].astype(float) > 0) & (df["away_goals"].astype(float) > 0)).astype(int).to_numpy()
    for idx, (_, row) in enumerate(df.iterrows()):
        base = {
            "match_id": row.get("match_id"),
            "date": row.get("date"),
            "competition": row.get("competition", "unknown"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "market": "btts",
            "line": np.nan,
        }
        p = float(np.asarray(p_btts, dtype=float)[idx])
        obs = int(actual_btts[idx])
        rows.append({**base, "side": "yes", "model_probability": p, "observed": obs})
        rows.append({**base, "side": "no", "model_probability": 1.0 - p, "observed": 1 - obs})

    return pd.DataFrame(rows)


def _calibration_bins(df: pd.DataFrame, prob_col: str, obs_col: str, *, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work[prob_col] = pd.to_numeric(work[prob_col], errors="coerce")
    work[obs_col] = pd.to_numeric(work[obs_col], errors="coerce")
    work = work.dropna(subset=[prob_col, obs_col])
    if work.empty:
        return pd.DataFrame()
    work["probability_bin"] = pd.cut(
        work[prob_col].clip(0, 1),
        bins=list(DEFAULT_CALIBRATION_BINS),
        include_lowest=True,
    ).astype(str)
    rows = []
    for key, g in work.groupby(group_cols + ["probability_bin"], dropna=False, observed=False):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(group_cols + ["probability_bin"], key, strict=False))
        rows.append(
            {
                **record,
                "rows": int(len(g)),
                "avg_model_probability": float(g[prob_col].mean()),
                "observed_rate": float(g[obs_col].mean()),
                "calibration_gap": float(g[prob_col].mean() - g[obs_col].mean()),
                "brier": _brier_binary(g[obs_col].to_numpy(), g[prob_col].to_numpy()),
                "log_loss": _binary_log_loss(g[obs_col].to_numpy(), g[prob_col].to_numpy()),
                "status": "available",
            }
        )
    return pd.DataFrame(rows)


def _score_probability(lambda_home: float, lambda_away: float, home_goals: int, away_goals: int) -> float:
    lh, la = clip_lambda([lambda_home, lambda_away])
    return float(poisson.pmf(int(home_goals), lh) * poisson.pmf(int(away_goals), la))


def _dixon_coles_tau(h: int, a: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """Dixon-Coles low-score dependence adjustment.

    Positive/negative rho changes the mass allocated to 0-0, 1-0, 0-1 and 1-1.
    We clip to a small positive floor because invalid rho/lambda combinations can
    make one tau negative.
    """
    if h == 0 and a == 0:
        tau = 1.0 - lambda_home * lambda_away * rho
    elif h == 0 and a == 1:
        tau = 1.0 + lambda_home * rho
    elif h == 1 and a == 0:
        tau = 1.0 + lambda_away * rho
    elif h == 1 and a == 1:
        tau = 1.0 - rho
    else:
        tau = 1.0
    return float(max(tau, 1e-9))


def _scoreline_distribution(
    lambda_home: float,
    lambda_away: float,
    *,
    max_home_goals: int = 8,
    max_away_goals: int = 8,
    dixon_coles_rho: float | None = None,
) -> pd.DataFrame:
    lh, la = clip_lambda([lambda_home, lambda_away])
    rows: list[dict[str, Any]] = []
    for h in range(max_home_goals + 1):
        for a in range(max_away_goals + 1):
            prob = float(poisson.pmf(h, lh) * poisson.pmf(a, la))
            if dixon_coles_rho is not None:
                prob *= _dixon_coles_tau(h, a, float(lh), float(la), float(dixon_coles_rho))
            rows.append({"scoreline": f"{h}-{a}", "home_goals": h, "away_goals": a, "probability": prob})
    dist = pd.DataFrame(rows)
    total = float(dist["probability"].sum())
    if total > 0:
        dist["probability"] = dist["probability"] / total
    return dist.sort_values("probability", ascending=False).reset_index(drop=True)


def _scoreline_rank(
    lambda_home: float,
    lambda_away: float,
    home_goals: int,
    away_goals: int,
    *,
    max_goals: int = 8,
    dixon_coles_rho: float | None = None,
) -> tuple[float, int | None, str, float, bool, bool, bool]:
    """Return actual-score probability and top-k coverage diagnostics.

    The ranking grid expands to include the actual scoreline. This avoids marking
    high-score real outcomes as unavailable while still keeping a finite,
    auditable scoreline diagnostic.
    """
    max_h = max(max_goals, int(home_goals))
    max_a = max(max_goals, int(away_goals))
    dist = _scoreline_distribution(
        lambda_home,
        lambda_away,
        max_home_goals=max_h,
        max_away_goals=max_a,
        dixon_coles_rho=dixon_coles_rho,
    )
    actual_scoreline = f"{int(home_goals)}-{int(away_goals)}"
    actual_rows = dist.index[dist["scoreline"].eq(actual_scoreline)].tolist()
    rank = int(actual_rows[0] + 1) if actual_rows else None
    actual_probability = float(dist.loc[actual_rows[0], "probability"]) if actual_rows else _score_probability(lambda_home, lambda_away, home_goals, away_goals)
    top1 = str(dist.iloc[0]["scoreline"]) if len(dist) else "not_available"
    top1_prob = float(dist.iloc[0]["probability"]) if len(dist) else 0.0
    return (
        actual_probability,
        rank,
        top1,
        top1_prob,
        bool(rank == 1) if rank is not None else False,
        bool(rank is not None and rank <= 3),
        bool(rank is not None and rank <= 5),
    )


def evaluate_goal_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate expected goals against actual goals for backtest predictions."""
    df = predictions.copy()
    required = {"home_goals", "away_goals", "lambda_home", "lambda_away"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions missing goal-error columns: {sorted(missing)}")
    df = df.dropna(subset=list(required)).copy()
    if df.empty:
        return pd.DataFrame(
            [
                {"metric_name": "rows", "value": 0.0, "sample_size": 0, "status": "not_available"},
            ]
        )

    actual_home = df["home_goals"].astype(float)
    actual_away = df["away_goals"].astype(float)
    pred_home = df["lambda_home"].astype(float)
    pred_away = df["lambda_away"].astype(float)
    actual_total = actual_home + actual_away
    pred_total = pred_home + pred_away

    metrics = [
        ("home_goals_mae", np.mean(np.abs(pred_home - actual_home))),
        ("away_goals_mae", np.mean(np.abs(pred_away - actual_away))),
        ("total_goals_mae", np.mean(np.abs(pred_total - actual_total))),
        ("home_goals_rmse", np.sqrt(np.mean((pred_home - actual_home) ** 2))),
        ("away_goals_rmse", np.sqrt(np.mean((pred_away - actual_away) ** 2))),
        ("total_goals_rmse", np.sqrt(np.mean((pred_total - actual_total) ** 2))),
        ("expected_home_goals_mean", pred_home.mean()),
        ("actual_home_goals_mean", actual_home.mean()),
        ("expected_away_goals_mean", pred_away.mean()),
        ("actual_away_goals_mean", actual_away.mean()),
        ("expected_total_goals_mean", pred_total.mean()),
        ("actual_total_goals_mean", actual_total.mean()),
        ("expected_total_minus_actual_total", (pred_total - actual_total).mean()),
    ]
    return pd.DataFrame(
        [
            {
                "metric_name": name,
                "value": float(value),
                "sample_size": int(len(df)),
                "status": "available",
            }
            for name, value in metrics
        ]
    )


def evaluate_goal_lines(
    predictions: pd.DataFrame,
    *,
    total_goal_lines: tuple[float, ...] = DEFAULT_TOTAL_GOAL_LINES,
) -> pd.DataFrame:
    """Evaluate statistical goal-line probabilities, not betting profitability."""
    line_rows = _line_probability_rows(predictions, total_goal_lines=total_goal_lines)
    if line_rows.empty:
        return pd.DataFrame(
            columns=[
                "market",
                "line",
                "side",
                "rows",
                "avg_model_probability",
                "observed_rate",
                "calibration_gap",
                "brier",
                "log_loss",
                "status",
            ]
        )
    rows = []
    group_cols = ["market", "line", "side"]
    for key, g in line_rows.groupby(group_cols, dropna=False):
        market, line, side = key
        rows.append(
            {
                "market": market,
                "line": None if pd.isna(line) else float(line),
                "side": side,
                "rows": int(len(g)),
                "avg_model_probability": float(g["model_probability"].mean()),
                "observed_rate": float(g["observed"].mean()),
                "calibration_gap": float(g["model_probability"].mean() - g["observed"].mean()),
                "brier": _brier_binary(g["observed"].to_numpy(), g["model_probability"].to_numpy()),
                "log_loss": _binary_log_loss(g["observed"].to_numpy(), g["model_probability"].to_numpy()),
                "accuracy": float(((g["model_probability"] >= 0.5).astype(int) == g["observed"].astype(int)).mean()),
                "status": "available",
            }
        )
    return pd.DataFrame(rows)


def evaluate_goal_line_calibration_bins(
    predictions: pd.DataFrame,
    *,
    total_goal_lines: tuple[float, ...] = DEFAULT_TOTAL_GOAL_LINES,
) -> pd.DataFrame:
    """Detailed calibration diagnostics for totals and BTTS probability bins."""
    line_rows = _line_probability_rows(predictions, total_goal_lines=total_goal_lines)
    if line_rows.empty:
        return pd.DataFrame()
    return _calibration_bins(
        line_rows,
        "model_probability",
        "observed",
        group_cols=["market", "line", "side"],
    )


def evaluate_scorelines(
    predictions: pd.DataFrame,
    *,
    max_goals: int = 8,
    dixon_coles_rho: float | None = None,
) -> pd.DataFrame:
    """Evaluate exact-score distribution implied by the goal lambdas."""
    df = predictions.copy()
    required = {"match_id", "home_goals", "away_goals", "lambda_home", "lambda_away"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions missing scoreline columns: {sorted(missing)}")
    df = df.dropna(subset=list(required)).copy()

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        actual_home = int(row["home_goals"])
        actual_away = int(row["away_goals"])
        (
            actual_prob,
            rank,
            top1,
            top1_prob,
            top1_coverage,
            top3_coverage,
            top5_coverage,
        ) = _scoreline_rank(
            float(row["lambda_home"]),
            float(row["lambda_away"]),
            actual_home,
            actual_away,
            max_goals=max_goals,
            dixon_coles_rho=dixon_coles_rho,
        )
        rows.append(
            {
                "match_id": row["match_id"],
                "actual_scoreline": f"{actual_home}-{actual_away}",
                "actual_scoreline_probability": actual_prob,
                "actual_scoreline_rank": rank,
                "top1_scoreline": top1,
                "top1_probability": top1_prob,
                "top1_coverage": top1_coverage,
                "top3_coverage": top3_coverage,
                "top5_coverage": top5_coverage,
                "scoreline_log_loss": float(-math.log(max(actual_prob, 1e-15))),
                "dixon_coles_rho": dixon_coles_rho,
                "status": "available",
            }
        )
    return pd.DataFrame(rows)


def _summarise_scoreline_eval(scoreline_eval: pd.DataFrame) -> dict[str, Any]:
    available = scoreline_eval[scoreline_eval["status"].eq("available")]
    return {
        "scoreline_log_loss": _safe_mean(available["scoreline_log_loss"]) if not available.empty else None,
        "actual_scoreline_probability_mean": _safe_mean(available["actual_scoreline_probability"]) if not available.empty else None,
        "top1_accuracy": float(available["top1_coverage"].mean()) if not available.empty else None,
        "top3_coverage": float(available["top3_coverage"].mean()) if not available.empty else None,
        "top5_coverage": float(available["top5_coverage"].mean()) if not available.empty else None,
    }


def _split_calibration_eval(df: pd.DataFrame, *, calibration_fraction: float = 0.50) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()
    work = df.copy()
    if "date" in work.columns:
        work["_sort_date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.sort_values(["_sort_date", "match_id"], na_position="last").drop(columns=["_sort_date"])
    elif "match_id" in work.columns:
        work = work.sort_values("match_id")
    cut = int(max(1, min(len(work) - 1, round(len(work) * calibration_fraction)))) if len(work) > 1 else len(work)
    return work.iloc[:cut].copy(), work.iloc[cut:].copy()


def _fit_isotonic(cal_p: pd.Series, cal_y: pd.Series) -> IsotonicRegression | None:
    p = pd.to_numeric(cal_p, errors="coerce")
    y = pd.to_numeric(cal_y, errors="coerce")
    mask = p.notna() & y.notna()
    p = p[mask].astype(float).clip(0, 1)
    y = y[mask].astype(float)
    # Isotonic calibration needs at least two probability points and both outcome classes.
    if len(p) < 20 or p.nunique() < 2 or y.nunique() < 2:
        return None
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(p, y)
    return model


def _calibrate_1x2(cal: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"p_home_win", "p_draw", "p_away_win", "home_goals", "away_goals"}
    if required - set(cal.columns) or required - set(eval_df.columns) or eval_df.empty:
        return eval_df.copy(), {"status": "not_available", "reason": "missing_columns_or_empty_eval"}

    class_defs = [
        ("H", "p_home_win", "p_home_win_calibrated"),
        ("D", "p_draw", "p_draw_calibrated"),
        ("A", "p_away_win", "p_away_win_calibrated"),
    ]
    out = eval_df.copy()
    applied = []
    for label, prob_col, calibrated_col in class_defs:
        cal_y = [
            int(_actual_outcome_label(h, a) == label)
            for h, a in zip(cal["home_goals"].astype(float), cal["away_goals"].astype(float), strict=False)
        ]
        model = _fit_isotonic(cal[prob_col], pd.Series(cal_y, index=cal.index))
        if model is None:
            out[calibrated_col] = out[prob_col].astype(float)
            applied.append({"class": label, "method": "identity_insufficient_calibration_data"})
        else:
            out[calibrated_col] = model.predict(out[prob_col].astype(float).clip(0, 1))
            applied.append({"class": label, "method": "isotonic"})

    prob_cols = ["p_home_win_calibrated", "p_draw_calibrated", "p_away_win_calibrated"]
    row_sum = out[prob_cols].sum(axis=1).replace(0, np.nan)
    out[prob_cols] = out[prob_cols].div(row_sum, axis=0).fillna(1 / 3)

    base_probs = out[["p_home_win", "p_draw", "p_away_win"]].astype(float).to_numpy()
    cal_probs = out[prob_cols].astype(float).to_numpy()
    labels = [_actual_outcome_label(h, a) for h, a in zip(out["home_goals"], out["away_goals"], strict=False)]
    obs = np.asarray([_onehot_1x2(x) for x in labels])
    summary = {
        "status": "available",
        "method": "one_vs_rest_isotonic_with_row_renormalization",
        "calibration_rows": int(len(cal)),
        "evaluation_rows": int(len(out)),
        "applied": applied,
        "pre_calibration": {
            "log_loss": safe_log_loss(pd.Series(labels), base_probs, labels=["H", "D", "A"]),
            "brier_multiclass": brier_multiclass(obs, base_probs),
            "rps": rank_probability_score(base_probs, obs),
        },
        "post_calibration": {
            "log_loss": safe_log_loss(pd.Series(labels), cal_probs, labels=["H", "D", "A"]),
            "brier_multiclass": brier_multiclass(obs, cal_probs),
            "rps": rank_probability_score(cal_probs, obs),
        },
    }
    return out, summary


def _calibrate_binary_markets(cal: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if cal.empty or eval_df.empty:
        return eval_df.copy(), []
    rows = []
    out_parts = []
    group_cols = ["market", "line", "side"]
    for key, ev in eval_df.groupby(group_cols, dropna=False):
        market, line, side = key
        c = cal[
            cal["market"].eq(market)
            & ((cal["line"].isna() & pd.isna(line)) | cal["line"].eq(line))
            & cal["side"].eq(side)
        ]
        ev = ev.copy()
        model = _fit_isotonic(c["model_probability"], c["observed"]) if not c.empty else None
        if model is None:
            ev["calibrated_probability"] = ev["model_probability"].astype(float)
            method = "identity_insufficient_calibration_data"
        else:
            ev["calibrated_probability"] = model.predict(ev["model_probability"].astype(float).clip(0, 1))
            method = "isotonic"
        rows.append(
            {
                "market": market,
                "line": None if pd.isna(line) else float(line),
                "side": side,
                "method": method,
                "calibration_rows": int(len(c)),
                "evaluation_rows": int(len(ev)),
                "pre_brier": _brier_binary(ev["observed"].to_numpy(), ev["model_probability"].to_numpy()),
                "post_brier": _brier_binary(ev["observed"].to_numpy(), ev["calibrated_probability"].to_numpy()),
                "pre_log_loss": _binary_log_loss(ev["observed"].to_numpy(), ev["model_probability"].to_numpy()),
                "post_log_loss": _binary_log_loss(ev["observed"].to_numpy(), ev["calibrated_probability"].to_numpy()),
                "pre_avg_probability": float(ev["model_probability"].mean()),
                "post_avg_probability": float(ev["calibrated_probability"].mean()),
                "observed_rate": float(ev["observed"].mean()),
                "pre_calibration_gap": float(ev["model_probability"].mean() - ev["observed"].mean()),
                "post_calibration_gap": float(ev["calibrated_probability"].mean() - ev["observed"].mean()),
                "status": "available",
            }
        )
        out_parts.append(ev)
    return pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(), rows


def evaluate_calibration_layer(
    predictions: pd.DataFrame,
    *,
    total_goal_lines: tuple[float, ...] = DEFAULT_TOTAL_GOAL_LINES,
    calibration_fraction: float = 0.50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Temporal holdout calibration diagnostics for 1X2, totals and BTTS.

    This is a calibration layer evaluation, not a value-pick test. The first
    part of the backtest window is used for calibration and the later part is
    used to compare pre/post calibration metrics.
    """
    cal_pred, eval_pred = _split_calibration_eval(predictions, calibration_fraction=calibration_fraction)
    _, one_x_two_summary = _calibrate_1x2(cal_pred, eval_pred)

    cal_lines = _line_probability_rows(cal_pred, total_goal_lines=total_goal_lines)
    eval_lines = _line_probability_rows(eval_pred, total_goal_lines=total_goal_lines)
    calibrated_lines, binary_rows = _calibrate_binary_markets(cal_lines, eval_lines)

    detail_rows = []
    detail_rows.append(
        {
            "market": "1x2",
            "line": None,
            "side": "all",
            "method": one_x_two_summary.get("method", "not_available"),
            "calibration_rows": one_x_two_summary.get("calibration_rows", len(cal_pred)),
            "evaluation_rows": one_x_two_summary.get("evaluation_rows", len(eval_pred)),
            "pre_log_loss": one_x_two_summary.get("pre_calibration", {}).get("log_loss"),
            "post_log_loss": one_x_two_summary.get("post_calibration", {}).get("log_loss"),
            "pre_brier": one_x_two_summary.get("pre_calibration", {}).get("brier_multiclass"),
            "post_brier": one_x_two_summary.get("post_calibration", {}).get("brier_multiclass"),
            "pre_rps": one_x_two_summary.get("pre_calibration", {}).get("rps"),
            "post_rps": one_x_two_summary.get("post_calibration", {}).get("rps"),
            "status": one_x_two_summary.get("status", "not_available"),
        }
    )
    detail_rows.extend(binary_rows)
    detail = pd.DataFrame(detail_rows)

    summary = {
        "purpose": "post_model_probability_calibration_not_profit",
        "calibration_fraction": calibration_fraction,
        "calibration_rows": int(len(cal_pred)),
        "evaluation_rows": int(len(eval_pred)),
        "one_x_two": one_x_two_summary,
        "binary_market_rows": int(len(calibrated_lines)) if not calibrated_lines.empty else 0,
        "markets": sorted(calibrated_lines["market"].dropna().unique().tolist()) if not calibrated_lines.empty else [],
        "method": "temporal_holdout_isotonic_when_enough_data_else_identity",
    }
    return detail, summary


def estimate_dixon_coles_rho(
    predictions: pd.DataFrame,
    *,
    rho_grid: tuple[float, ...] = DEFAULT_DIXON_COLES_RHO_GRID,
    calibration_fraction: float = 0.50,
) -> tuple[float | None, dict[str, Any]]:
    cal, eval_df = _split_calibration_eval(predictions, calibration_fraction=calibration_fraction)
    required = {"home_goals", "away_goals", "lambda_home", "lambda_away"}
    cal = cal.dropna(subset=list(required)) if required.issubset(cal.columns) else pd.DataFrame()
    if len(cal) < 20:
        return None, {"status": "not_available", "reason": "insufficient_calibration_rows", "calibration_rows": int(len(cal))}
    best_rho: float | None = None
    best_loss = float("inf")
    for rho in rho_grid:
        losses = []
        for _, row in cal.iterrows():
            prob, *_ = _scoreline_rank(
                float(row["lambda_home"]),
                float(row["lambda_away"]),
                int(row["home_goals"]),
                int(row["away_goals"]),
                dixon_coles_rho=float(rho),
            )
            losses.append(-math.log(max(prob, 1e-15)))
        loss = float(np.mean(losses))
        if loss < best_loss:
            best_loss = loss
            best_rho = float(rho)
    info = {
        "status": "available",
        "method": "temporal_holdout_grid_search",
        "rho": best_rho,
        "calibration_rows": int(len(cal)),
        "evaluation_rows": int(len(eval_df)),
        "calibration_scoreline_log_loss": best_loss,
    }
    return best_rho, info


def evaluate_dixon_coles_scorelines(
    predictions: pd.DataFrame,
    *,
    calibration_fraction: float = 0.50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rho, info = estimate_dixon_coles_rho(predictions, calibration_fraction=calibration_fraction)
    _, eval_df = _split_calibration_eval(predictions, calibration_fraction=calibration_fraction)
    if rho is None or eval_df.empty:
        return pd.DataFrame(), info
    independent = evaluate_scorelines(eval_df, dixon_coles_rho=None)
    adjusted = evaluate_scorelines(eval_df, dixon_coles_rho=float(rho))
    summary = {
        **info,
        "independent_scoreline_metrics": _summarise_scoreline_eval(independent),
        "dixon_coles_scoreline_metrics": _summarise_scoreline_eval(adjusted),
        "principle": "low_score_distribution_calibration_not_value_betting",
    }
    return adjusted, summary


def evaluate_statistical_engine(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate simulator statistics from walk-forward prediction rows.

    This is intentionally separated from value/ROI evaluation. It measures
    whether the statistical engine predicts football distributions well enough
    for reports, simulations and later selective market research.
    """
    goal_errors = evaluate_goal_errors(predictions)
    line_eval = evaluate_goal_lines(predictions)
    line_calibration_bins = evaluate_goal_line_calibration_bins(predictions)
    scoreline_eval = evaluate_scorelines(predictions)
    calibration_detail, calibration_summary = evaluate_calibration_layer(predictions)
    dixon_coles_scorelines, dixon_coles_summary = evaluate_dixon_coles_scorelines(predictions)

    goal_metric = {
        str(row["metric_name"]): float(row["value"])
        for _, row in goal_errors.iterrows()
        if row.get("status") == "available"
    }
    summary: dict[str, Any] = {
        "evaluation_purpose": "statistical_engine_quality_not_profit",
        "bets_or_profit_evaluated": False,
        "n_predictions": int(len(predictions)),
        "goal_metrics": goal_metric,
        "scoreline_metrics": _summarise_scoreline_eval(scoreline_eval),
        "line_markets_evaluated": sorted(line_eval["market"].dropna().unique().tolist()) if not line_eval.empty else [],
        "calibration_layer": calibration_summary,
        "dixon_coles": dixon_coles_summary,
        "model_improvement_features": {
            "detailed_line_calibration": True,
            "calibration_layer_1x2_totals_btts": True,
            "dixon_coles_low_score_adjustment": True,
            "time_decay_and_shrinkage_supported_by_goal_model": True,
            "elo_features_present": True,
            "corners_cards_negative_binomial_planned": True,
        },
        "principles": {
            "statistical_engine_separate_from_value_pick_engine": True,
            "roi_not_used_for_model_selection": True,
            "offline_only": True,
        },
    }
    return (
        goal_errors,
        line_eval,
        line_calibration_bins,
        scoreline_eval,
        calibration_detail,
        dixon_coles_scorelines,
        summary,
    )
