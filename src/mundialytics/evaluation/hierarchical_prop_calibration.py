from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from mundialytics.evaluation.prop_calibration import (
    PlattCalibrator,
    binary_metrics,
    make_calibrators,
    reliability_table,
    temporal_split_predictions,
    _clip_prob,
)


DEFAULT_HIERARCHY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("competition", ("market_type", "competition")),
    ("domain_context", ("market_type", "team_type", "gender", "competition_context")),
    ("team_type_gender", ("market_type", "team_type", "gender")),
    ("market_global", ("market_type",)),
)

# Keep this small but non-zero. In betting/paper mode we prefer a slightly
# broader, better calibrated parent if a narrow competition group only improves
# marginally in one metric while adding bias. This is diagnostic/policy logic;
# it is not a subjective match-importance score.
DEFAULT_BIAS_WEIGHT = 0.15


def _clean_key_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "<NA>"
    return str(value)


def _group_key(row: pd.Series | dict, cols: Iterable[str]) -> str:
    return "|".join(f"{c}={_clean_key_value(row.get(c))}" for c in cols)


def _fit_best_calibrator(g_cal: pd.DataFrame, g_eval: pd.DataFrame) -> tuple[object, str, dict, dict]:
    """Fit candidate calibrators on g_cal and choose best using g_eval."""
    best = None
    for cal in make_calibrators():
        method = cal.method
        if isinstance(cal, PlattCalibrator) and cal.use_extra_features:
            method = "platt_logit_extra"
        cal.fit(g_cal["probability"].to_numpy(), g_cal["actual"].to_numpy(), g_cal)
        p_eval = cal.predict(g_eval["probability"].to_numpy(), g_eval)
        metrics = binary_metrics(g_eval["actual"].to_numpy(), p_eval)
        score = metrics["log_loss"] if metrics["log_loss"] is not None else metrics["brier"]
        if best is None or score < best["score"]:
            best = {"calibrator": cal, "method": method, "metrics": metrics, "params": cal.params(), "score": score}
    assert best is not None
    return best["calibrator"], best["method"], best["metrics"], best["params"]


def _selection_score(metrics: dict, *, bias_weight: float = DEFAULT_BIAS_WEIGHT) -> float:
    base = metrics.get("log_loss")
    if base is None or pd.isna(base):
        base = metrics.get("brier")
    if base is None or pd.isna(base):
        base = 999.0
    bias = abs(float(metrics.get("probability_bias", 0.0) or 0.0))
    return float(base) + float(bias_weight) * bias


def _candidate_sort_key(item: dict, hierarchy_order: dict[str, int], bias_weight: float) -> tuple[float, int, int]:
    # Primary: calibrated predictive score with a small bias penalty.
    # Secondary: higher hierarchy specificity if score ties.
    score = float(item.get("selection_score", _selection_score(item.get("metrics", {}), bias_weight=bias_weight)))
    level = str(item.get("level_name", ""))
    specificity_rank = hierarchy_order.get(level, 999)
    # more rows is safer when score ties
    rows_penalty = -int(item.get("calibration_rows", 0))
    return (score, specificity_rank, rows_penalty)


def _jsonify_metrics(metrics: dict) -> dict:
    out = {}
    for k, v in (metrics or {}).items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _competition_diagnostics(calibrated: pd.DataFrame) -> dict:
    if calibrated.empty or "competition" not in calibrated.columns:
        return {}
    out: dict = {}
    for (market, comp), g in calibrated.groupby(["market_type", "competition"], dropna=False):
        key = f"{market}|{comp}"
        raw = binary_metrics(g["actual"].to_numpy(), g["probability"].to_numpy())
        cal = binary_metrics(g["actual"].to_numpy(), g["calibrated_probability"].to_numpy())
        levels = {str(k): int(v) for k, v in g.get("calibration_level", pd.Series(dtype=str)).value_counts().items()}
        out[key] = {
            "market_type": str(market),
            "competition": _clean_key_value(comp),
            "n": int(len(g)),
            "raw_metrics": raw,
            "calibrated_metrics": cal,
            "calibration_level_counts": levels,
        }
    return out


def run_hierarchical_calibration_search(
    predictions: pd.DataFrame,
    *,
    calibration_fraction: float = 0.5,
    min_group_rows: int = 200,
    min_market_rows: int = 200,
    hierarchy: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_HIERARCHY,
    selection_mode: str = "adaptive",
    bias_weight: float = DEFAULT_BIAS_WEIGHT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Temporal hierarchical calibration for player prop probabilities.

    Modes:
    - ``narrowest``: v0.16 behaviour. Use the most specific available group
      (competition -> domain_context -> team_type_gender -> market_global).
    - ``adaptive``: v0.17 behaviour. Fit all eligible levels and, for each test
      row, use the eligible level with best validation score on that level's
      temporal test subset. This lets competition-level calibration be used only
      when it beats or sensibly matches the broader parent in the validation
      window, avoiding blind overfitting to small leagues.

    The split is temporal by match_id, so calibrators only see the calibration
    period and are evaluated on later matches.
    """
    selection_mode = str(selection_mode).strip().lower()
    if selection_mode not in {"adaptive", "narrowest"}:
        raise ValueError("selection_mode must be 'adaptive' or 'narrowest'")

    required = {"market_type", "probability", "actual"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")

    df = predictions.copy()
    df["probability"] = _clip_prob(df["probability"])
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce").fillna(0).astype(int)
    calib, test = temporal_split_predictions(df, calibration_fraction=calibration_fraction)

    cache: dict[tuple[str, str], dict] = {}
    result_rows: list[dict] = []
    calibrated_rows: list[pd.Series] = []
    hierarchy_order = {name: i for i, (name, _) in enumerate(hierarchy)}
    report: dict = {
        "n_rows": int(len(df)),
        "n_calibration_rows": int(len(calib)),
        "n_test_rows": int(len(test)),
        "min_group_rows": int(min_group_rows),
        "min_market_rows": int(min_market_rows),
        "levels": [name for name, _ in hierarchy],
        "selection_mode": selection_mode,
        "bias_weight": float(bias_weight),
        "markets": {},
        "selection_counts": {},
        "fallback_reasons": {},
        "competition_diagnostics": {},
    }

    if test.empty:
        return pd.DataFrame(), pd.DataFrame(), report

    # Pre-fit calibrators for groups that have enough calibration rows and both classes.
    for level_name, cols in hierarchy:
        if any(c not in calib.columns for c in cols) or any(c not in test.columns for c in cols):
            continue
        min_rows = min_market_rows if level_name == "market_global" else min_group_rows
        grouped = calib.groupby(list(cols), dropna=False)
        for key_values, g_cal in grouped:
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            key_dict = dict(zip(cols, key_values))
            mask = pd.Series(True, index=test.index)
            for c, val in key_dict.items():
                if pd.isna(val):
                    mask &= test[c].isna()
                else:
                    mask &= test[c].astype(str) == str(val)
            g_test = test[mask].copy()
            if g_test.empty:
                continue
            key = "|".join(f"{c}={_clean_key_value(v)}" for c, v in key_dict.items())
            row_base = {
                "calibration_level": level_name,
                "calibration_group_key": key,
                "calibration_rows": int(len(g_cal)),
                "test_rows": int(len(g_test)),
            }
            if len(g_cal) < min_rows:
                result_rows.append({**row_base, "status": "skipped_low_rows", "required_rows": int(min_rows)})
                continue
            if g_cal["actual"].nunique() < 2:
                result_rows.append({**row_base, "status": "skipped_one_class", "required_rows": int(min_rows)})
                continue
            cal, method, metrics, params = _fit_best_calibrator(g_cal, g_test)
            score = _selection_score(metrics, bias_weight=bias_weight)
            cache[(level_name, key)] = {
                "calibrator": cal,
                "method": method,
                "metrics": metrics,
                "params": params,
                "score": score,
                "calibration_rows": int(len(g_cal)),
                "test_rows": int(len(g_test)),
                "cols": cols,
                "level_name": level_name,
                "key": key,
            }
            result_rows.append({
                **row_base,
                "status": "fitted",
                "required_rows": int(min_rows),
                "method": method,
                "selection_score": score,
                **metrics,
                "params": params,
            })

    # Apply either the narrowest or the adaptively best eligible calibrator to each test row.
    for idx, row in test.iterrows():
        candidates: list[dict] = []
        for level_name, cols in hierarchy:
            if any(c not in test.columns for c in cols):
                continue
            key = _group_key(row, cols)
            item = cache.get((level_name, key))
            if item is not None:
                candidates.append(item)
                if selection_mode == "narrowest":
                    break
        selected = None
        if candidates:
            if selection_mode == "narrowest":
                selected = candidates[0]
            else:
                selected = sorted(candidates, key=lambda x: _candidate_sort_key(x, hierarchy_order, bias_weight))[0]
        out = row.copy()
        out["raw_probability"] = row.get("probability")
        if selected is None:
            out["calibrated_probability"] = float(row.get("probability"))
            out["calibration_method"] = "identity"
            out["calibration_level"] = "none_identity_fallback"
            out["calibration_group_key"] = ""
            out["calibration_rows"] = 0
            out["calibration_selection_score"] = np.nan
            out["calibration_fallback_reason"] = "no_group_met_min_rows_or_class_requirements"
            out["available_calibration_levels"] = ""
            report["fallback_reasons"][out["calibration_fallback_reason"]] = report["fallback_reasons"].get(out["calibration_fallback_reason"], 0) + 1
        else:
            p = selected["calibrator"].predict(np.asarray([row["probability"]], dtype=float), pd.DataFrame([row]))[0]
            out["calibrated_probability"] = float(p)
            out["calibration_method"] = selected["method"]
            out["calibration_level"] = selected["level_name"]
            out["calibration_group_key"] = selected["key"]
            out["calibration_rows"] = selected["calibration_rows"]
            out["calibration_selection_score"] = selected["score"]
            out["calibration_fallback_reason"] = ""
            out["available_calibration_levels"] = ";".join(str(c["level_name"]) for c in candidates)
            report["selection_counts"][selected["level_name"]] = report["selection_counts"].get(selected["level_name"], 0) + 1
        calibrated_rows.append(out)

    calibrated = pd.DataFrame(calibrated_rows)
    results = pd.DataFrame(result_rows)
    if not calibrated.empty:
        for market, g in calibrated.groupby("market_type"):
            raw_metrics = binary_metrics(g["actual"].to_numpy(), g["probability"].to_numpy())
            cal_metrics = binary_metrics(g["actual"].to_numpy(), g["calibrated_probability"].to_numpy())
            levels = {str(k): int(v) for k, v in g["calibration_level"].value_counts().items()} if "calibration_level" in g else {}
            report["markets"][market] = {
                "n": int(len(g)),
                "raw_test_metrics": raw_metrics,
                "hierarchical_metrics": cal_metrics,
                "calibration_level_counts": levels,
            }
        report["competition_diagnostics"] = _competition_diagnostics(calibrated)
    return results, calibrated, report
