from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

EPS = 1e-6


def _clip_prob(p: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    arr = np.nan_to_num(arr, nan=0.5, posinf=1 - EPS, neginf=EPS)
    return np.clip(arr, EPS, 1 - EPS)


def _logit(p: np.ndarray | pd.Series) -> np.ndarray:
    p = _clip_prob(p)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _safe_log_loss(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    p = _clip_prob(p)
    return float(log_loss(y, np.vstack([1 - p, p]).T, labels=[0, 1]))


def binary_metrics(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series) -> dict:
    y_arr = np.asarray(y, dtype=int)
    p_arr = _clip_prob(p)
    pred = (p_arr >= 0.5).astype(int)
    return {
        "n": int(len(y_arr)),
        "actual_rate": float(y_arr.mean()) if len(y_arr) else 0.0,
        "avg_probability": float(p_arr.mean()) if len(p_arr) else 0.0,
        "probability_bias": float(p_arr.mean() - y_arr.mean()) if len(y_arr) else 0.0,
        "brier": float(brier_score_loss(y_arr, p_arr)) if len(np.unique(y_arr)) >= 1 else None,
        "log_loss": _safe_log_loss(y_arr, p_arr),
        "accuracy_at_0_5": float((pred == y_arr).mean()) if len(y_arr) else 0.0,
    }


@dataclass
class CalibrationResult:
    market_type: str
    method: str
    metrics: dict
    params: dict


class BasePropCalibrator:
    method: str = "base"

    def fit(self, p: np.ndarray, y: np.ndarray, extra: pd.DataFrame | None = None) -> "BasePropCalibrator":
        return self

    def predict(self, p: np.ndarray, extra: pd.DataFrame | None = None) -> np.ndarray:
        return _clip_prob(p)

    def params(self) -> dict:
        return {}


class IdentityCalibrator(BasePropCalibrator):
    method = "identity"


class RateShiftCalibrator(BasePropCalibrator):
    """Add an intercept shift in logit space so mean probability matches calibration rate."""

    method = "rate_shift"

    def __init__(self) -> None:
        self.shift_ = 0.0
        self.train_rate_ = 0.0
        self.train_avg_probability_ = 0.0

    def fit(self, p: np.ndarray, y: np.ndarray, extra: pd.DataFrame | None = None) -> "RateShiftCalibrator":
        p = _clip_prob(p)
        y = np.asarray(y, dtype=int)
        self.train_rate_ = float(np.mean(y))
        self.train_avg_probability_ = float(np.mean(p))
        self.shift_ = float(_logit([self.train_rate_])[0] - _logit([self.train_avg_probability_])[0])
        return self

    def predict(self, p: np.ndarray, extra: pd.DataFrame | None = None) -> np.ndarray:
        return _clip_prob(_sigmoid(_logit(p) + self.shift_))

    def params(self) -> dict:
        return {"shift": self.shift_, "train_rate": self.train_rate_, "train_avg_probability": self.train_avg_probability_}


class PlattCalibrator(BasePropCalibrator):
    """Logistic regression calibrator using logit(raw probability) and optional stability features."""

    method = "platt_logit"

    def __init__(self, use_extra_features: bool = False) -> None:
        self.use_extra_features = use_extra_features
        self.model_: LogisticRegression | None = None
        self.feature_cols_: list[str] = []

    def _make_X(self, p: np.ndarray, extra: pd.DataFrame | None = None) -> np.ndarray:
        base = _logit(p).reshape(-1, 1)
        if not self.use_extra_features or extra is None:
            self.feature_cols_ = ["logit_probability"]
            return base
        cols = []
        values = [base]
        for c in ["expected_minutes", "sample_size", "expected_count"]:
            if c in extra.columns:
                vals = pd.to_numeric(extra[c], errors="coerce").fillna(0).to_numpy(dtype=float).reshape(-1, 1)
                # Scale robustly enough for logistic regression.
                if c == "sample_size":
                    vals = np.log1p(vals) / 10.0
                elif c == "expected_minutes":
                    vals = vals / 90.0
                values.append(vals)
                cols.append(c)
        self.feature_cols_ = ["logit_probability"] + cols
        return np.hstack(values)

    def fit(self, p: np.ndarray, y: np.ndarray, extra: pd.DataFrame | None = None) -> "PlattCalibrator":
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            self.model_ = None
            return self
        X = self._make_X(p, extra)
        self.model_ = LogisticRegression(max_iter=1000, solver="lbfgs")
        self.model_.fit(X, y)
        return self

    def predict(self, p: np.ndarray, extra: pd.DataFrame | None = None) -> np.ndarray:
        if self.model_ is None:
            return _clip_prob(p)
        X = self._make_X(p, extra)
        return _clip_prob(self.model_.predict_proba(X)[:, 1])

    def params(self) -> dict:
        if self.model_ is None:
            return {"fallback": "identity"}
        return {
            "feature_cols": self.feature_cols_,
            "coef": self.model_.coef_.ravel().tolist(),
            "intercept": self.model_.intercept_.ravel().tolist(),
        }


class IsotonicPropCalibrator(BasePropCalibrator):
    method = "isotonic"

    def __init__(self) -> None:
        self.model_: IsotonicRegression | None = None

    def fit(self, p: np.ndarray, y: np.ndarray, extra: pd.DataFrame | None = None) -> "IsotonicPropCalibrator":
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2 or len(y) < 50:
            self.model_ = None
            return self
        self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
        self.model_.fit(_clip_prob(p), y)
        return self

    def predict(self, p: np.ndarray, extra: pd.DataFrame | None = None) -> np.ndarray:
        if self.model_ is None:
            return _clip_prob(p)
        return _clip_prob(self.model_.predict(_clip_prob(p)))

    def params(self) -> dict:
        if self.model_ is None:
            return {"fallback": "identity"}
        return {"n_thresholds": int(len(self.model_.X_thresholds_))}


def make_calibrators() -> list[BasePropCalibrator]:
    return [
        IdentityCalibrator(),
        RateShiftCalibrator(),
        PlattCalibrator(use_extra_features=False),
        PlattCalibrator(use_extra_features=True),
        IsotonicPropCalibrator(),
    ]


def reliability_table(df: pd.DataFrame, prob_col: str = "probability", actual_col: str = "actual", bins: int = 10) -> pd.DataFrame:
    tmp = df[[prob_col, actual_col]].copy()
    tmp[prob_col] = _clip_prob(tmp[prob_col])
    tmp[actual_col] = pd.to_numeric(tmp[actual_col], errors="coerce").fillna(0).astype(int)
    edges = np.linspace(0, 1, bins + 1)
    tmp["prob_bin"] = pd.cut(tmp[prob_col], bins=edges, include_lowest=True)
    rows = []
    for b, g in tmp.groupby("prob_bin", observed=False):
        if g.empty:
            continue
        rows.append({
            "prob_bin": str(b),
            "n": int(len(g)),
            "avg_probability": float(g[prob_col].mean()),
            "actual_rate": float(g[actual_col].mean()),
            "calibration_error": float(g[prob_col].mean() - g[actual_col].mean()),
        })
    return pd.DataFrame(rows)


def temporal_split_predictions(df: pd.DataFrame, calibration_fraction: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        sort_cols = ["date", "match_id", "market_type", "player"]
    else:
        sort_cols = [c for c in ["match_id", "market_type", "player"] if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols).reset_index(drop=True)
    # Split by match_id to avoid rows from same match in both calib and test.
    match_ids = work[["match_id"]].drop_duplicates()["match_id"].astype(str).tolist() if "match_id" in work.columns else []
    if len(match_ids) >= 2:
        cut = max(1, min(len(match_ids) - 1, int(len(match_ids) * calibration_fraction)))
        calib_ids = set(match_ids[:cut])
        test_ids = set(match_ids[cut:])
        return work[work["match_id"].astype(str).isin(calib_ids)].copy(), work[work["match_id"].astype(str).isin(test_ids)].copy()
    cut = max(1, min(len(work) - 1, int(len(work) * calibration_fraction)))
    return work.iloc[:cut].copy(), work.iloc[cut:].copy()


def run_market_calibration_search(
    predictions: pd.DataFrame,
    calibration_fraction: float = 0.5,
    min_market_rows: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = {"market_type", "probability", "actual"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")

    df = predictions.copy()
    df["probability"] = _clip_prob(df["probability"])
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce").fillna(0).astype(int)
    calib, test = temporal_split_predictions(df, calibration_fraction=calibration_fraction)

    results: list[dict] = []
    pred_parts: list[pd.DataFrame] = []
    market_report: dict = {
        "n_rows": int(len(df)),
        "n_calibration_rows": int(len(calib)),
        "n_test_rows": int(len(test)),
        "markets": {},
    }

    for market, g_test in test.groupby("market_type"):
        g_cal = calib[calib["market_type"] == market].copy()
        if len(g_test) == 0:
            continue
        raw_metrics = binary_metrics(g_test["actual"].to_numpy(), g_test["probability"].to_numpy())
        market_report["markets"][market] = {
            "calibration_rows": int(len(g_cal)),
            "test_rows": int(len(g_test)),
            "raw_test_metrics": raw_metrics,
            "warnings": [],
        }
        if len(g_cal) < min_market_rows:
            market_report["markets"][market]["warnings"].append(f"low_calibration_rows={len(g_cal)}, required>={min_market_rows}")
        if g_cal["actual"].nunique() < 2:
            market_report["markets"][market]["warnings"].append("calibration split has only one class; complex calibrators will fall back")

        best_for_market: dict | None = None
        calibrated_market_frames: dict[str, pd.DataFrame] = {}
        for cal in make_calibrators():
            # Need fresh object; make_calibrators provides fresh list.
            method = cal.method
            if isinstance(cal, PlattCalibrator) and cal.use_extra_features:
                method = "platt_logit_extra"
            cal.fit(g_cal["probability"].to_numpy(), g_cal["actual"].to_numpy(), g_cal)
            p_cal = cal.predict(g_test["probability"].to_numpy(), g_test)
            metrics = binary_metrics(g_test["actual"].to_numpy(), p_cal)
            row = {
                "market_type": market,
                "method": method,
                **metrics,
                "params": cal.params(),
            }
            results.append(row)
            out = g_test.copy()
            out["raw_probability"] = out["probability"]
            out["calibrated_probability"] = p_cal
            out["calibration_method"] = method
            calibrated_market_frames[method] = out
            # Primary objective: log_loss if available, otherwise brier.
            score = metrics["log_loss"] if metrics["log_loss"] is not None else metrics["brier"]
            if best_for_market is None or score < best_for_market["score"]:
                best_for_market = {"method": method, "score": score, "metrics": metrics, "params": cal.params()}

        if best_for_market is not None:
            market_report["markets"][market]["best_method"] = best_for_market["method"]
            market_report["markets"][market]["best_metrics"] = best_for_market["metrics"]
            market_report["markets"][market]["best_params"] = best_for_market["params"]
            pred_parts.append(calibrated_market_frames[best_for_market["method"]])

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Store params separately-friendly as string in CSV callers can json.dumps.
        results_df = results_df.sort_values(["market_type", "log_loss", "brier"], na_position="last").reset_index(drop=True)
    calibrated_predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return results_df, calibrated_predictions, market_report


def incoherence_checks(predictions: pd.DataFrame) -> dict:
    df = predictions.copy()
    report: dict = {"rows": int(len(df)), "warnings": [], "checks": {}}
    if df.empty:
        report["warnings"].append("empty predictions")
        return report
    if "probability" in df.columns:
        p = pd.to_numeric(df["probability"], errors="coerce")
        invalid = int(((p < 0) | (p > 1) | p.isna()).sum())
        report["checks"]["invalid_probability_rows"] = invalid
        if invalid:
            report["warnings"].append(f"invalid_probability_rows={invalid}")
    for col in ["actual", "expected_minutes", "sample_size", "expected_count", "actual_count"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            report["checks"][f"{col}_missing"] = int(vals.isna().sum())
            if col in {"expected_minutes", "sample_size", "expected_count", "actual_count"}:
                neg = int((vals < 0).sum())
                report["checks"][f"{col}_negative"] = neg
                if neg:
                    report["warnings"].append(f"{col}_negative={neg}")
    if {"market_type", "actual", "probability"}.issubset(df.columns):
        market_rows = []
        for market, g in df.groupby("market_type"):
            y = pd.to_numeric(g["actual"], errors="coerce").fillna(0).astype(int)
            p = _clip_prob(g["probability"])
            bias = float(p.mean() - y.mean())
            market_rows.append({"market_type": market, "n": int(len(g)), "actual_rate": float(y.mean()), "avg_probability": float(p.mean()), "bias": bias})
            if abs(bias) > 0.05:
                report["warnings"].append(f"{market}: avg_probability differs from actual_rate by {bias:+.3f}")
        report["checks"]["market_bias"] = market_rows
    if {"match_id", "player", "market_type"}.issubset(df.columns):
        dup_key = [c for c in ["match_id", "team", "player", "market_type", "line"] if c in df.columns]
        dup = int(df.duplicated(dup_key).sum())
        report["checks"]["duplicate_rows_by_" + "_".join(dup_key)] = dup
        if dup:
            report["warnings"].append(f"duplicate_rows_by_{'_'.join(dup_key)}={dup}")
    if "expected_minutes" in df.columns:
        m = pd.to_numeric(df["expected_minutes"], errors="coerce")
        over = int((m > 130).sum())
        report["checks"]["expected_minutes_over_130"] = over
        if over:
            report["warnings"].append(f"expected_minutes_over_130={over}")
    return report
