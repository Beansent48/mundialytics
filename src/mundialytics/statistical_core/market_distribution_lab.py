from __future__ import annotations

"""Distribution and market-side evaluation lab for bookmaker-style stat markets.

v0.39 moves the decision layer away from one global hit-rate leaderboard.  It
scores each market side (for example ``corners_over`` or
``goalkeeper_saves_under``) by:

- exact/count prediction error from ``expected_stat`` vs ``settled_stat``;
- central interval/range coverage using a transparent Poisson approximation;
- over/under hit-rate and calibration by fair-odds bucket;
- line-distance/cushion buckets; and
- a decision matrix that avoids calling ultra-conservative high-hit-rate rows a
  betting edge.

The module deliberately stays model-agnostic: it can consume the massive
``settled_event_line_signals.csv`` produced by ``build_event_line_backtest.py``.
"""

import json
import math
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:  # scipy is available in the project env, but keep a fallback for portability.
    from scipy.stats import poisson
except Exception:  # pragma: no cover
    poisson = None


TEXT_COLS = {
    "match_id": "string",
    "date": "string",
    "market": "string",
    "scope": "string",
    "selection": "string",
    "over_under": "string",
    "signal_group": "string",
    "team": "string",
    "player": "string",
    "goalkeeper": "string",
    "competition": "string",
    "competition_context": "string",
    "team_type": "string",
    "gender": "string",
    "target_quality": "string",
    "data_quality_flag": "string",
    "saves_data_quality_flag": "string",
    "model_family": "string",
    "expected_components": "string",
}

FAIR_ODDS_BINS = [1.0, 1.15, 1.30, 1.60, 2.20, 3.50, 999.0]
FAIR_ODDS_LABELS = ["1.01-1.15_ultra_conservative", "1.15-1.30_conservative", "1.30-1.60_usable", "1.60-2.20_interesting", "2.20-3.50_aggressive", "3.50+_longshot"]
LINE_MARGIN_BINS = [-999.0, -2.0, -1.0, -0.25, 0.25, 1.0, 2.0, 999.0]
LINE_MARGIN_LABELS = ["negative_gt2", "negative_1_2", "negative_small", "near_line", "positive_0.25_1", "positive_1_2", "positive_gt2"]
QUALITY_SCORE = {
    "real_target": 1.0,
    "provider_real_target": 1.0,
    "match_total": 0.75,
    "derived_target": 0.65,
    "unknown_quality": 0.40,
    "": 0.30,
}


def read_line_signals(
    path: str | Path,
    *,
    min_model_probability: float = 0.0,
    markets: str | None = None,
    target_quality: str = "all",
    max_rows: int = 0,
    split_mode: str = "auto",
) -> pd.DataFrame:
    """Read and prefilter a potentially huge settled line-signal CSV."""
    p = Path(path)
    df = pd.read_csv(p, low_memory=False, dtype={k: v for k, v in TEXT_COLS.items()})
    return prepare_line_signals(
        df,
        min_model_probability=min_model_probability,
        markets=markets,
        target_quality=target_quality,
        max_rows=max_rows,
        split_mode=split_mode,
    )



def _contains_any(text: pd.Series, terms: list[str]) -> pd.Series:
    s = text.astype("string").fillna("").str.lower()
    out = pd.Series(False, index=s.index)
    for term in terms:
        out = out | s.str.contains(term, regex=False, na=False)
    return out


def infer_target_quality_from_flags(work: pd.DataFrame) -> pd.Series:
    """Infer missing/unknown target quality from source flags and scope.

    Older line-signal files labelled many team-level rows as ``unknown_quality`` even
    though the target was a direct boxscore/event stat.  This inference is deliberately
    conservative and only upgrades when the data/source flags make that clear.
    """
    base = work.get("target_quality", pd.Series([""] * len(work), index=work.index)).astype("string").fillna("").str.lower().str.strip()
    scope = work.get("scope", pd.Series([""] * len(work), index=work.index)).astype("string").fillna("").str.lower().str.strip()
    flag = work.get("data_quality_flag", pd.Series([""] * len(work), index=work.index)).astype("string").fillna("").str.lower()
    saves_flag = work.get("saves_data_quality_flag", pd.Series([""] * len(work), index=work.index)).astype("string").fillna("").str.lower()
    combined = flag + ";" + saves_flag
    out = base.copy()
    needs = out.isin(["", "nan", "none", "unknown", "unknown_quality"])

    derived = needs & _contains_any(combined, ["derived_saves", "sot_minus_goals", "proxy"])
    out.loc[derived] = "derived_target"

    realish = needs & _contains_any(combined, [
        "provider_boxscore_real_stats",
        "provider_fixture_stats_real_stats",
        "provider_player_goalkeeper_saves",
        "provider_saves_real",
        "raw_event",
        "raw_event_goalkeeper_saves",
        "real_events",
    ])
    out.loc[realish & scope.eq("match")] = "match_total"
    out.loc[realish & ~scope.eq("match")] = "real_target"
    out.loc[out.isin(["", "nan", "none", "unknown"])] = "unknown_quality"
    return out


def prepare_line_signals(
    df: pd.DataFrame,
    *,
    min_model_probability: float = 0.0,
    markets: str | None = None,
    target_quality: str = "all",
    max_rows: int = 0,
    split_mode: str = "auto",
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    for col in ["model_probability", "fair_odds", "line", "settled_stat", "expected_stat", "actual_win"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in ["market", "selection", "over_under", "signal_group", "target_quality", "scope"]:
        if col in work.columns:
            work[col] = work[col].astype("string").fillna("").str.lower().str.strip()
    # Upgrade target quality labels where older line-signal files had unknown_quality
    # but the data/source flags clearly identify real boxscore/event or derived targets.
    work["target_quality_original"] = work.get("target_quality", pd.Series([""] * len(work), index=work.index)).astype("string").fillna("")
    work["target_quality"] = infer_target_quality_from_flags(work)
    if "selection" not in work.columns and "over_under" in work.columns:
        work["selection"] = work["over_under"]
    if "signal_group" not in work.columns or work["signal_group"].eq("").all():
        work["signal_group"] = work["market"].astype(str) + "_" + work["selection"].astype(str)
    if "actual_win" not in work.columns or work["actual_win"].isna().all():
        if {"settled_stat", "line", "selection"}.issubset(work.columns):
            work["actual_win"] = np.where(
                work["selection"].eq("over"),
                (work["settled_stat"] > work["line"]).astype(float),
                (work["settled_stat"] < work["line"]).astype(float),
            )
    work = work.dropna(subset=["model_probability", "fair_odds", "line", "settled_stat", "expected_stat", "actual_win"], how="any")
    work = work[work["model_probability"].between(0.000001, 0.999999)].copy()
    if min_model_probability > 0:
        work = work[work["model_probability"].ge(float(min_model_probability))].copy()
    if markets:
        wanted = {m.strip().lower() for m in str(markets).replace(";", ",").split(",") if m.strip()}
        if wanted:
            work = work[work["market"].isin(wanted)].copy()
    if target_quality and str(target_quality).lower() not in {"", "all"} and "target_quality" in work.columns:
        wanted_q = {q.strip().lower() for q in str(target_quality).replace(";", ",").split(",") if q.strip()}
        work = work[work["target_quality"].isin(wanted_q)].copy()
    if max_rows and int(max_rows) > 0 and len(work) > int(max_rows):
        # Keep a deterministic high-signal sample per signal group so quick runs remain representative.
        work = work.sort_values("model_probability", ascending=False, kind="mergesort")
        if "signal_group" in work.columns:
            groups = max(1, work["signal_group"].nunique(dropna=False))
            per_group = max(1, int(max_rows) // groups)
            work = work.groupby("signal_group", dropna=False, group_keys=False).head(per_group).head(int(max_rows)).copy()
        else:
            work = work.head(int(max_rows)).copy()
    work["fair_odds_bucket"] = pd.cut(work["fair_odds"].clip(lower=1.000001), bins=FAIR_ODDS_BINS, labels=FAIR_ODDS_LABELS, include_lowest=True, right=True).astype(str)
    work["line_margin"] = np.where(work["selection"].eq("over"), work["expected_stat"] - work["line"], work["line"] - work["expected_stat"])
    work["line_margin_bucket"] = pd.cut(work["line_margin"], bins=LINE_MARGIN_BINS, labels=LINE_MARGIN_LABELS, include_lowest=True).astype(str)
    work["abs_error"] = (work["settled_stat"] - work["expected_stat"]).abs()
    work["squared_error"] = (work["settled_stat"] - work["expected_stat"]) ** 2
    work["split"] = assign_split(work, mode=split_mode)
    return work.reset_index(drop=True)


def _stable_unit(value: str) -> float:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16 ** 12)


def _hash_split_for_match_ids(match_ids: pd.Series, train_frac: float = 0.60, validation_frac: float = 0.20) -> pd.Series:
    vals = match_ids.astype(str).map(_stable_unit)
    return pd.Series(
        np.where(vals < train_frac, "train", np.where(vals < train_frac + validation_frac, "validation", "test")),
        index=match_ids.index,
        dtype="string",
    )


def chronological_split(df: pd.DataFrame, train_frac: float = 0.60, validation_frac: float = 0.20) -> pd.Series:
    """Chronological split with a safe hash fallback for missing dates.

    v0.39 put date-less StatsBomb rows at the end, which made all real-target rows
    land in test.  Date-less matches are now split deterministically by match_id hash
    instead of being silently treated as a future test-only block.
    """
    if df is None or df.empty:
        return pd.Series(dtype="string")
    match_ids = df.get("match_id", pd.Series(range(len(df)), index=df.index)).astype(str)
    dates = pd.to_datetime(df.get("date", pd.Series([pd.NaT] * len(df), index=df.index)), errors="coerce")
    tmp = pd.DataFrame({"date": dates, "match_id": match_ids})
    split_by_match: dict[str, str] = {}

    valid = tmp[tmp["date"].notna()].drop_duplicates("match_id").sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    n_valid = len(valid)
    if n_valid:
        train_end = int(max(1, round(n_valid * train_frac)))
        val_end = int(max(train_end + 1, round(n_valid * (train_frac + validation_frac))))
        val_end = min(val_end, n_valid)
        for i, mid in enumerate(valid["match_id"].astype(str)):
            split_by_match[mid] = "train" if i < train_end else ("validation" if i < val_end else "test")

    missing = tmp[tmp["date"].isna()].drop_duplicates("match_id")
    if len(missing):
        miss_splits = _hash_split_for_match_ids(missing["match_id"], train_frac=train_frac, validation_frac=validation_frac)
        for mid, sp in zip(missing["match_id"].astype(str), miss_splits.astype(str)):
            split_by_match[mid] = sp

    return match_ids.map(split_by_match).fillna("test").astype("string")


def stratified_hash_split(df: pd.DataFrame, train_frac: float = 0.60, validation_frac: float = 0.20) -> pd.Series:
    """Deterministic match-level hash split.

    Use this as a diagnostic split when date coverage is incomplete or when comparing
    target-quality/source slices that would be test-only chronologically.
    """
    if df is None or df.empty:
        return pd.Series(dtype="string")
    match_ids = df.get("match_id", pd.Series(range(len(df)), index=df.index)).astype(str)
    return _hash_split_for_match_ids(match_ids, train_frac=train_frac, validation_frac=validation_frac)


def assign_split(df: pd.DataFrame, mode: str = "auto") -> pd.Series:
    mode = str(mode or "auto").lower().strip()
    if mode in {"hash", "stratified_hash", "diagnostic_hash"}:
        return stratified_hash_split(df)
    if mode in {"chronological", "time", "date"}:
        return chronological_split(df)
    # auto: if too many rows have missing dates, use hash. Otherwise use chronological
    # with hash fallback for missing-date matches.
    dates = pd.to_datetime(df.get("date", pd.Series([pd.NaT] * len(df), index=df.index)), errors="coerce")
    missing_rate = float(dates.isna().mean()) if len(dates) else 1.0
    if missing_rate > 0.35:
        return stratified_hash_split(df)
    return chronological_split(df)


def unique_stat_predictions(signals: pd.DataFrame) -> pd.DataFrame:
    if signals is None or signals.empty:
        return pd.DataFrame()
    cols = [c for c in [
        "match_id", "date", "market", "scope", "team", "player", "goalkeeper", "competition", "target_quality",
        "model_family", "expected_components", "settled_stat", "expected_stat", "split"
    ] if c in signals.columns]
    # Same event/stat appears once per line/side. Keep one copy for exact/range evaluation.
    out = signals[cols].drop_duplicates().copy()
    for col in ["settled_stat", "expected_stat"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["settled_stat", "expected_stat"])
    out["abs_error"] = (out["settled_stat"] - out["expected_stat"]).abs()
    out["squared_error"] = (out["settled_stat"] - out["expected_stat"]) ** 2
    out["count_log_loss"] = [poisson_nll(mu, y) for mu, y in zip(out["expected_stat"], out["settled_stat"])]
    return out.reset_index(drop=True)


def poisson_nll(mu: float, y: float) -> float:
    try:
        m = max(0.01, min(80.0, float(mu)))
        k = max(0, int(round(float(y))))
        if poisson is not None:
            return float(-poisson.logpmf(k, m))
        return float(m - k * math.log(m) + math.lgamma(k + 1))
    except Exception:
        return float("nan")


def poisson_quantile(mu: float, q: float) -> int:
    m = max(0.01, min(80.0, float(mu)))
    if poisson is not None:
        return int(poisson.ppf(q, m))
    # Portable fallback.
    cdf = 0.0
    k = 0
    while k < 200:
        cdf += math.exp(-m) * (m ** k) / math.factorial(k)
        if cdf >= q:
            return k
        k += 1
    return k


def summarize_exact_error(preds: pd.DataFrame) -> pd.DataFrame:
    if preds is None or preds.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["split", "market", "scope", "target_quality", "model_family"] if c in preds.columns]
    out = preds.groupby(group_cols, dropna=False).agg(
        n=("settled_stat", "size"),
        avg_actual=("settled_stat", "mean"),
        avg_expected=("expected_stat", "mean"),
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda s: float(np.sqrt(np.nanmean(s)))),
        count_log_loss=("count_log_loss", "mean"),
    ).reset_index()
    out["mean_bias"] = out["avg_expected"] - out["avg_actual"]
    return out.sort_values(["split", "market", "scope", "target_quality"]).reset_index(drop=True)


def build_range_frame(preds: pd.DataFrame) -> pd.DataFrame:
    if preds is None or preds.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, r in preds.iterrows():
        mu = float(r["expected_stat"])
        actual = float(r["settled_stat"])
        common = {c: r.get(c, "") for c in ["match_id", "date", "market", "scope", "team", "player", "goalkeeper", "target_quality", "model_family", "split"] if c in preds.columns}
        for coverage, lo_q, hi_q in [(0.50, 0.25, 0.75), (0.60, 0.20, 0.80), (0.80, 0.10, 0.90)]:
            lo = poisson_quantile(mu, lo_q)
            hi = poisson_quantile(mu, hi_q)
            rows.append({
                **common,
                "nominal_coverage": coverage,
                "range_low": lo,
                "range_high": hi,
                "range_width": hi - lo,
                "actual_in_range": int(lo <= actual <= hi),
                "settled_stat": actual,
                "expected_stat": mu,
            })
    return pd.DataFrame(rows)


def summarize_range_coverage(range_frame: pd.DataFrame) -> pd.DataFrame:
    if range_frame is None or range_frame.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["split", "market", "scope", "target_quality", "nominal_coverage"] if c in range_frame.columns]
    out = range_frame.groupby(group_cols, dropna=False).agg(
        n=("actual_in_range", "size"),
        empirical_coverage=("actual_in_range", "mean"),
        avg_range_width=("range_width", "mean"),
        avg_expected=("expected_stat", "mean"),
        avg_actual=("settled_stat", "mean"),
    ).reset_index()
    out["coverage_gap"] = out["empirical_coverage"] - out["nominal_coverage"].astype(float)
    return out.sort_values(["split", "market", "scope", "target_quality", "nominal_coverage"]).reset_index(drop=True)


def _agg_hit(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.groupby(group_cols, dropna=False).agg(
        n=("actual_win", "size"),
        hit_rate=("actual_win", "mean"),
        avg_model_probability=("model_probability", "mean"),
        avg_fair_odds=("fair_odds", "mean"),
        avg_line_margin=("line_margin", "mean"),
        avg_expected_stat=("expected_stat", "mean"),
        avg_settled_stat=("settled_stat", "mean"),
        avg_line=("line", "mean"),
    ).reset_index()
    out["calibration_gap"] = out["hit_rate"] - out["avg_model_probability"]
    out["abs_calibration_gap"] = out["calibration_gap"].abs()
    return out


def summarize_fair_odds_buckets(signals: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["split", "market", "scope", "selection", "signal_group", "target_quality", "fair_odds_bucket"] if c in signals.columns]
    return _agg_hit(signals, group_cols).sort_values(group_cols).reset_index(drop=True) if group_cols else pd.DataFrame()


def summarize_line_margin(signals: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["split", "market", "scope", "selection", "signal_group", "target_quality", "line_margin_bucket"] if c in signals.columns]
    return _agg_hit(signals, group_cols).sort_values(group_cols).reset_index(drop=True) if group_cols else pd.DataFrame()


def summarize_market_side(signals: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["split", "market", "scope", "selection", "signal_group", "target_quality"] if c in signals.columns]
    return _agg_hit(signals, group_cols).sort_values(group_cols).reset_index(drop=True) if group_cols else pd.DataFrame()


def build_decision_matrix(signals: pd.DataFrame, min_sample: int = 100) -> pd.DataFrame:
    """Create robust market-side decision rows using validation/test split.

    The decision is intentionally conservative. High hit-rate alone is not enough:
    ultra-low fair odds, weak target quality, bad calibration and unstable validation/test
    performance are flagged.
    """
    if signals is None or signals.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["market", "scope", "selection", "signal_group", "target_quality", "fair_odds_bucket"] if c in signals.columns]
    by_split = _agg_hit(signals, ["split", *group_cols])
    if by_split.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, g in by_split.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))
        split_map = {str(r["split"]): r for _, r in g.iterrows()}
        val = split_map.get("validation")
        test = split_map.get("test")
        train = split_map.get("train")
        if test is None:
            continue
        val_n = int(val["n"]) if val is not None else 0
        test_n = int(test["n"])
        train_n = int(train["n"]) if train is not None else 0
        val_hit = float(val["hit_rate"]) if val is not None else np.nan
        test_hit = float(test["hit_rate"])
        val_prob = float(val["avg_model_probability"]) if val is not None else np.nan
        test_prob = float(test["avg_model_probability"])
        test_gap = float(test["calibration_gap"])
        stability_gap = abs(test_hit - val_hit) if math.isfinite(val_hit) else np.nan
        quality = str(key_dict.get("target_quality", "") or "").lower()
        qscore = QUALITY_SCORE.get(quality, 0.45)
        fair_bucket = str(key_dict.get("fair_odds_bucket", ""))
        avg_fair = float(test["avg_fair_odds"])
        decision = "needs_review"
        reason: list[str] = []
        if test_n < min_sample or val_n < min_sample:
            decision = "insufficient_data"
            reason.append("not_enough_validation_or_test_sample")
        if fair_bucket.startswith("1.01-1.15") or avg_fair < 1.15:
            if decision != "insufficient_data":
                decision = "too_conservative_monitor"
            reason.append("fair_odds_too_low_to_infer_value")
        if qscore < 0.55:
            if decision not in {"insufficient_data", "too_conservative_monitor"}:
                decision = "needs_data_quality"
            reason.append("weak_or_unknown_target_quality")
        if abs(test_gap) > 0.10:
            if decision not in {"insufficient_data", "too_conservative_monitor"}:
                decision = "needs_calibration"
            reason.append("calibration_gap_gt_10pp")
        if math.isfinite(stability_gap) and stability_gap > 0.12:
            if decision not in {"insufficient_data", "too_conservative_monitor"}:
                decision = "unstable_validation_test"
            reason.append("validation_test_gap_gt_12pp")
        if test_hit < 0.52:
            decision = "avoid"
            reason.append("test_hit_rate_near_or_below_random")
        if not reason:
            if test_hit >= 0.58 and abs(test_gap) <= 0.07 and qscore >= 0.55 and avg_fair >= 1.15:
                decision = "candidate"
                reason.append("stable_calibrated_signal")
            else:
                decision = "needs_calibration"
                reason.append("signal_present_but_not_clean_enough")
        # A transparent score for sorting, not a betting ROI score.
        score = (
            (test_hit - 0.50) * 100
            - abs(test_gap) * 65
            - (stability_gap if math.isfinite(stability_gap) else 0.05) * 35
            + math.log10(max(test_n, 1)) * 2.0
            + qscore * 4.0
        )
        if fair_bucket.startswith("1.01-1.15"):
            score -= 8.0
        rows.append({
            **key_dict,
            "train_n": train_n,
            "validation_n": val_n,
            "test_n": test_n,
            "validation_hit_rate": val_hit,
            "test_hit_rate": test_hit,
            "validation_avg_model_probability": val_prob,
            "test_avg_model_probability": test_prob,
            "test_calibration_gap": test_gap,
            "validation_test_hit_gap": stability_gap,
            "test_avg_fair_odds": avg_fair,
            "test_avg_line_margin": float(test["avg_line_margin"]),
            "target_quality_score": qscore,
            "decision": decision,
            "decision_score": score,
            "reason_codes": ";".join(dict.fromkeys(reason)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["decision_score", "test_n"], ascending=[False, False]).reset_index(drop=True)


def write_outputs(signals: pd.DataFrame, out_dir: str | Path, *, min_sample: int = 100, write_range_rows: bool = False, split_mode: str = "auto") -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds = unique_stat_predictions(signals)
    exact = summarize_exact_error(preds)
    ranges = build_range_frame(preds)
    range_summary = summarize_range_coverage(ranges)
    fair = summarize_fair_odds_buckets(signals)
    margin = summarize_line_margin(signals)
    side = summarize_market_side(signals)
    decision = build_decision_matrix(signals, min_sample=min_sample)

    exact.to_csv(out / "stat_prediction_error.csv", index=False)
    range_summary.to_csv(out / "range_coverage.csv", index=False)
    fair.to_csv(out / "fair_odds_bucket_performance.csv", index=False)
    margin.to_csv(out / "line_margin_performance.csv", index=False)
    side.to_csv(out / "market_side_performance.csv", index=False)
    decision.to_csv(out / "market_side_decision_matrix.csv", index=False)
    if write_range_rows:
        ranges.to_csv(out / "range_predictions_rows.csv", index=False)

    summary = {
        "version": "v0.39.1_distribution_market_side_lab",
        "split_mode": str(split_mode),
        "signals_rows": int(len(signals)),
        "unique_stat_predictions": int(len(preds)),
        "markets": signals.get("market", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "signal_groups": signals.get("signal_group", pd.Series(dtype=str)).value_counts(dropna=False).head(50).to_dict(),
        "target_quality": signals.get("target_quality", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "split_by_target_quality": signals.groupby(["target_quality", "split"], dropna=False).size().unstack(fill_value=0).to_dict() if {"target_quality", "split"}.issubset(signals.columns) else {},
        "decision_counts": decision.get("decision", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not decision.empty else {},
        "top_candidates": decision.head(20).to_dict(orient="records") if not decision.empty else [],
        "outputs": [
            "stat_prediction_error.csv",
            "range_coverage.csv",
            "fair_odds_bucket_performance.csv",
            "line_margin_performance.csv",
            "market_side_performance.csv",
            "market_side_decision_matrix.csv",
        ],
        "important_warning": "This evaluates statistical signal quality, not betting ROI. Use historical odds or paper tracking later for ROI/yield/CLV.",
    }
    (out / "market_distribution_lab_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
