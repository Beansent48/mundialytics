from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MODEL_LINE_COLUMNS = [
    "match_id",
    "date",
    "kickoff_utc",
    "home_team",
    "away_team",
    "competition",
    "season",
    "market_key",
    "market",
    "scope",
    "subject_team",
    "subject_player",
    "goalkeeper",
    "line",
    "side",
    "selection",
    "model_probability",
    "fair_odds",
    "min_acceptable_odds",
    "expected_stat",
    "settled_stat",
    "actual_win",
    "target_quality",
    "data_quality_flag",
    "saves_data_quality_flag",
    "signal_group",
    "confidence_label",
    "decision",
    "decision_reason_codes",
    "model_family",
]

ODDS_INPUT_COLUMNS = [
    "snapshot_time_utc",
    "bookmaker",
    "provider",
    "provider_event_id",
    "internal_match_id",
    "match_id",
    "date",
    "home_team",
    "away_team",
    "market_key",
    "market",
    "scope",
    "subject_team",
    "subject_player",
    "line",
    "side",
    "bookmaker_odds",
    "is_live",
    "source_url",
    "notes",
]

VALUE_EDGE_COLUMNS = [
    "snapshot_time_utc",
    "bookmaker",
    "provider",
    "match_id",
    "date",
    "home_team",
    "away_team",
    "market_key",
    "scope",
    "subject_team",
    "subject_player",
    "line",
    "side",
    "model_probability",
    "fair_odds",
    "min_acceptable_odds",
    "bookmaker_odds",
    "implied_probability",
    "edge",
    "ev",
    "value_label",
    "stake_virtual",
    "confidence_label",
    "decision",
    "reason_codes",
    "actual_win",
    "profit_1u",
]


def norm_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    text = str(value).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def norm_key(value: object) -> str:
    return norm_text(value).replace(" ", "_")


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def safe_probability(value: object) -> float:
    p = safe_float(value)
    if not math.isfinite(p):
        return float("nan")
    return min(max(p, 1e-6), 1.0 - 1e-6)


def fair_odds_from_probability(probability: object) -> float:
    p = safe_probability(probability)
    if not math.isfinite(p):
        return float("nan")
    return 1.0 / p


def min_acceptable_odds_from_probability(
    probability: object,
    *,
    min_ev: float = 0.03,
    min_edge: float = 0.02,
    commission: float = 0.0,
) -> float:
    """Smallest decimal odds that clears both EV and edge requirements.

    EV rule: p * odds - 1 >= min_ev  => odds >= (1 + min_ev) / p.
    Edge rule: p - 1/odds >= min_edge => odds >= 1 / (p - min_edge).
    The stricter of the two is used. Commission is handled conservatively by
    requiring a slightly higher gross price on exchange-like markets.
    """
    p = safe_probability(probability)
    if not math.isfinite(p):
        return float("nan")
    min_ev_odds = (1.0 + float(min_ev)) / p
    if p > float(min_edge):
        min_edge_odds = 1.0 / (p - float(min_edge))
    else:
        min_edge_odds = float("inf")
    required = max(min_ev_odds, min_edge_odds)
    if commission and math.isfinite(required):
        required = 1.0 + (required - 1.0) / max(1e-6, 1.0 - float(commission))
    return float(required)


def classify_fair_odds_bucket(odds: object) -> str:
    o = safe_float(odds)
    if not math.isfinite(o):
        return "unknown"
    if o < 1.15:
        return "1.01-1.15_ultra_conservative"
    if o < 1.30:
        return "1.15-1.30_conservative"
    if o < 1.60:
        return "1.30-1.60_usable"
    if o < 2.20:
        return "1.60-2.20_interesting"
    return "2.20+_aggressive"


def confidence_from_row(row: pd.Series | dict) -> str:
    decision = norm_text(row.get("decision"))
    target_quality = norm_text(row.get("target_quality"))
    gap = abs(safe_float(row.get("test_calibration_gap"), 0.0))
    n = safe_float(row.get("test_n"), 0.0)
    fair = safe_float(row.get("test_avg_fair_odds", row.get("fair_odds", float("nan"))))
    if decision == "candidate" and target_quality == "real target" and gap <= 0.03 and n >= 500 and fair >= 1.15:
        return "high"
    if decision == "candidate" and gap <= 0.05 and n >= 100:
        return "medium"
    if "too conservative" in decision:
        return "monitor_low_price"
    if "calibration" in decision:
        return "needs_calibration"
    return "low"


def standard_model_line_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a signal/prediction DataFrame into the API-ready model-line schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=MODEL_LINE_COLUMNS)
    work = df.copy()
    for col in ["match_id", "date", "home_team", "away_team", "team", "player", "goalkeeper", "market", "scope", "selection", "signal_group"]:
        if col not in work.columns:
            work[col] = ""
    if "market_key" not in work.columns:
        work["market_key"] = work["market"].map(norm_key)
    if "side" not in work.columns:
        work["side"] = work["selection"].map(norm_key)
    if "subject_team" not in work.columns:
        work["subject_team"] = work.get("team", "")
    if "subject_player" not in work.columns:
        player = work.get("player", pd.Series([""] * len(work), index=work.index)).astype("string").fillna("")
        gk = work.get("goalkeeper", pd.Series([""] * len(work), index=work.index)).astype("string").fillna("")
        work["subject_player"] = np.where(player.astype(str).str.len().gt(0), player, gk)
    if "model_probability" in work.columns:
        work["model_probability"] = pd.to_numeric(work["model_probability"], errors="coerce").clip(1e-6, 1 - 1e-6)
    else:
        work["model_probability"] = np.nan
    if "fair_odds" not in work.columns:
        work["fair_odds"] = work["model_probability"].map(fair_odds_from_probability)
    else:
        work["fair_odds"] = pd.to_numeric(work["fair_odds"], errors="coerce")
        work["fair_odds"] = work["fair_odds"].fillna(work["model_probability"].map(fair_odds_from_probability))
    for numeric_col in ["line", "expected_stat", "settled_stat", "actual_win", "min_acceptable_odds"]:
        if numeric_col in work.columns:
            work[numeric_col] = pd.to_numeric(work[numeric_col], errors="coerce")
    for col in MODEL_LINE_COLUMNS:
        if col not in work.columns:
            work[col] = "" if col not in {"line", "model_probability", "fair_odds", "min_acceptable_odds", "expected_stat", "settled_stat", "actual_win"} else np.nan
    return work[MODEL_LINE_COLUMNS].copy()


def standard_odds_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize provider/bookmaker odds into the universal odds schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=ODDS_INPUT_COLUMNS)
    work = df.copy()
    rename = {}
    for c in ["book_odds", "odds", "decimal_odds", "price"]:
        if c in work.columns:
            rename[c] = "bookmaker_odds"
            break
    for c in ["side", "selection", "over_under"]:
        if c in work.columns:
            rename.setdefault(c, "side")
            break
    if "internal_match_id" not in work.columns and "match_id" in work.columns:
        work["internal_match_id"] = work["match_id"]
    work = work.rename(columns=rename)
    if "match_id" not in work.columns and "internal_match_id" in work.columns:
        work["match_id"] = work["internal_match_id"]
    if "market_key" not in work.columns and "market" in work.columns:
        work["market_key"] = work["market"].map(norm_key)
    if "market" not in work.columns and "market_key" in work.columns:
        work["market"] = work["market_key"]
    if "side" not in work.columns:
        work["side"] = ""
    for col in ODDS_INPUT_COLUMNS:
        if col not in work.columns:
            work[col] = "" if col not in {"line", "bookmaker_odds"} else np.nan
    work["bookmaker_odds"] = pd.to_numeric(work["bookmaker_odds"], errors="coerce")
    work["line"] = pd.to_numeric(work["line"], errors="coerce")
    return work[ODDS_INPUT_COLUMNS].copy()


def join_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "match_id" in work.columns:
        work["_match_key"] = work["match_id"].astype("string").fillna("").map(norm_text)
    else:
        work["_match_key"] = ""
    work["_market_key"] = work.get("market_key", work.get("market", "")).map(norm_key)
    work["_side_key"] = work.get("side", work.get("selection", "")).map(norm_key)
    work["_scope_key"] = work.get("scope", "").map(norm_key)
    work["_team_key"] = work.get("subject_team", work.get("team", "")).astype("string").fillna("").map(norm_text)
    work["_player_key"] = work.get("subject_player", work.get("player", "")).astype("string").fillna("").map(norm_text)
    work["_line_key"] = pd.to_numeric(work.get("line", np.nan), errors="coerce").round(3).fillna(-9999.0)
    return work


def merge_model_lines_with_odds(model_lines: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Attach bookmaker prices to model lines using the universal join keys.

    This is intentionally strict on market/side/line and flexible on empty subject fields.
    For player/team markets, subject fields should be filled by the provider adapter later.
    """
    m = standard_model_line_frame(model_lines)
    o = standard_odds_input_frame(odds)
    if m.empty:
        return pd.DataFrame(columns=VALUE_EDGE_COLUMNS)
    if o.empty:
        out = m.copy()
        for col in ["snapshot_time_utc", "bookmaker", "provider", "bookmaker_odds", "implied_probability", "edge", "ev", "value_label", "stake_virtual", "profit_1u"]:
            out[col] = np.nan if col not in {"snapshot_time_utc", "bookmaker", "provider", "value_label"} else ""
        out["value_label"] = "no_odds"
        return out[[c for c in VALUE_EDGE_COLUMNS if c in out.columns]].copy()
    left = join_key_columns(m)
    right = join_key_columns(o)
    # Primary strict keys; subject fields are included only when odds actually provide them.
    base_keys = ["_match_key", "_market_key", "_side_key", "_line_key"]
    optional_keys = []
    if right["_scope_key"].astype(str).str.len().gt(0).any():
        optional_keys.append("_scope_key")
    if right["_team_key"].astype(str).str.len().gt(0).any():
        optional_keys.append("_team_key")
    if right["_player_key"].astype(str).str.len().gt(0).any():
        optional_keys.append("_player_key")
    keys = base_keys + optional_keys
    keep_right = keys + ["snapshot_time_utc", "bookmaker", "provider", "provider_event_id", "bookmaker_odds", "is_live", "source_url", "notes"]
    right_small = right[keep_right].dropna(subset=["bookmaker_odds"]).drop_duplicates(keys + ["bookmaker", "provider", "snapshot_time_utc"])
    merged = left.merge(right_small, on=keys, how="left", suffixes=("", "_odds"))
    merged["implied_probability"] = 1.0 / pd.to_numeric(merged["bookmaker_odds"], errors="coerce")
    merged["edge"] = pd.to_numeric(merged["model_probability"], errors="coerce") - merged["implied_probability"]
    merged["ev"] = pd.to_numeric(merged["model_probability"], errors="coerce") * pd.to_numeric(merged["bookmaker_odds"], errors="coerce") - 1.0
    merged["profit_1u"] = np.where(merged.get("actual_win", np.nan).eq(1), pd.to_numeric(merged["bookmaker_odds"], errors="coerce") - 1.0, -1.0)
    merged.loc[merged["bookmaker_odds"].isna(), ["implied_probability", "edge", "ev", "profit_1u"]] = np.nan
    merged["value_label"] = merged.apply(classify_value_row, axis=1)
    merged["stake_virtual"] = merged.apply(suggest_virtual_stake, axis=1)
    merged["reason_codes"] = merged.apply(value_reason_codes, axis=1)
    for col in VALUE_EDGE_COLUMNS:
        if col not in merged.columns:
            merged[col] = "" if col not in {"line", "model_probability", "fair_odds", "min_acceptable_odds", "bookmaker_odds", "implied_probability", "edge", "ev", "stake_virtual", "actual_win", "profit_1u"} else np.nan
    return merged[VALUE_EDGE_COLUMNS].copy()


def classify_value_row(row: pd.Series | dict, *, min_edge: float = 0.02, min_ev: float = 0.03) -> str:
    odds = safe_float(row.get("bookmaker_odds"))
    if not math.isfinite(odds):
        return "no_odds"
    if odds < 1.01:
        return "bad_odds"
    ev = safe_float(row.get("ev"))
    edge = safe_float(row.get("edge"))
    fair = safe_float(row.get("fair_odds"))
    min_acc = safe_float(row.get("min_acceptable_odds"))
    if math.isfinite(min_acc) and odds < min_acc:
        return "below_min_acceptable"
    if ev >= 0.08 and edge >= 0.04:
        return "high_value"
    if ev >= min_ev and edge >= min_edge:
        return "value"
    if math.isfinite(fair) and odds >= fair:
        return "fair_or_tiny_edge"
    return "no_value"


def suggest_virtual_stake(row: pd.Series | dict, bankroll_units: float = 100.0, max_stake: float = 1.0) -> float:
    label = str(row.get("value_label", ""))
    odds = safe_float(row.get("bookmaker_odds"))
    p = safe_probability(row.get("model_probability"))
    if label not in {"value", "high_value"} or not math.isfinite(odds) or odds <= 1 or not math.isfinite(p):
        return 0.0
    b = odds - 1.0
    q = 1.0 - p
    kelly = (b * p - q) / b if b > 0 else 0.0
    fraction = max(0.0, min(kelly * 0.25, max_stake / bankroll_units))
    stake = round(fraction * bankroll_units, 3)
    return float(stake)


def value_reason_codes(row: pd.Series | dict) -> str:
    codes: list[str] = []
    label = str(row.get("value_label", ""))
    odds = safe_float(row.get("bookmaker_odds"))
    min_acc = safe_float(row.get("min_acceptable_odds"))
    fair = safe_float(row.get("fair_odds"))
    if not math.isfinite(odds):
        codes.append("no_matching_bookmaker_odds")
    else:
        if math.isfinite(fair) and odds >= fair:
            codes.append("book_odds_above_fair")
        if math.isfinite(min_acc) and odds >= min_acc:
            codes.append("book_odds_above_min_acceptable")
        if label in {"value", "high_value"}:
            codes.append("positive_ev_and_edge")
    target_quality = norm_text(row.get("target_quality"))
    if target_quality:
        codes.append(f"target_quality_{target_quality.replace(' ', '_')}")
    conf = norm_text(row.get("confidence_label"))
    if conf:
        codes.append(f"confidence_{conf.replace(' ', '_')}")
    return ";".join(codes)


def write_contract_files(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=MODEL_LINE_COLUMNS).to_csv(out_dir / "model_market_lines_schema.csv", index=False)
    pd.DataFrame(columns=ODDS_INPUT_COLUMNS).to_csv(out_dir / "historical_odds_input_schema.csv", index=False)
    pd.DataFrame(columns=VALUE_EDGE_COLUMNS).to_csv(out_dir / "value_edges_schema.csv", index=False)
    contract = {
        "version": "v0.40_odds_ready_contract",
        "model_line_columns": MODEL_LINE_COLUMNS,
        "historical_odds_input_columns": ODDS_INPUT_COLUMNS,
        "value_edge_columns": VALUE_EDGE_COLUMNS,
        "join_keys": ["match_id", "market_key", "line", "side", "scope", "subject_team", "subject_player"],
        "fair_odds_formula": "fair_odds = 1 / model_probability",
        "ev_formula": "ev = model_probability * bookmaker_odds - 1",
        "edge_formula": "edge = model_probability - (1 / bookmaker_odds)",
        "note": "Provider-specific adapters should map raw odds into historical_odds_input.csv without changing model code.",
    }
    (out_dir / "odds_ready_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
