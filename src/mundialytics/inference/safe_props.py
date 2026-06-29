from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from mundialytics.data.events import add_basic_event_metrics
from mundialytics.data.competition_taxonomy import enrich_competition_metadata
from mundialytics.identity.normalization import canonical_player_name, canonical_team_name
from mundialytics.data.provider_identity import attach_identity_map_to_lineups, load_identity_map, canonical_fallback_player_id
from mundialytics.evaluation.prop_calibration import (
    BasePropCalibrator,
    IdentityCalibrator,
    IsotonicPropCalibrator,
    PlattCalibrator,
    RateShiftCalibrator,
    _clip_prob,
)
from mundialytics.models.player_event_model import PlayerEventModel
from mundialytics.features.player_features import EVENT_COLUMNS

DEFAULT_MARKETS = (
    "player_shots",
    "player_shots_on_target",
    "player_fouls_committed",
    "player_yellow_card",
)

# Operational probability caps. These are not statistical claims; they are safety
# guards to prevent an aggressive calibrator (especially isotonic) from emitting
# near-certainty on football props.
MARKET_MAX_CAPS = {
    "player_shots": 0.95,
    "player_fouls_committed": 0.95,
    "player_shots_on_target": 0.75,
    "player_yellow_card": 0.45,
}
MARKET_MIN_CAPS = {
    "player_shots": 0.02,
    "player_fouls_committed": 0.02,
    "player_shots_on_target": 0.01,
    "player_yellow_card": 0.005,
}
LOW_SAMPLE_UPPER_CAP = 0.65
VERY_LOW_SAMPLE_UPPER_CAP = 0.50
LOW_SAMPLE_MINUTES = 270.0
VERY_LOW_SAMPLE_MINUTES = 90.0


CURRENT_LINEUP_REQUIRED_COLUMNS = {"match_id", "date", "team", "opponent", "player", "position", "expected_minutes", "started"}
CALIBRATION_HIERARCHY = (
    ("competition", ("market_type", "competition")),
    ("domain_context", ("market_type", "team_type", "gender", "competition_context")),
    ("team_type_gender", ("market_type", "team_type", "gender")),
    ("market_global", ("market_type",)),
)


def _row_group_key(row: pd.Series | dict, cols: Iterable[str]) -> str:
    return "|".join(f"{c}={str(row.get(c, '<NA>'))}" for c in cols)


def validate_current_lineups(lineups: pd.DataFrame, strict: bool = True) -> list[str]:
    """Validate that inference candidates come from a current lineup/squad file.

    This is the operational gate that prevents retired historical players from
    being predicted. Historical events train the model; this file is the only
    allowed candidate set.
    """
    warnings: list[str] = []
    missing = sorted(CURRENT_LINEUP_REQUIRED_COLUMNS - set(lineups.columns))
    if missing and strict:
        raise ValueError(f"current lineups missing required columns: {missing}")
    if missing:
        warnings.append(f"lineups_missing_optional_runtime_columns={missing}")
    if "date" in lineups.columns:
        null_rate = float(pd.to_datetime(lineups["date"], errors="coerce").isna().mean()) if len(lineups) else 0.0
        if null_rate > 0.0 and strict:
            raise ValueError(f"current lineup date_null_rate={null_rate:.3f}; fix lineup dates before inference")
        if null_rate > 0.0:
            warnings.append(f"lineup_date_null_rate={null_rate:.3f}")
    if "expected_minutes" in lineups.columns:
        m = pd.to_numeric(lineups["expected_minutes"], errors="coerce")
        bad = int((m.isna() | (m <= 0) | (m > 130)).sum())
        if bad and strict:
            raise ValueError(f"current lineup has invalid expected_minutes rows={bad}")
        if bad:
            warnings.append(f"invalid_expected_minutes_rows={bad}")
    dup_key = [c for c in ["match_id", "team", "player"] if c in lineups.columns]
    if dup_key:
        dup = int(lineups.duplicated(dup_key).sum())
        if dup and strict:
            raise ValueError(f"current lineup duplicate candidates by {dup_key}: {dup}")
        if dup:
            warnings.append(f"duplicate_lineup_candidates={dup}")
    return warnings


def method_to_calibrator(method: str) -> BasePropCalibrator:
    method = str(method).strip().lower()
    if method in {"identity", "raw", "none"}:
        return IdentityCalibrator()
    if method == "rate_shift":
        return RateShiftCalibrator()
    if method == "platt_logit":
        return PlattCalibrator(use_extra_features=False)
    if method == "platt_logit_extra":
        return PlattCalibrator(use_extra_features=True)
    if method == "isotonic":
        return IsotonicPropCalibrator()
    raise ValueError(f"Unsupported calibration method: {method}")


def choose_best_methods(calibration_results: pd.DataFrame | None = None) -> dict[str, str]:
    """Return best calibration method by market from calibration_search_results.csv-like data.

    If no results are supplied, a conservative default is returned based on the
    latest validated run. The caller can still override methods explicitly.
    """
    default = {
        "player_fouls_committed": "isotonic",
        "player_shots": "isotonic",
        "player_shots_on_target": "isotonic",
        "player_yellow_card": "platt_logit_extra",
    }
    if calibration_results is None or calibration_results.empty:
        return default
    required = {"market_type", "method"}
    if not required.issubset(calibration_results.columns):
        return default
    score_cols = [c for c in ["log_loss", "brier"] if c in calibration_results.columns]
    if not score_cols:
        return default
    out = default.copy()
    work = calibration_results.copy()
    for c in score_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    for market, g in work.groupby("market_type"):
        g2 = g.sort_values(score_cols, na_position="last")
        if not g2.empty:
            out[str(market)] = str(g2.iloc[0]["method"])
    return out


def fit_market_calibrators(
    backtest_predictions: pd.DataFrame,
    best_methods: dict[str, str] | None = None,
    min_rows: int = 200,
) -> dict[str, BasePropCalibrator]:
    """Fit one calibrator per market using historical backtest predictions."""
    if backtest_predictions is None or backtest_predictions.empty:
        return {}
    required = {"market_type", "probability", "actual"}
    missing = required - set(backtest_predictions.columns)
    if missing:
        raise ValueError(f"calibration predictions missing required columns: {sorted(missing)}")
    best_methods = best_methods or choose_best_methods(None)
    calibrators: dict[str, BasePropCalibrator] = {}
    work = backtest_predictions.copy()
    work["probability"] = _clip_prob(work["probability"])
    work["actual"] = pd.to_numeric(work["actual"], errors="coerce").fillna(0).astype(int)
    for market, g in work.groupby("market_type"):
        if len(g) < min_rows or g["actual"].nunique() < 2:
            cal = IdentityCalibrator()
        else:
            method = best_methods.get(str(market), "identity")
            cal = method_to_calibrator(method)
            cal.fit(g["probability"].to_numpy(), g["actual"].to_numpy(), g)
        calibrators[str(market)] = cal
    return calibrators


def fit_hierarchical_calibrators(
    backtest_predictions: pd.DataFrame | None,
    best_methods: dict[str, str] | None = None,
    min_group_rows: int = 200,
    min_market_rows: int = 200,
) -> tuple[dict[tuple[str, str], BasePropCalibrator], dict[tuple[str, str], int]]:
    """Fit production-time calibrators by competition/domain with fallbacks.

    This uses all supplied historical calibration predictions. The hierarchy is
    selected at inference time from current lineup metadata. It does not learn a
    subjective importance score; it only respects objective competition labels.
    """
    if backtest_predictions is None or backtest_predictions.empty:
        return {}, {}
    required = {"market_type", "probability", "actual"}
    missing = required - set(backtest_predictions.columns)
    if missing:
        raise ValueError(f"calibration predictions missing required columns: {sorted(missing)}")
    work = enrich_competition_metadata(backtest_predictions.copy(), overwrite=True)
    work["probability"] = _clip_prob(work["probability"])
    work["actual"] = pd.to_numeric(work["actual"], errors="coerce").fillna(0).astype(int)
    best_methods = best_methods or choose_best_methods(None)
    cals: dict[tuple[str, str], BasePropCalibrator] = {}
    rows: dict[tuple[str, str], int] = {}
    for level, cols in CALIBRATION_HIERARCHY:
        if any(c not in work.columns for c in cols):
            continue
        min_rows = min_market_rows if level == "market_global" else min_group_rows
        for key_values, g in work.groupby(list(cols), dropna=False):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            key_dict = dict(zip(cols, key_values))
            key = "|".join(f"{c}={str(v)}" for c, v in key_dict.items())
            if len(g) < min_rows or g["actual"].nunique() < 2:
                continue
            market = str(key_dict.get("market_type", g["market_type"].iloc[0]))
            method = best_methods.get(market, "identity")
            cal = method_to_calibrator(method)
            cal.fit(g["probability"].to_numpy(), g["actual"].to_numpy(), g)
            cals[(level, key)] = cal
            rows[(level, key)] = int(len(g))
    return cals, rows


def select_hierarchical_calibrator(
    calibrators: dict[tuple[str, str], BasePropCalibrator],
    calibration_rows: dict[tuple[str, str], int],
    row: pd.Series | dict,
    market_type: str,
) -> tuple[BasePropCalibrator, str, str, int]:
    lookup = dict(row)
    lookup["market_type"] = market_type
    for level, cols in CALIBRATION_HIERARCHY:
        key = _row_group_key(lookup, cols)
        cal = calibrators.get((level, key))
        if cal is not None:
            return cal, level, key, calibration_rows.get((level, key), 0)
    return IdentityCalibrator(), "none_identity_fallback", "", 0




def _market_policy_for(market_calibration_policy: dict | None, market_type: str) -> dict:
    if not market_calibration_policy:
        return {}
    if "markets" in market_calibration_policy and isinstance(market_calibration_policy["markets"], dict):
        return market_calibration_policy["markets"].get(str(market_type), {}) or {}
    return market_calibration_policy.get(str(market_type), {}) or {}

def safe_cap_probability(market_type: str, probability: float, sample_size: float | int | None = None) -> tuple[float, list[str]]:
    """Clip calibrated probability with market-aware safety caps and low-sample guards."""
    warnings: list[str] = []
    p = float(_clip_prob([probability])[0])
    floor = MARKET_MIN_CAPS.get(market_type, 0.01)
    cap = MARKET_MAX_CAPS.get(market_type, 0.95)
    sample = 0.0 if sample_size is None or pd.isna(sample_size) else float(sample_size)
    if sample < VERY_LOW_SAMPLE_MINUTES:
        cap = min(cap, VERY_LOW_SAMPLE_UPPER_CAP)
        warnings.append(f"very_low_player_sample_minutes={sample:.0f}")
    elif sample < LOW_SAMPLE_MINUTES:
        cap = min(cap, LOW_SAMPLE_UPPER_CAP)
        warnings.append(f"low_player_sample_minutes={sample:.0f}")
    if p > cap:
        warnings.append(f"probability_capped_from_{p:.3f}_to_{cap:.3f}")
    if p < floor:
        warnings.append(f"probability_floored_from_{p:.3f}_to_{floor:.3f}")
    return float(np.clip(p, floor, cap)), warnings


def prepare_player_events_for_model(player_events: pd.DataFrame) -> pd.DataFrame:
    df = enrich_competition_metadata(player_events, overwrite=True)
    if "team" in df.columns:
        df["team"] = df["team"].map(canonical_team_name)
    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].map(canonical_team_name)
    if "player" in df.columns:
        df["player"] = df["player"].map(canonical_player_name)
    for col in EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return add_basic_event_metrics(df)


def predict_props_for_lineups(
    player_events: pd.DataFrame,
    lineups: pd.DataFrame,
    markets: Iterable[str] = DEFAULT_MARKETS,
    line: str = "1+",
    calibration_predictions: pd.DataFrame | None = None,
    calibration_results: pd.DataFrame | None = None,
    min_calibration_rows: int = 200,
    strict_lineup_contract: bool = False,
    use_hierarchical_calibration: bool = True,
    min_hierarchical_group_rows: int = 200,
    market_calibration_policy: dict | None = None,
    identity_map: pd.DataFrame | str | None = None,
) -> pd.DataFrame:
    """Predict props only for players explicitly supplied in current lineups.

    Historical/retired players may be present in player_events for training, but
    the output candidate set is strictly controlled by `lineups`.
    """
    hist = prepare_player_events_for_model(player_events)
    model = PlayerEventModel().fit(hist)

    methods = choose_best_methods(calibration_results)
    calibrators = fit_market_calibrators(calibration_predictions, methods, min_rows=min_calibration_rows) if calibration_predictions is not None else {}
    hierarchical_calibrators, hierarchical_rows = ({}, {})
    if use_hierarchical_calibration and calibration_predictions is not None:
        hierarchical_calibrators, hierarchical_rows = fit_hierarchical_calibrators(
            calibration_predictions, methods, min_group_rows=min_hierarchical_group_rows, min_market_rows=min_calibration_rows
        )

    lu = lineups.copy()
    id_map_df = load_identity_map(identity_map) if isinstance(identity_map, (str, bytes)) else identity_map
    if id_map_df is not None and not getattr(id_map_df, "empty", True):
        lu = attach_identity_map_to_lineups(lu, id_map_df)
    lu = enrich_competition_metadata(lu, overwrite=True)
    lineup_contract_warnings = validate_current_lineups(lu, strict=strict_lineup_contract)
    if "player" not in lu.columns:
        raise ValueError("lineups must contain a 'player' column")
    if "team" not in lu.columns:
        raise ValueError("lineups must contain a 'team' column")
    if "opponent" not in lu.columns:
        lu["opponent"] = None
    if "expected_minutes" not in lu.columns:
        if "minutes" in lu.columns:
            lu["expected_minutes"] = lu["minutes"]
        else:
            lu["expected_minutes"] = 90.0
    if "started" not in lu.columns:
        lu["started"] = 1
    for col in ["team", "opponent"]:
        lu[col] = lu[col].map(lambda x: canonical_team_name(x) if pd.notna(x) else x)
    lu["player"] = lu["player"].map(canonical_player_name)
    if "player_id_global" not in lu.columns:
        lu["player_id_global"] = lu["player"].map(canonical_fallback_player_id)
    else:
        lu["player_id_global"] = lu["player_id_global"].where(lu["player_id_global"].notna() & ~lu["player_id_global"].astype(str).str.lower().isin(["nan", "none", ""]), lu["player"].map(canonical_fallback_player_id))
    if "team_scope" not in lu.columns:
        lu["team_scope"] = "unknown"
    if "competition" not in lu.columns:
        lu["competition"] = "unknown"
    if "player_context_id" not in lu.columns:
        from mundialytics.identity.normalization import player_context_id
        lu["player_context_id"] = [player_context_id(p, t, s, c) for p, t, s, c in zip(lu["player"], lu["team"], lu["team_scope"], lu["competition"])]
    lu["expected_minutes"] = pd.to_numeric(lu["expected_minutes"], errors="coerce").fillna(90.0).clip(lower=1, upper=130)

    rows: list[dict] = []
    for _, r in lu.iterrows():
        player = str(r["player"])
        position = r.get("position") if pd.notna(r.get("position")) else None
        competition_context = r.get("competition_context") if pd.notna(r.get("competition_context")) else None
        team_type = r.get("team_type") if pd.notna(r.get("team_type")) else None
        expected_minutes = float(r["expected_minutes"])
        team_context = {}
        for ctx_col in ["elo_diff", "expected_possession"]:
            if ctx_col in lu.columns and pd.notna(r.get(ctx_col)):
                team_context[ctx_col] = r.get(ctx_col)
        player_id = r.get("player_id_global")
        player_match = model.resolve_player_identity(player, player_id)
        resolved_player_id = player_match.matched_player_id_global if player_match.status == "matched" and player_match.matched_player_id_global else player_id
        resolved_player_name = player_match.matched_player if player_match.status == "matched" and player_match.matched_player else player
        identity_warnings: list[str] = []
        if r.get("identity_map_status") == "unmatched":
            identity_warnings.append("identity_map_unmatched_using_name_fallback")
        elif r.get("identity_map_status") == "ambiguous":
            identity_warnings.append("identity_map_ambiguous_using_name_fallback")
        elif pd.notna(r.get("identity_map_status")) and str(r.get("identity_map_status")) not in {"matched", "no_identity_map", "None", "nan", ""}:
            identity_warnings.append(f"identity_map_status={r.get('identity_map_status')}")
        if player_match.status == "ambiguous":
            identity_warnings.append("ambiguous_player_identity_match")
        elif player_match.status == "unmatched":
            identity_warnings.append("unmatched_player_identity_using_prior")
        elif player_match.method not in {"exact_player_id", "exact_normalized_name"}:
            identity_warnings.append(f"player_identity_resolved_by_{player_match.method}")
        for market in markets:
            pred = model.predict_market(
                player,
                str(market),
                line,
                expected_minutes=expected_minutes,
                team_context=team_context or None,
                position=position,
                player_id_global=resolved_player_id,
                competition_context=competition_context,
                team_type=team_type,
            )
            policy = _market_policy_for(market_calibration_policy, str(market))
            policy_source = str(policy.get("recommended_source", "hierarchical" if use_hierarchical_calibration else "simple_market"))
            if policy_source == "simple_market" or (not hierarchical_calibrators):
                cal = calibrators.get(str(market), IdentityCalibrator())
                calibration_level = "market_global_simple"
                calibration_group_key = f"market_type={market}"
                calibration_rows_used = 0
            else:
                cal, calibration_level, calibration_group_key, calibration_rows_used = select_hierarchical_calibrator(
                    hierarchical_calibrators, hierarchical_rows, r, str(market)
                )
            extra = pd.DataFrame([{
                "expected_minutes": pred.expected_minutes,
                "sample_size": pred.sample_size,
                "expected_count": pred.expected_count,
            }])
            calibrated = float(cal.predict(np.array([pred.probability]), extra)[0])
            safe_p, warn = safe_cap_probability(str(market), calibrated, sample_size=pred.sample_size)
            confidence = "normal"
            if pred.sample_size < VERY_LOW_SAMPLE_MINUTES:
                confidence = "very_low_sample"
            elif pred.sample_size < LOW_SAMPLE_MINUTES:
                confidence = "low_sample"
            if str(market) == "player_yellow_card":
                confidence = "cautious" if confidence == "normal" else confidence
            sample_profile = model.player_sample_profile(player, player_id_global=resolved_player_id)
            cross_context_flag = bool(str(team_type) == "national_team" and sample_profile.get("club_minutes_sample", 0.0) > 0)
            rows.append({
                "match_id": r.get("match_id"),
                "date": r.get("date"),
                "competition": r.get("competition"),
                "team_scope": r.get("team_scope"),
                "team_type": r.get("team_type"),
                "competition_context": r.get("competition_context"),
                "gender": r.get("gender"),
                "team": r.get("team"),
                "opponent": r.get("opponent"),
                "player": player,
                "player_id_global": r.get("player_id_global"),
                "provider": r.get("provider"),
                "provider_player_id": r.get("provider_player_id"),
                "provider_player_name": r.get("provider_player_name"),
                "canonical_player_id": r.get("canonical_player_id"),
                "historical_player_id_global": r.get("historical_player_id_global"),
                "historical_player_name": r.get("historical_player_name"),
                "identity_map_status": r.get("identity_map_status"),
                "identity_map_method": r.get("identity_map_method"),
                "identity_map_confidence": r.get("identity_map_confidence"),
                "identity_map_reason": r.get("identity_map_reason"),
                "resolved_player_id_global": resolved_player_id,
                "matched_player_name": resolved_player_name,
                "player_match_method": player_match.method,
                "player_match_confidence": player_match.confidence,
                "player_match_status": player_match.status,
                "player_context_id": r.get("player_context_id"),
                "position": position,
                "started": r.get("started"),
                "elo_diff": r.get("elo_diff"),
                "expected_possession": r.get("expected_possession"),
                "market_type": str(market),
                "line": str(line),
                "raw_probability": pred.probability,
                "calibrated_probability": calibrated,
                "safe_probability": safe_p,
                "calibration_method": getattr(cal, "method", "identity"),
                "calibration_level": calibration_level,
                "calibration_group_key": calibration_group_key,
                "calibration_rows": calibration_rows_used,
                "calibration_policy_source": policy_source,
                "calibration_policy_status": policy.get("readiness_status"),
                "calibration_policy_reason": policy.get("reason"),
                "expected_count": pred.expected_count,
                "expected_minutes": pred.expected_minutes,
                "sample_size": pred.sample_size,
                "club_minutes_sample": sample_profile.get("club_minutes_sample", 0.0),
                "national_minutes_sample": sample_profile.get("national_minutes_sample", 0.0),
                "cross_context_feature_used": cross_context_flag,
                "confidence_flag": confidence,
                "warnings": ";".join(list(lineup_contract_warnings) + identity_warnings + warn),
                "explanation": pred.explanation,
            })
    return pd.DataFrame(rows)
