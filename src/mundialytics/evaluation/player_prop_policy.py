from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MARKET_SAFE_CAPS = {
    "player_shots": {"min": 0.02, "max": 0.95},
    "player_fouls_committed": {"min": 0.02, "max": 0.95},
    "player_shots_on_target": {"min": 0.01, "max": 0.75},
    "player_yellow_card": {"min": 0.005, "max": 0.45},
}

MARKET_DEFAULT_STATUS = {
    "player_shots": "usable",
    "player_shots_on_target": "usable_caution",
    "player_fouls_committed": "caution",
    "player_yellow_card": "caution",
}


def _num(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _score(metrics: dict | None, *, bias_weight: float = 0.15) -> float:
    metrics = metrics or {}
    base = _num(metrics.get("log_loss"), None)
    if base is None:
        base = _num(metrics.get("brier"), 999.0)
    bias = abs(_num(metrics.get("probability_bias"), 0.0) or 0.0)
    return float(base) + bias_weight * bias


def _market_thresholds(market: str) -> dict:
    # These are readiness heuristics for paper mode, not guaranteed profitability.
    if market in {"player_shots", "player_fouls_committed"}:
        return {"green_log_loss": 0.64, "yellow_log_loss": 0.70, "green_bias": 0.025, "yellow_bias": 0.055}
    if market == "player_shots_on_target":
        return {"green_log_loss": 0.52, "yellow_log_loss": 0.58, "green_bias": 0.025, "yellow_bias": 0.055}
    if market == "player_yellow_card":
        return {"green_log_loss": 0.42, "yellow_log_loss": 0.50, "green_bias": 0.020, "yellow_bias": 0.050}
    return {"green_log_loss": 0.65, "yellow_log_loss": 0.75, "green_bias": 0.03, "yellow_bias": 0.06}


def readiness_status(market: str, metrics: dict) -> str:
    t = _market_thresholds(market)
    ll = _num(metrics.get("log_loss"), 999.0) or 999.0
    bias = abs(_num(metrics.get("probability_bias"), 999.0) or 999.0)
    if ll <= t["green_log_loss"] and bias <= t["green_bias"]:
        base = "green"
    elif ll <= t["yellow_log_loss"] and bias <= t["yellow_bias"]:
        base = "yellow"
    else:
        base = "orange"
    # Cards and fouls remain operationally noisy without referee/team-style data.
    if market == "player_yellow_card" and base == "green":
        return "yellow_caution"
    if market == "player_fouls_committed" and base == "green":
        return "yellow_caution"
    return base


def choose_market_calibration_policy(
    *,
    market: str,
    simple_metrics: dict | None,
    hierarchical_metrics: dict | None,
    simple_method: str | None = None,
    hierarchical_level_counts: dict | None = None,
    bias_weight: float = 0.15,
) -> dict:
    """Choose simple vs hierarchical calibration for an operational market.

    The policy is intentionally conservative: use hierarchy when it is better by
    score, or when it substantially reduces bias without a meaningful log-loss
    penalty. Otherwise keep the simpler market-level calibrator.
    """
    simple_metrics = simple_metrics or {}
    hierarchical_metrics = hierarchical_metrics or {}
    simple_score = _score(simple_metrics, bias_weight=bias_weight)
    h_score = _score(hierarchical_metrics, bias_weight=bias_weight)
    simple_ll = _num(simple_metrics.get("log_loss"), 999.0) or 999.0
    h_ll = _num(hierarchical_metrics.get("log_loss"), 999.0) or 999.0
    simple_bias = abs(_num(simple_metrics.get("probability_bias"), 999.0) or 999.0)
    h_bias = abs(_num(hierarchical_metrics.get("probability_bias"), 999.0) or 999.0)

    use_hier = False
    reason = "simple_lower_score"
    if hierarchical_metrics:
        if h_score <= simple_score + 0.002:
            use_hier = True
            reason = "hierarchical_lower_or_tied_score"
        elif h_bias + 0.02 < simple_bias and h_ll <= simple_ll + 0.03:
            use_hier = True
            reason = "hierarchical_bias_improvement_with_acceptable_log_loss"

    chosen = hierarchical_metrics if use_hier else simple_metrics
    chosen_source = "hierarchical" if use_hier else "simple_market"
    return {
        "market_type": market,
        "recommended_source": chosen_source,
        "recommended_simple_method": simple_method,
        "use_hierarchical": bool(use_hier),
        "reason": reason,
        "simple_score": simple_score,
        "hierarchical_score": h_score if hierarchical_metrics else None,
        "simple_metrics": simple_metrics,
        "hierarchical_metrics": hierarchical_metrics,
        "hierarchical_level_counts": hierarchical_level_counts or {},
        "chosen_metrics": chosen,
        "readiness_status": readiness_status(market, chosen),
        "operational_status_hint": MARKET_DEFAULT_STATUS.get(market, "review"),
        "safe_probability_caps": MARKET_SAFE_CAPS.get(market, {"min": 0.01, "max": 0.95}),
    }


def build_player_prop_policy(temporal_report: dict, hierarchical_report: dict | None = None) -> dict:
    calibration_report = temporal_report.get("calibration_report", {}) if temporal_report else {}
    simple_markets = calibration_report.get("markets", {}) if isinstance(calibration_report, dict) else {}
    # The temporal report may contain the hierarchical payload under outputs.
    if hierarchical_report is None:
        hierarchical_report = (
            temporal_report.get("outputs", {})
            .get("hierarchical", {})
            .get("markets", {})
        ) if temporal_report else {}
    h_markets = hierarchical_report.get("markets", hierarchical_report) if isinstance(hierarchical_report, dict) else {}
    markets = sorted(set(simple_markets.keys()) | set(h_markets.keys()))
    policy_rows = []
    for market in markets:
        s = simple_markets.get(market, {})
        h = h_markets.get(market, {})
        policy_rows.append(choose_market_calibration_policy(
            market=market,
            simple_metrics=s.get("best_metrics", {}),
            hierarchical_metrics=h.get("hierarchical_metrics", {}),
            simple_method=s.get("best_method"),
            hierarchical_level_counts=h.get("calibration_level_counts", {}),
        ))
    return {
        "status": "PLAYER_PROP_POLICY_COMPLETE",
        "markets": {row["market_type"]: row for row in policy_rows},
        "summary": {
            "markets": len(policy_rows),
            "hierarchical_selected": int(sum(1 for r in policy_rows if r["use_hierarchical"])),
            "simple_selected": int(sum(1 for r in policy_rows if not r["use_hierarchical"])),
        },
    }
