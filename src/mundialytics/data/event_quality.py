from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

REQUIRED_PLAYER_PROP_COLUMNS = {
    "player_shots": "shots",
    "player_shots_on_target": "shots_on_target",
    "player_fouls_committed": "fouls_committed",
    "player_fouls_drawn": "fouls_drawn",
    "player_yellow_card": "yellow_cards",
}

OPTIONAL_BUT_IMPORTANT_COLUMNS = [
    "minutes",
    "started",
    "replaced_by",
    "replacement_minute",
    "position",
    "team_scope",
    "competition",
    "source",
]

@dataclass(frozen=True)
class EventReadinessThresholds:
    min_matches: int = 50
    min_player_rows: int = 500
    min_total_events_per_market: int = 10
    min_minutes_coverage: float = 0.80


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def diagnose_player_event_dataset(
    player_events: pd.DataFrame,
    *,
    lineups: pd.DataFrame | None = None,
    required_markets: Iterable[str] | None = None,
    thresholds: EventReadinessThresholds | None = None,
) -> dict:
    """Return a strict coverage report for player-prop event data.

    This is deliberately stricter than the goal-model data quality report: if
    we want player props, event columns must be present and populated. We do not
    silently drop missing prop markets.
    """
    th = thresholds or EventReadinessThresholds()
    markets = list(required_markets or REQUIRED_PLAYER_PROP_COLUMNS.keys())
    df = player_events.copy()
    rows = int(len(df))
    n_matches = int(df["match_id"].astype(str).nunique()) if "match_id" in df.columns and rows else 0
    n_players = int(df["player"].astype(str).nunique()) if "player" in df.columns and rows else 0

    minutes_coverage = 0.0
    if "minutes" in df.columns and rows:
        minutes_coverage = float(_safe_num(df["minutes"]).notna().mean())

    market_checks: list[dict] = []
    all_market_ok = True
    for market in markets:
        col = REQUIRED_PLAYER_PROP_COLUMNS.get(market, market)
        exists = col in df.columns
        total = float(_safe_num(df[col]).fillna(0).sum()) if exists else 0.0
        non_null = int(_safe_num(df[col]).notna().sum()) if exists else 0
        passed = bool(exists and total >= th.min_total_events_per_market)
        if not passed:
            all_market_ok = False
        market_checks.append({
            "market": market,
            "column": col,
            "exists": exists,
            "non_null_rows": non_null,
            "total_events": total,
            "passed": passed,
            "detail": f"{col}: total={total:g}, required>={th.min_total_events_per_market}",
        })

    optional_coverage: dict[str, dict] = {}
    for col in OPTIONAL_BUT_IMPORTANT_COLUMNS:
        if col in df.columns:
            optional_coverage[col] = {
                "exists": True,
                "non_null_rows": int(df[col].notna().sum()),
                "coverage": float(df[col].notna().mean()) if rows else 0.0,
            }
        else:
            optional_coverage[col] = {"exists": False, "non_null_rows": 0, "coverage": 0.0}

    lineup_summary = None
    if lineups is not None:
        lu = lineups.copy()
        lineup_summary = {
            "rows": int(len(lu)),
            "matches": int(lu["match_id"].astype(str).nunique()) if "match_id" in lu.columns and len(lu) else 0,
            "players": int(lu["player"].astype(str).nunique()) if "player" in lu.columns and len(lu) else 0,
            "has_replacements": bool("replaced_by" in lu.columns and lu["replaced_by"].notna().any()),
            "has_replacement_minutes": bool("replacement_minute" in lu.columns and lu["replacement_minute"].notna().any()),
            "minutes_coverage": float(_safe_num(lu["minutes"]).notna().mean()) if "minutes" in lu.columns and len(lu) else 0.0,
        }

    checks = [
        {"check": "match_count", "passed": n_matches >= th.min_matches, "detail": f"matches={n_matches}, required>={th.min_matches}"},
        {"check": "player_rows", "passed": rows >= th.min_player_rows, "detail": f"rows={rows}, required>={th.min_player_rows}"},
        {"check": "minutes_coverage", "passed": minutes_coverage >= th.min_minutes_coverage, "detail": f"coverage={minutes_coverage:.1%}, required>={th.min_minutes_coverage:.1%}"},
        {"check": "required_markets", "passed": all_market_ok, "detail": "all required market columns must exist and contain events"},
    ]
    passed = all(c["passed"] for c in checks)
    return {
        "status": "EVENT_DATA_READY" if passed else "EVENT_DATA_NOT_READY",
        "passed": passed,
        "rows": rows,
        "matches": n_matches,
        "players": n_players,
        "minutes_coverage": minutes_coverage,
        "checks": checks,
        "market_checks": market_checks,
        "optional_coverage": optional_coverage,
        "lineup_summary": lineup_summary,
        "recommendation": (
            "Use for player-prop validation/paper mode. Still check provider definitions vs bookmaker settlement rules."
            if passed else
            "Do not validate player props with this file yet; supply real event data such as StatsBomb Open Data or Wyscout."
        ),
    }


def assert_event_data_ready(report: dict) -> None:
    if not report.get("passed"):
        failed = [c for c in report.get("checks", []) if not c.get("passed")]
        market_failed = [m for m in report.get("market_checks", []) if not m.get("passed")]
        details = "; ".join([c.get("detail", "") for c in failed] + [m.get("detail", "") for m in market_failed])
        raise ValueError(f"Event data is not ready for player props: {details}")
