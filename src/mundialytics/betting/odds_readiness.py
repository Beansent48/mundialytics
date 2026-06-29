from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from mundialytics.betting.odds_contract import (
    ODDS_INPUT_COLUMNS,
    join_key_columns,
    standard_model_line_frame,
    standard_odds_input_frame,
)


@dataclass(frozen=True)
class MarketRequirement:
    market_key: str
    scope: str
    priority: str
    subject_required: str
    historical_odds_need: str
    provider_difficulty: str
    notes: str


MARKET_REQUIREMENTS: list[MarketRequirement] = [
    MarketRequirement("1x2", "match", "core", "none", "required", "easy", "Usually covered by every football odds provider."),
    MarketRequirement("btts", "match", "core", "none", "required", "easy", "Usually covered by mainstream odds APIs."),
    MarketRequirement("goals", "match", "core", "none", "required", "easy", "Totals are usually covered; line granularity matters."),
    MarketRequirement("corners", "match", "priority", "none", "required", "medium", "Coverage varies by competition and provider."),
    MarketRequirement("team_corners", "team", "priority", "subject_team", "required", "medium", "Needs team-specific alternate lines."),
    MarketRequirement("yellow_cards", "match", "priority", "none", "required", "medium", "Card lines may be missing for lower-liquidity matches."),
    MarketRequirement("team_yellow_cards", "team", "priority", "subject_team", "required", "medium_hard", "One of the strongest model signals; must verify team-card markets exist."),
    MarketRequirement("fouls", "match", "priority", "none", "required", "hard", "Often not available in generic odds APIs."),
    MarketRequirement("team_fouls", "team", "priority", "subject_team", "required", "hard", "Strong signal but bookmaker/API coverage is the main risk."),
    MarketRequirement("shots", "match", "secondary", "none", "required", "medium", "Useful but model quality is weaker than cards/fouls."),
    MarketRequirement("team_shots", "team", "secondary", "subject_team", "required", "medium_hard", "Use carefully; range coverage showed undercoverage."),
    MarketRequirement("shots_on_target", "match", "priority", "none", "required", "medium", "Market availability varies; line matching must be strict."),
    MarketRequirement("team_shots_on_target", "team", "priority", "subject_team", "required", "medium_hard", "Strong under signal, but provider coverage must be checked."),
    MarketRequirement("player_shots", "player", "secondary", "subject_player", "required", "hard", "Needs lineup-safe player identity matching."),
    MarketRequirement("player_shots_on_target", "player", "secondary", "subject_player", "required", "hard", "Needs player props support and strict SOT rules."),
    MarketRequirement("player_fouls_committed", "player", "experimental", "subject_player", "required", "very_hard", "Likely missing unless provider has deep player props."),
    MarketRequirement("player_yellow_card", "player", "experimental", "subject_player", "required", "hard", "Often available only for selected matches."),
    MarketRequirement("goalkeeper_saves", "player", "priority", "subject_player", "required", "hard", "Promising signal; must confirm player-level save props exist."),
    MarketRequirement("team_goalkeeper_saves", "team", "priority", "subject_team", "required", "hard", "Can be easier to map than goalkeeper name if provider supports team saves."),
]


def market_requirements_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(x) for x in MARKET_REQUIREMENTS])


def file_readiness_report(paths: dict[str, str | Path]) -> dict:
    """Return a non-throwing readiness report for required local artifacts."""
    files: dict[str, dict] = {}
    all_present = True
    for name, value in paths.items():
        p = Path(value)
        exists = p.exists()
        is_file = p.is_file()
        size_bytes = p.stat().st_size if exists and is_file else 0
        files[name] = {
            "path": str(p),
            "exists": bool(exists),
            "is_file": bool(is_file),
            "size_bytes": int(size_bytes),
        }
        all_present = all_present and exists and is_file and size_bytes > 0
    return {"ready": bool(all_present), "files": files}


def _non_empty_rate(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return 0.0
    s = df[col].astype("string").fillna("").str.strip()
    return float(s.ne("").mean())


def summarize_odds_template(template: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Summarize what odds must be sourced from a provider/API."""
    odds = standard_odds_input_frame(template)
    req = market_requirements_frame()
    if odds.empty:
        empty_cols = [
            "market_key", "market", "scope", "side", "rows", "unique_matches", "unique_lines",
            "subject_team_fill_rate", "subject_player_fill_rate", "priority", "provider_difficulty",
            "subject_required", "readiness_flag",
        ]
        return pd.DataFrame(columns=empty_cols), {"rows": 0, "unique_matches": 0, "markets": 0, "ready_for_provider_quote_collection": False}
    group_cols = ["market_key", "market", "scope", "side"]
    for col in group_cols:
        if col not in odds.columns:
            odds[col] = ""
    rows = []
    for keys, part in odds.groupby(group_cols, dropna=False):
        market_key, market, scope, side = keys
        rows.append({
            "market_key": market_key,
            "market": market,
            "scope": scope,
            "side": side,
            "rows": int(len(part)),
            "unique_matches": int(part["match_id"].astype("string").nunique()) if "match_id" in part.columns else 0,
            "unique_lines": int(pd.to_numeric(part["line"], errors="coerce").nunique()),
            "subject_team_fill_rate": _non_empty_rate(part, "subject_team"),
            "subject_player_fill_rate": _non_empty_rate(part, "subject_player"),
        })
    out = pd.DataFrame(rows)
    out = out.merge(req, on=["market_key", "scope"], how="left")
    out["priority"] = out["priority"].fillna("unknown")
    out["provider_difficulty"] = out["provider_difficulty"].fillna("unknown")
    out["subject_required"] = out["subject_required"].fillna("unknown")
    out["readiness_flag"] = "ok"
    team_missing = out["subject_required"].eq("subject_team") & out["subject_team_fill_rate"].lt(0.99)
    player_missing = out["subject_required"].eq("subject_player") & out["subject_player_fill_rate"].lt(0.99)
    out.loc[team_missing, "readiness_flag"] = "missing_subject_team"
    out.loc[player_missing, "readiness_flag"] = "missing_subject_player"
    out = out.sort_values(["priority", "provider_difficulty", "rows"], ascending=[True, True, False], kind="mergesort")
    summary = {
        "rows": int(len(odds)),
        "unique_matches": int(odds["match_id"].astype("string").nunique()) if "match_id" in odds.columns else 0,
        "markets": int(out["market_key"].nunique()),
        "market_scope_sides": int(len(out)),
        "readiness_flags": out["readiness_flag"].value_counts(dropna=False).to_dict(),
        "provider_difficulty_rows": out.groupby("provider_difficulty")["rows"].sum().sort_values(ascending=False).to_dict(),
        "ready_for_provider_quote_collection": bool(out["readiness_flag"].eq("ok").all()),
    }
    return out, summary


def audit_historical_odds_coverage(model_lines: pd.DataFrame, historical_odds: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Audit how many model lines can be priced by mapped historical odds.

    The audit mirrors the strict contract join keys: match, market, side, line, plus
    scope/team/player when provided by the odds file.
    """
    m = standard_model_line_frame(model_lines)
    o = standard_odds_input_frame(historical_odds)
    if m.empty:
        empty = pd.DataFrame(columns=["market_key", "scope", "side", "model_lines", "priced_lines", "coverage_pct"])
        return empty, {"model_lines": 0, "odds_rows": int(len(o)), "priced_lines": 0, "coverage_pct": 0.0, "ready_for_value_backtest": False}
    left = join_key_columns(m)
    right = join_key_columns(o)
    base_keys = ["_match_key", "_market_key", "_side_key", "_line_key"]
    optional_keys: list[str] = []
    if not right.empty and right["_scope_key"].astype(str).str.len().gt(0).any():
        optional_keys.append("_scope_key")
    if not right.empty and right["_team_key"].astype(str).str.len().gt(0).any():
        optional_keys.append("_team_key")
    if not right.empty and right["_player_key"].astype(str).str.len().gt(0).any():
        optional_keys.append("_player_key")
    keys = base_keys + optional_keys
    if right.empty:
        left["_priced"] = False
    else:
        right_keys = right.dropna(subset=["bookmaker_odds"])[keys].drop_duplicates()
        priced_raw = left[keys].merge(right_keys.assign(_priced=True), on=keys, how="left")["_priced"]
        priced = priced_raw.eq(True)
        left["_priced"] = priced.to_numpy()
    group_cols = ["market_key", "scope", "side"]
    for col in group_cols:
        if col not in left.columns:
            left[col] = ""
    grouped = left.groupby(group_cols, dropna=False).agg(
        model_lines=("match_id", "size"),
        priced_lines=("_priced", "sum"),
    ).reset_index()
    grouped["coverage_pct"] = (grouped["priced_lines"] / grouped["model_lines"]).fillna(0.0)
    grouped = grouped.sort_values(["coverage_pct", "model_lines"], ascending=[True, False], kind="mergesort")
    summary = {
        "model_lines": int(len(m)),
        "odds_rows": int(len(o)),
        "priced_lines": int(left["_priced"].sum()),
        "coverage_pct": float(left["_priced"].mean()) if len(left) else 0.0,
        "join_keys_used": ["match_id", "market_key", "side", "line"] + [x.replace("_", "").replace("key", "") for x in optional_keys],
        "ready_for_value_backtest": bool(left["_priced"].mean() >= 0.80) if len(left) else False,
    }
    return grouped, summary
