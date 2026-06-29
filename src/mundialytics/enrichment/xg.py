from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mundialytics.data_quality.team_registry import normalize_provider_name, provider_alias_map


XG_ENRICHMENT_VERSION = "v0.49.6_xg_enrichment"

CANONICAL_XG_COLUMNS = [
    "provider",
    "provider_match_id",
    "date",
    "competition",
    "season",
    "home_team",
    "away_team",
    "home_xg",
    "away_xg",
    "home_npxg",
    "away_npxg",
    "xg_match_confidence",
]


@dataclass(frozen=True)
class XGEnrichmentOutputs:
    enriched_matches: pd.DataFrame
    canonical_xg: pd.DataFrame
    join_report: pd.DataFrame
    summary: dict[str, Any]


COLUMN_ALIASES = {
    "date": ["date", "match_date", "datetime", "kickoff"],
    "provider_match_id": ["provider_match_id", "match_id_provider", "understat_id", "id"],
    "competition": ["competition", "league", "div"],
    "season": ["season", "season_name"],
    "home_team": ["home_team", "home", "h_team", "h", "home_name"],
    "away_team": ["away_team", "away", "a_team", "a", "away_name"],
    "home_xg": ["home_xg", "home_xg_value", "hxg", "h_xg", "xg_home", "xgh", "h_xG"],
    "away_xg": ["away_xg", "away_xg_value", "axg", "a_xg", "xg_away", "xga", "a_xG"],
    "home_npxg": ["home_npxg", "home_npxg_value", "hnpxg", "h_npxg"],
    "away_npxg": ["away_npxg", "away_npxg_value", "anpxg", "a_npxg"],
}


def _pick_column(df: pd.DataFrame, canonical: str) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for alias in COLUMN_ALIASES.get(canonical, [canonical]):
        if alias in df.columns:
            return alias
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def canonicalize_xg_matches(xg: pd.DataFrame, *, provider: str = "unknown") -> pd.DataFrame:
    rows = pd.DataFrame()
    for canonical in CANONICAL_XG_COLUMNS:
        col = _pick_column(xg, canonical)
        if col is not None:
            rows[canonical] = xg[col]
        else:
            rows[canonical] = pd.NA

    rows["provider"] = rows["provider"].fillna(provider).astype(str)
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.date.astype("string")
    for col in ["home_team", "away_team", "competition", "season"]:
        rows[col] = rows[col].fillna("").astype(str)
    for col in ["home_xg", "away_xg", "home_npxg", "away_npxg"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows["xg_match_confidence"] = rows["xg_match_confidence"].fillna("provider_match")
    return rows[CANONICAL_XG_COLUMNS].copy()


def _build_match_key(df: pd.DataFrame, home_col: str = "home_team", away_col: str = "away_team") -> pd.Series:
    dates = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")
    home = df[home_col].map(normalize_provider_name)
    away = df[away_col].map(normalize_provider_name)
    return dates.fillna("") + "|" + home + "|" + away


def _apply_registry_aliases_to_xg(xg: pd.DataFrame, registry: pd.DataFrame, provider_column: str) -> pd.DataFrame:
    if registry.empty or provider_column not in registry.columns:
        return xg
    reverse: dict[str, str] = {}
    for _, row in registry.iterrows():
        canonical = normalize_provider_name(row.get("football_data_name") or row.get("canonical_team_name"))
        provider_name = normalize_provider_name(row.get(provider_column))
        if provider_name and canonical:
            reverse[provider_name] = canonical
    out = xg.copy()
    for side in ["home", "away"]:
        col = f"{side}_team"
        out[f"{col}_join_name"] = out[col].map(lambda v: reverse.get(normalize_provider_name(v), normalize_provider_name(v)))
    return out


def enrich_matches_with_xg(
    matches: pd.DataFrame,
    xg: pd.DataFrame,
    *,
    registry: pd.DataFrame | None = None,
    provider: str = "unknown",
    provider_alias_column: str = "understat_name",
    dataset_name: str = "xg_enriched_matches",
) -> XGEnrichmentOutputs:
    """Attach match-level xG from a canonical or provider CSV."""
    required = {"match_id", "date", "home_team", "away_team"}
    missing = sorted(required - set(matches.columns))
    if missing:
        summary = {
            "version": XG_ENRICHMENT_VERSION,
            "dataset_name": dataset_name,
            "status": "blocked",
            "missing_required_columns": missing,
        }
        return XGEnrichmentOutputs(matches.copy(), pd.DataFrame(columns=CANONICAL_XG_COLUMNS), pd.DataFrame(), summary)

    canonical_xg = canonicalize_xg_matches(xg, provider=provider)
    registry_df = registry if registry is not None else pd.DataFrame()
    canonical_xg = _apply_registry_aliases_to_xg(canonical_xg, registry_df, provider_alias_column)

    match_df = matches.copy()
    # Re-running enrichment should replace prior xG enrichment columns instead of
    # producing pandas suffixes such as home_xg_x/home_xg_y.
    match_df = match_df.drop(
        columns=[
            "home_xg",
            "away_xg",
            "home_npxg",
            "away_npxg",
            "xg_provider",
            "xg_available",
            "provider",
            "provider_match_id",
            "xg_match_confidence",
        ],
        errors="ignore",
    )
    match_df["date"] = pd.to_datetime(match_df["date"], errors="coerce")
    match_df["match_join_key"] = _build_match_key(match_df)
    if "home_team_join_name" in canonical_xg.columns:
        xg_key = (
            pd.to_datetime(canonical_xg["date"], errors="coerce").dt.date.astype("string").fillna("")
            + "|"
            + canonical_xg["home_team_join_name"].fillna("").astype(str)
            + "|"
            + canonical_xg["away_team_join_name"].fillna("").astype(str)
        )
    else:
        xg_key = _build_match_key(canonical_xg)
    canonical_xg = canonical_xg.copy()
    canonical_xg["match_join_key"] = xg_key

    # Prefer explicit match_id when available; otherwise date/team key.
    xg_by_id = canonical_xg.dropna(subset=["provider_match_id"]).copy()
    xg_by_key = canonical_xg.drop_duplicates("match_join_key", keep="first").copy()

    enriched = match_df.merge(
        xg_by_key[[
            "match_join_key",
            "provider",
            "provider_match_id",
            "home_xg",
            "away_xg",
            "home_npxg",
            "away_npxg",
            "xg_match_confidence",
        ]],
        on="match_join_key",
        how="left",
        validate="many_to_one",
        suffixes=("", "_xg_provider"),
    )
    enriched = enriched.drop(columns=["match_join_key"])
    enriched["xg_provider"] = enriched["provider"].where(enriched["home_xg"].notna() & enriched["away_xg"].notna(), pd.NA)
    enriched["xg_available"] = enriched["home_xg"].notna() & enriched["away_xg"].notna()

    join_report = match_df[["match_id", "date", "home_team", "away_team", "match_join_key"]].merge(
        xg_by_key[["match_join_key", "provider", "provider_match_id", "home_xg", "away_xg"]],
        on="match_join_key",
        how="left",
        validate="many_to_one",
    )
    join_report["join_status"] = join_report["home_xg"].notna() & join_report["away_xg"].notna()
    join_report["join_status"] = join_report["join_status"].map({True: "matched", False: "unmatched"})

    matched = int(enriched["xg_available"].sum())
    status = "ok" if matched == len(enriched) else ("warning" if matched > 0 else "blocked")
    summary = {
        "version": XG_ENRICHMENT_VERSION,
        "dataset_name": dataset_name,
        "status": status,
        "provider": provider,
        "input_match_rows": int(len(matches)),
        "input_xg_rows": int(len(canonical_xg)),
        "output_rows": int(len(enriched)),
        "matches_with_xg": matched,
        "coverage_rate": float(matched / len(enriched)) if len(enriched) else 0.0,
        "unmatched_examples": join_report.loc[join_report["join_status"] == "unmatched", ["match_id", "home_team", "away_team"]].head(20).to_dict("records"),
        "raw_data_changed": False,
        "model_logic_changed": False,
        "leakage_policy": "xg_columns_are_post_match_observations_in_canonical_matches_but_only_prior_rolling_xg_features_may_be_used_as_model_inputs",
    }
    return XGEnrichmentOutputs(enriched, canonical_xg, join_report, summary)
