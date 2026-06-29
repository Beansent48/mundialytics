from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


MATCH_DATASET_FOUNDATION_VERSION = "v0.49.4_match_dataset_foundation"

REQUIRED_MATCH_COLUMNS = [
    "match_id",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "neutral",
]

NUMERIC_MATCH_COLUMNS = [
    "home_goals",
    "away_goals",
    "neutral",
    "home_shots",
    "away_shots",
    "home_sot",
    "away_sot",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
    "home_xg",
    "away_xg",
    "home_external_elo",
    "away_external_elo",
    "home_clubelo",
    "away_clubelo",
    "home_elo",
    "away_elo",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "goals": ["home_goals", "away_goals"],
    "shots": ["home_shots", "away_shots"],
    "shots_on_target": ["home_sot", "away_sot"],
    "corners": ["home_corners", "away_corners"],
    "fouls": ["home_fouls", "away_fouls"],
    "yellow_cards": ["home_yellow_cards", "away_yellow_cards"],
    "red_cards": ["home_red_cards", "away_red_cards"],
    "xg": ["home_xg", "away_xg"],
    "external_elo": ["home_external_elo", "away_external_elo"],
    "clubelo": ["home_clubelo", "away_clubelo"],
}

ANOMALY_COLUMNS = [
    "severity",
    "issue",
    "match_id",
    "date",
    "competition",
    "season",
    "home_team",
    "away_team",
    "evidence",
    "recommendation",
]

DROPPED_ROW_COLUMNS = [
    "row_index",
    "match_id",
    "reason",
    "evidence",
]


@dataclass(frozen=True)
class MatchDatasetFoundationOutputs:
    cleaned_matches: pd.DataFrame
    feature_coverage: pd.DataFrame
    quality_by_competition_season: pd.DataFrame
    anomalies: pd.DataFrame
    dropped_rows: pd.DataFrame
    summary: dict[str, Any]


def _empty_anomalies() -> pd.DataFrame:
    return pd.DataFrame(columns=ANOMALY_COLUMNS)


def _empty_dropped_rows() -> pd.DataFrame:
    return pd.DataFrame(columns=DROPPED_ROW_COLUMNS)


def _row_context(row: pd.Series, issue: str, severity: str, evidence: str, recommendation: str) -> dict[str, Any]:
    return {
        "severity": severity,
        "issue": issue,
        "match_id": row.get("match_id"),
        "date": row.get("date"),
        "competition": row.get("competition", "unknown"),
        "season": row.get("season", "unknown"),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _coverage_status(rate: float, required_present: bool) -> str:
    if not required_present or rate <= 0:
        return "unavailable"
    if rate >= 0.95:
        return "available"
    if rate >= 0.60:
        return "partial"
    return "sparse"


def build_feature_coverage(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = int(len(matches))
    for feature, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in matches.columns]
        missing = [c for c in cols if c not in matches.columns]
        if n == 0 or missing:
            rows.append({
                "feature_group": feature,
                "required_columns": ",".join(cols),
                "available_columns": ",".join(present),
                "missing_columns": ",".join(missing),
                "rows": n,
                "rows_with_full_group": 0,
                "coverage_rate": 0.0,
                "status": "unavailable",
                "notes": "required columns missing" if missing else "empty dataset",
            })
            continue
        full = int(matches[cols].notna().all(axis=1).sum())
        rate = float(full / n) if n else 0.0
        rows.append({
            "feature_group": feature,
            "required_columns": ",".join(cols),
            "available_columns": ",".join(present),
            "missing_columns": "",
            "rows": n,
            "rows_with_full_group": full,
            "coverage_rate": rate,
            "status": _coverage_status(rate, True),
            "notes": "",
        })
    return pd.DataFrame(rows)


def build_quality_by_competition_season(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=[
            "competition",
            "season",
            "rows",
            "date_min",
            "date_max",
            "n_teams",
            "goal_coverage_rate",
            "corners_coverage_rate",
            "cards_coverage_rate",
            "shots_coverage_rate",
            "sot_coverage_rate",
            "scope_values",
        ])

    df = matches.copy()
    df["competition"] = df.get("competition", "unknown").fillna("unknown").astype(str)
    df["season"] = df.get("season", "unknown").fillna("unknown").astype(str)
    rows: list[dict[str, Any]] = []
    for (competition, season), g in df.groupby(["competition", "season"], dropna=False):
        teams = pd.concat([g["home_team"], g["away_team"]], ignore_index=True).dropna().astype(str).nunique()
        def cov(cols: list[str]) -> float:
            if not all(c in g.columns for c in cols) or len(g) == 0:
                return 0.0
            return float(g[cols].notna().all(axis=1).mean())
        rows.append({
            "competition": competition,
            "season": season,
            "rows": int(len(g)),
            "date_min": str(pd.to_datetime(g["date"], errors="coerce").min().date()) if pd.to_datetime(g["date"], errors="coerce").notna().any() else None,
            "date_max": str(pd.to_datetime(g["date"], errors="coerce").max().date()) if pd.to_datetime(g["date"], errors="coerce").notna().any() else None,
            "n_teams": int(teams),
            "goal_coverage_rate": cov(["home_goals", "away_goals"]),
            "corners_coverage_rate": cov(["home_corners", "away_corners"]),
            "cards_coverage_rate": cov(["home_yellow_cards", "away_yellow_cards"]),
            "shots_coverage_rate": cov(["home_shots", "away_shots"]),
            "sot_coverage_rate": cov(["home_sot", "away_sot"]),
            "scope_values": ",".join(sorted(g.get("team_scope", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())),
        })
    return pd.DataFrame(rows).sort_values(["competition", "season"]).reset_index(drop=True)


def _detect_anomalies(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if matches.empty:
        return _empty_anomalies()

    df = matches.copy()
    for col in NUMERIC_MATCH_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for side in ["home", "away"]:
        goals = f"{side}_goals"
        if goals in df.columns:
            for _, row in df[df[goals].notna() & (df[goals] < 0)].iterrows():
                rows.append(_row_context(row, "negative_goals", "critical", f"{goals}={row.get(goals)}", "Drop or fix the row before training."))
            for _, row in df[df[goals].notna() & (df[goals] > 12)].iterrows():
                rows.append(_row_context(row, "extreme_goal_count", "warning", f"{goals}={row.get(goals)}", "Verify the score against the provider source."))

        shots = f"{side}_shots"
        sot = f"{side}_sot"
        if shots in df.columns and sot in df.columns:
            mask = df[shots].notna() & df[sot].notna() & (df[sot] > df[shots])
            for _, row in df[mask].iterrows():
                rows.append(_row_context(row, "shots_on_target_exceed_shots", "warning", f"{sot}={row.get(sot)} > {shots}={row.get(shots)}", "Check provider mapping before using shot features."))

        corners = f"{side}_corners"
        if corners in df.columns:
            mask = df[corners].notna() & (df[corners] > 25)
            for _, row in df[mask].iterrows():
                rows.append(_row_context(row, "extreme_corner_count", "warning", f"{corners}={row.get(corners)}", "Verify high corner count before fitting corner models."))

        yellow = f"{side}_yellow_cards"
        if yellow in df.columns:
            mask = df[yellow].notna() & (df[yellow] > 10)
            for _, row in df[mask].iterrows():
                rows.append(_row_context(row, "extreme_yellow_card_count", "warning", f"{yellow}={row.get(yellow)}", "Verify card data and referee/team context."))

    if "match_id" in df.columns:
        duplicated = df[df["match_id"].astype(str).duplicated(keep=False)]
        for _, row in duplicated.iterrows():
            rows.append(_row_context(row, "duplicated_match_id", "critical", f"match_id={row.get('match_id')}", "Ensure match_id is unique before training."))

    fixture_key_cols = ["date", "competition", "home_team", "away_team"]
    if all(c in df.columns for c in fixture_key_cols):
        duplicated_fixture = df[df.duplicated(subset=fixture_key_cols, keep=False)]
        for _, row in duplicated_fixture.iterrows():
            rows.append(_row_context(row, "duplicated_fixture_key", "warning", "same date+competition+home+away appears more than once", "Verify postponed/replayed fixture handling."))

    scopes = sorted(df.get("team_scope", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    if len(scopes) > 1:
        rows.append({
            "severity": "critical",
            "issue": "mixed_team_scopes",
            "match_id": None,
            "date": None,
            "competition": None,
            "season": None,
            "home_team": None,
            "away_team": None,
            "evidence": f"scopes={scopes}",
            "recommendation": "Train club and national models separately.",
        })

    return pd.DataFrame(rows, columns=ANOMALY_COLUMNS) if rows else _empty_anomalies()


def prepare_match_dataset(
    matches: pd.DataFrame,
    *,
    dataset_name: str = "match_dataset",
    drop_incomplete_goals: bool = False,
    deduplicate_match_ids: bool = True,
) -> MatchDatasetFoundationOutputs:
    """Clean and profile canonical match rows before modelling.

    This function is deliberately conservative. It does not invent missing values
    and it does not engineer model-specific features. It only coerces obvious
    types, removes structurally unsafe rows, reports feature coverage and flags
    anomalies that need review before using richer statistical markets.
    """
    required_missing = [c for c in REQUIRED_MATCH_COLUMNS if c not in matches.columns]
    if required_missing:
        summary = {
            "version": MATCH_DATASET_FOUNDATION_VERSION,
            "dataset_name": dataset_name,
            "status": "blocked",
            "input_rows": int(len(matches)),
            "output_rows": 0,
            "dropped_rows": 0,
            "warnings": ["missing_required_columns"],
            "missing_required_columns": required_missing,
            "model_logic_changed": False,
            "raw_data_changed": False,
        }
        return MatchDatasetFoundationOutputs(
            cleaned_matches=matches.copy(),
            feature_coverage=build_feature_coverage(matches),
            quality_by_competition_season=pd.DataFrame(),
            anomalies=_empty_anomalies(),
            dropped_rows=_empty_dropped_rows(),
            summary=summary,
        )

    df = matches.copy()
    input_rows = int(len(df))
    dropped: list[dict[str, Any]] = []

    if "source_file" not in df.columns:
        df["source_file"] = df.get("source", "unknown")

    df["match_id"] = df["match_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in NUMERIC_MATCH_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    unsafe_masks: list[tuple[str, pd.Series, str]] = [
        ("invalid_date", df["date"].isna(), "date could not be parsed"),
        ("missing_home_team", df["home_team"].isna() | (df["home_team"].astype(str).str.strip() == ""), "home_team missing"),
        ("missing_away_team", df["away_team"].isna() | (df["away_team"].astype(str).str.strip() == ""), "away_team missing"),
    ]
    if drop_incomplete_goals:
        unsafe_masks.extend([
            ("missing_home_goals", df["home_goals"].isna(), "home_goals missing"),
            ("missing_away_goals", df["away_goals"].isna(), "away_goals missing"),
        ])

    drop_idx: set[int] = set()
    for reason, mask, evidence in unsafe_masks:
        for idx, row in df[mask].iterrows():
            drop_idx.add(idx)
            dropped.append({
                "row_index": int(idx) if isinstance(idx, int) else str(idx),
                "match_id": row.get("match_id"),
                "reason": reason,
                "evidence": evidence,
            })

    if drop_idx:
        df = df.drop(index=list(drop_idx)).copy()

    if deduplicate_match_ids and not df.empty:
        dup_mask = df["match_id"].duplicated(keep="first")
        for idx, row in df[dup_mask].iterrows():
            dropped.append({
                "row_index": int(idx) if isinstance(idx, int) else str(idx),
                "match_id": row.get("match_id"),
                "reason": "duplicated_match_id_kept_first",
                "evidence": "same match_id already appeared earlier in the cleaned dataset",
            })
        df = df[~dup_mask].copy()

    if "neutral" in df.columns:
        df["neutral"] = pd.to_numeric(df["neutral"], errors="coerce").fillna(0).astype(int)
    for col in ["competition", "season", "stage", "team_scope", "source", "source_file"]:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)

    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    feature_coverage = build_feature_coverage(df)
    quality_by_comp = build_quality_by_competition_season(df)
    anomalies = _detect_anomalies(df)
    dropped_df = pd.DataFrame(dropped, columns=DROPPED_ROW_COLUMNS) if dropped else _empty_dropped_rows()

    warnings: list[str] = []
    if not df.empty:
        scopes = sorted(df.get("team_scope", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    else:
        scopes = []
    if len(df) == 0:
        warnings.append("empty_cleaned_dataset")
    if dropped:
        warnings.append("rows_dropped")
    if not anomalies.empty:
        warnings.append("anomalies_detected")
    if len(scopes) != 1:
        warnings.append("mixed_or_missing_scopes")
    sparse_features = feature_coverage[
        feature_coverage["feature_group"].isin(["shots", "shots_on_target", "corners", "fouls", "yellow_cards"])
        & feature_coverage["status"].isin(["unavailable", "sparse"])
    ]["feature_group"].tolist()
    if sparse_features:
        warnings.append("sparse_event_feature_coverage")

    critical_count = int((anomalies.get("severity", pd.Series(dtype=str)) == "critical").sum()) if not anomalies.empty else 0
    status = "ok"
    if len(df) == 0 or critical_count > 0:
        status = "blocked"
    elif warnings:
        status = "warning"

    summary: dict[str, Any] = {
        "version": MATCH_DATASET_FOUNDATION_VERSION,
        "dataset_name": dataset_name,
        "status": status,
        "input_rows": input_rows,
        "output_rows": int(len(df)),
        "dropped_rows": int(len(dropped_df)),
        "anomaly_rows": int(len(anomalies)),
        "critical_anomaly_rows": critical_count,
        "warning_anomaly_rows": int((anomalies.get("severity", pd.Series(dtype=str)) == "warning").sum()) if not anomalies.empty else 0,
        "date_min": str(df["date"].min().date()) if not df.empty and df["date"].notna().any() else None,
        "date_max": str(df["date"].max().date()) if not df.empty and df["date"].notna().any() else None,
        "competitions": sorted(df.get("competition", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())[:50],
        "seasons": sorted(df.get("season", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())[:50],
        "team_scopes": scopes,
        "n_home_teams": int(df["home_team"].nunique()) if "home_team" in df.columns else 0,
        "n_away_teams": int(df["away_team"].nunique()) if "away_team" in df.columns else 0,
        "feature_coverage": {
            str(row["feature_group"]): {
                "coverage_rate": float(row["coverage_rate"]),
                "status": str(row["status"]),
                "rows_with_full_group": int(row["rows_with_full_group"]),
            }
            for _, row in feature_coverage.iterrows()
        },
        "warnings": warnings,
        "recommendations": [
            "Use this cleaned canonical dataset for historical validation.",
            "Prefer multi-season club datasets with time-decay rather than one-season training windows.",
            "Do not train club and national rows in the same model.",
            "Only open corners/cards models when their feature coverage is available or partial with enough rows.",
            "Keep raw files immutable; write cleaned/profiling outputs to data/processed or outputs.",
        ],
        "raw_data_changed": False,
        "model_logic_changed": False,
        "leakage_policy": "data_foundation_outputs_do_not_use_future match outcomes as pre-match features",
    }

    return MatchDatasetFoundationOutputs(
        cleaned_matches=df,
        feature_coverage=feature_coverage,
        quality_by_competition_season=quality_by_comp,
        anomalies=anomalies,
        dropped_rows=dropped_df,
        summary=summary,
    )
