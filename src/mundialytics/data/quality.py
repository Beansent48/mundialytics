from __future__ import annotations

import pandas as pd


def data_quality_report(matches: pd.DataFrame) -> dict:
    report = {
        "rows": int(len(matches)),
        "date_min": str(pd.to_datetime(matches["date"]).min().date()) if len(matches) else None,
        "date_max": str(pd.to_datetime(matches["date"]).max().date()) if len(matches) else None,
        "duplicated_match_ids": int(matches["match_id"].duplicated().sum()) if "match_id" in matches.columns else None,
        "scopes": sorted(matches.get("team_scope", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "unknown_scope_rows": int((matches.get("team_scope", pd.Series(["unknown"] * len(matches), index=matches.index)).astype(str) == "unknown").sum()),
        "competitions": sorted(matches.get("competition", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())[:25],
        "n_home_teams": int(matches["home_team"].nunique()) if "home_team" in matches.columns else None,
        "n_away_teams": int(matches["away_team"].nunique()) if "away_team" in matches.columns else None,
        "missing_home_goals": int(matches["home_goals"].isna().sum()) if "home_goals" in matches.columns else None,
        "missing_away_goals": int(matches["away_goals"].isna().sum()) if "away_goals" in matches.columns else None,
        "negative_goals": int(((pd.to_numeric(matches.get("home_goals", 0), errors="coerce") < 0) | (pd.to_numeric(matches.get("away_goals", 0), errors="coerce") < 0)).sum()),
    }
    warnings = []
    if report["duplicated_match_ids"]:
        warnings.append("duplicated_match_ids")
    if len(report["scopes"]) != 1:
        warnings.append("mixed_or_missing_scopes")
    if report["negative_goals"]:
        warnings.append("negative_goals")
    if report["missing_home_goals"] or report["missing_away_goals"]:
        warnings.append("incomplete_matches_present")
    report["warnings"] = warnings
    return report
