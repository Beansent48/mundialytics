from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mundialytics.data.loaders import to_long_team_rows
from mundialytics.features.team_features import add_pre_match_rolling_features
from mundialytics.ratings.elo import EloRater


MODEL_READY_SNAPSHOTS_VERSION = "v0.49.6_enriched_hybrid_model_ready_snapshots"

IDENTITY_COLUMNS = [
    "match_id",
    "date",
    "competition",
    "season",
    "stage",
    "team_scope",
    "home_team",
    "away_team",
    "neutral",
    "venue_country",
    "venue_city",
    "home_team_country",
    "away_team_country",
    "home_team_city",
    "away_team_city",
]

TARGET_COLUMNS = [
    "target_home_goals",
    "target_away_goals",
    "target_total_goals",
    "target_1x2",
    "target_btts",
    "target_home_xg",
    "target_away_xg",
    "target_home_npxg",
    "target_away_npxg",
    "target_home_xa",
    "target_away_xa",
    "target_home_shots",
    "target_away_shots",
    "target_home_sot",
    "target_away_sot",
    "target_home_shots_inside_box",
    "target_away_shots_inside_box",
    "target_home_shots_outside_box",
    "target_away_shots_outside_box",
    "target_home_header_shots",
    "target_away_header_shots",
    "target_home_left_foot_shots",
    "target_away_left_foot_shots",
    "target_home_right_foot_shots",
    "target_away_right_foot_shots",
    "target_home_corners",
    "target_away_corners",
    "target_home_fouls",
    "target_away_fouls",
    "target_home_yellow_cards",
    "target_away_yellow_cards",
    "target_home_red_cards",
    "target_away_red_cards",
    "target_home_possession",
    "target_away_possession",
    "target_home_field_tilt",
    "target_away_field_tilt",
]

LEAGUE_CONTEXT_COLUMNS = [
    "league_match_count_pre",
    "league_goal_rate_pre",
    "league_home_goal_rate_pre",
    "league_away_goal_rate_pre",
    "league_draw_rate_pre",
    "league_btts_rate_pre",
    "league_over25_rate_pre",
]

CALENDAR_CONTEXT_COLUMNS = [
    "home_rest_days_pre",
    "away_rest_days_pre",
    "rest_days_diff_pre",
    "season_match_index_pre",
    "season_progress_pre",
]

EXTERNAL_STRENGTH_COLUMNS = [
    "home_clubelo_pre",
    "away_clubelo_pre",
    "clubelo_diff_pre",
    "clubelo_available_pre",
    "home_external_elo_pre",
    "away_external_elo_pre",
    "external_elo_diff_pre",
    "external_elo_available_pre",
    "venue_advantage_side_pre",
    "home_advantage_factor_pre",
    "effective_neutral_pre",
]

TEAM_FEATURE_BASES = [
    "team_match_count_pre",
    "goals_for_last3",
    "goals_for_last5",
    "goals_for_last10",
    "goals_against_last3",
    "goals_against_last5",
    "goals_against_last10",
    "xg_for_last3",
    "xg_for_last5",
    "xg_for_last10",
    "xg_against_last3",
    "xg_against_last5",
    "xg_against_last10",
    "shots_for_last3",
    "shots_for_last5",
    "shots_for_last10",
    "shots_against_last3",
    "shots_against_last5",
    "shots_against_last10",
    "sot_for_last3",
    "sot_for_last5",
    "sot_for_last10",
    "sot_against_last3",
    "sot_against_last5",
    "sot_against_last10",
    "corners_for_last3",
    "corners_for_last5",
    "corners_for_last10",
    "corners_against_last3",
    "corners_against_last5",
    "corners_against_last10",
    "fouls_for_last3",
    "fouls_for_last5",
    "fouls_for_last10",
    "fouls_against_last3",
    "fouls_against_last5",
    "fouls_against_last10",
    "yellow_cards_for_last3",
    "yellow_cards_for_last5",
    "yellow_cards_for_last10",
    "yellow_cards_against_last3",
    "yellow_cards_against_last5",
    "yellow_cards_against_last10",
    "goal_diff_last5",
    "xg_diff_last5",
    "shot_diff_last5",
]


def _normalise_venue_token(value: Any) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _venue_value(row: pd.Series, *names: str) -> str:
    for name in names:
        if name in row and pd.notna(row.get(name)) and str(row.get(name)).strip():
            return str(row.get(name)).strip()
    return ""


def _venue_advantage_side(row: pd.Series) -> str:
    neutral = int(pd.to_numeric(pd.Series([row.get("neutral", 0)]), errors="coerce").fillna(0).iloc[0])
    if neutral == 0:
        return "home"

    venue_country = _normalise_venue_token(_venue_value(row, "venue_country", "host_country", "country"))
    venue_city = _normalise_venue_token(_venue_value(row, "venue_city", "city"))

    home_country = _normalise_venue_token(_venue_value(row, "home_team_country", "home_country"))
    away_country = _normalise_venue_token(_venue_value(row, "away_team_country", "away_country"))
    home_city = _normalise_venue_token(_venue_value(row, "home_team_city", "home_city"))
    away_city = _normalise_venue_token(_venue_value(row, "away_team_city", "away_city"))
    home_team = _normalise_venue_token(_venue_value(row, "home_team"))
    away_team = _normalise_venue_token(_venue_value(row, "away_team"))

    home_match = bool((venue_country and venue_country == home_country) or (venue_city and venue_city in {home_city, home_team}))
    away_match = bool((venue_country and venue_country == away_country) or (venue_city and venue_city in {away_city, away_team}))
    if home_match and not away_match:
        return "home"
    if away_match and not home_match:
        return "away"
    return "none"


def _venue_home_advantage_factor(row: pd.Series) -> float:
    side = _venue_advantage_side(row)
    if side == "home":
        return 1.0
    if side == "away":
        return -1.0
    return 0.0


@dataclass(frozen=True)
class ModelReadySnapshotsOutputs:
    snapshots: pd.DataFrame
    feature_contract: pd.DataFrame
    summary: dict[str, Any]


def _target_outcome(home_goals: Any, away_goals: Any) -> str | None:
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None
    hg = float(home_goals)
    ag = float(away_goals)
    if hg > ag:
        return "H"
    if hg < ag:
        return "A"
    return "D"


def _build_league_context(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.sort_values(["date", "match_id"]).copy()
    df["total_goals"] = pd.to_numeric(df["home_goals"], errors="coerce") + pd.to_numeric(df["away_goals"], errors="coerce")
    df["is_draw"] = (pd.to_numeric(df["home_goals"], errors="coerce") == pd.to_numeric(df["away_goals"], errors="coerce")).astype(float)
    df["is_btts"] = (
        (pd.to_numeric(df["home_goals"], errors="coerce") > 0)
        & (pd.to_numeric(df["away_goals"], errors="coerce") > 0)
    ).astype(float)
    df["is_over25"] = (df["total_goals"] > 2.5).astype(float)

    group_cols = ["competition"]
    if "team_scope" in df.columns:
        group_cols.append("team_scope")

    rows: list[pd.DataFrame] = []
    for _, g in df.groupby(group_cols, dropna=False, sort=False):
        g = g.sort_values(["date", "match_id"]).copy()
        completed = g[["home_goals", "away_goals"]].notna().all(axis=1).astype(int)
        count_pre = completed.cumsum().shift(1).fillna(0).astype(float)

        def prior_mean(col: str) -> pd.Series:
            values = pd.to_numeric(g[col], errors="coerce")
            sums = values.where(completed.astype(bool), 0.0).cumsum().shift(1).fillna(0.0)
            return sums / count_pre.replace(0, pd.NA)

        out = pd.DataFrame({
            "match_id": g["match_id"].values,
            "league_match_count_pre": count_pre.values,
            "league_goal_rate_pre": prior_mean("total_goals").values,
            "league_home_goal_rate_pre": prior_mean("home_goals").values,
            "league_away_goal_rate_pre": prior_mean("away_goals").values,
            "league_draw_rate_pre": prior_mean("is_draw").values,
            "league_btts_rate_pre": prior_mean("is_btts").values,
            "league_over25_rate_pre": prior_mean("is_over25").values,
        })
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["match_id", *LEAGUE_CONTEXT_COLUMNS])


def _build_elo_pre(matches: pd.DataFrame) -> pd.DataFrame:
    """Return Elo values known before each row, including neutral venue logic.

    Neutral fixtures remove scheduled home advantage unless the venue country/city
    implies a real home-country/home-city edge for one of the teams.
    """
    df = matches.sort_values(["date", "match_id"]).copy()
    rater = EloRater()
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_elo = rater.get(home)
        away_elo = rater.get(away)
        advantage_factor = float(_venue_home_advantage_factor(row))
        adjusted_home = home_elo + (rater.config.home_advantage * advantage_factor)
        rows.append({
            "match_id": row["match_id"],
            "home_elo_pre": home_elo,
            "away_elo_pre": away_elo,
            "elo_diff_pre": home_elo - away_elo,
            "venue_advantage_side_pre": _venue_advantage_side(row),
            "home_advantage_factor_pre": advantage_factor,
            "effective_neutral_pre": bool(advantage_factor == 0.0),
            "expected_home_score_elo_pre": rater.expected_score(adjusted_home, away_elo),
        })
        if pd.notna(row.get("home_goals")) and pd.notna(row.get("away_goals")):
            rater.update_match(row)
    return pd.DataFrame(rows)



def _wide_team_features(matches: pd.DataFrame) -> pd.DataFrame:
    long_rows = to_long_team_rows(matches)
    long_features = add_pre_match_rolling_features(long_rows)
    dynamic_features = [
        c for c in long_features.columns
        if c.endswith(("last3", "last5", "last10"))
        or c in {"team_match_count_pre", "goal_diff_last5", "xg_diff_last5", "shot_diff_last5"}
    ]
    available = sorted(set([c for c in TEAM_FEATURE_BASES if c in long_features.columns] + dynamic_features))
    rows = []
    for side, is_home in [("home", 1), ("away", 0)]:
        side_df = long_features[long_features["is_home"] == is_home][["match_id", *available]].copy()
        side_df = side_df.rename(columns={c: f"{side}_{c}" for c in available})
        rows.append(side_df)
    if not rows:
        return pd.DataFrame(columns=["match_id"])
    out = rows[0].merge(rows[1], on="match_id", how="outer", validate="one_to_one")
    return out



def _build_calendar_context(matches: pd.DataFrame) -> pd.DataFrame:
    """Build pre-match calendar features without using current match outcomes."""
    df = matches.sort_values(["date", "match_id"]).copy()
    long_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        for side, team_col in [("home", "home_team"), ("away", "away_team")]:
            long_rows.append({
                "match_id": row["match_id"],
                "date": row["date"],
                "season": row.get("season", "unknown"),
                "competition": row.get("competition", "unknown"),
                "side": side,
                "team": row.get(team_col),
            })
    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return pd.DataFrame(columns=["match_id", *CALENDAR_CONTEXT_COLUMNS])

    long_df = long_df.sort_values(["team", "date", "match_id"]).copy()
    long_df["previous_match_date"] = long_df.groupby("team")["date"].shift(1)
    long_df["rest_days_pre"] = (long_df["date"] - long_df["previous_match_date"]).dt.days

    side_parts = []
    for side in ["home", "away"]:
        part = long_df[long_df["side"] == side][["match_id", "rest_days_pre"]].copy()
        part = part.rename(columns={"rest_days_pre": f"{side}_rest_days_pre"})
        side_parts.append(part)

    out = side_parts[0].merge(side_parts[1], on="match_id", how="outer", validate="one_to_one")
    out["rest_days_diff_pre"] = (
        pd.to_numeric(out["home_rest_days_pre"], errors="coerce")
        - pd.to_numeric(out["away_rest_days_pre"], errors="coerce")
    )

    season_rows = []
    group_cols = ["competition", "season"]
    for _, g in df.groupby(group_cols, dropna=False, sort=False):
        g = g.sort_values(["date", "match_id"]).copy()
        idx_pre = pd.Series(range(len(g)), index=g.index, dtype="float")
        denom = max(len(g) - 1, 1)
        season_rows.append(pd.DataFrame({
            "match_id": g["match_id"].values,
            "season_match_index_pre": idx_pre.values,
            "season_progress_pre": (idx_pre / denom).values,
        }))
    season_context = pd.concat(season_rows, ignore_index=True) if season_rows else pd.DataFrame(columns=["match_id", "season_match_index_pre", "season_progress_pre"])
    out = out.merge(season_context, on="match_id", how="outer", validate="one_to_one")
    return out


def _build_external_strength_context(matches: pd.DataFrame) -> pd.DataFrame:
    """Expose optional provider ratings as pre-match features if present."""
    df = matches.copy()
    out = pd.DataFrame({"match_id": df["match_id"].astype(str)})
    for provider_col, out_col in [
        ("home_clubelo", "home_clubelo_pre"),
        ("away_clubelo", "away_clubelo_pre"),
        ("home_external_elo", "home_external_elo_pre"),
        ("away_external_elo", "away_external_elo_pre"),
    ]:
        if provider_col in df.columns:
            out[out_col] = pd.to_numeric(df[provider_col], errors="coerce")
        else:
            out[out_col] = pd.NA
    out["clubelo_diff_pre"] = pd.to_numeric(out["home_clubelo_pre"], errors="coerce") - pd.to_numeric(out["away_clubelo_pre"], errors="coerce")
    out["clubelo_available_pre"] = out["home_clubelo_pre"].notna() & out["away_clubelo_pre"].notna()
    out["external_elo_diff_pre"] = pd.to_numeric(out["home_external_elo_pre"], errors="coerce") - pd.to_numeric(out["away_external_elo_pre"], errors="coerce")
    out["external_elo_available_pre"] = out["home_external_elo_pre"].notna() & out["away_external_elo_pre"].notna()
    return out


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num / den.where(den.abs() > 1e-9)


def _add_derived_ratio_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable rolling conversion/pressure features from prior rolls."""
    out = snapshots.copy()
    for side in ["home", "away"]:
        for window in [3, 5, 10]:
            goals_for = out.get(f"{side}_goals_for_last{window}")
            shots_for = out.get(f"{side}_shots_for_last{window}")
            sot_for = out.get(f"{side}_sot_for_last{window}")
            xg_for = out.get(f"{side}_xg_for_last{window}")
            goals_against = out.get(f"{side}_goals_against_last{window}")
            shots_against = out.get(f"{side}_shots_against_last{window}")
            sot_against = out.get(f"{side}_sot_against_last{window}")
            xg_against = out.get(f"{side}_xg_against_last{window}")

            if goals_for is not None and shots_for is not None:
                out[f"{side}_goal_conversion_last{window}"] = _safe_ratio(goals_for, shots_for)
            if sot_for is not None and shots_for is not None:
                out[f"{side}_sot_rate_last{window}"] = _safe_ratio(sot_for, shots_for)
            if goals_for is not None and sot_for is not None:
                out[f"{side}_sot_conversion_last{window}"] = _safe_ratio(goals_for, sot_for)
            if xg_for is not None and shots_for is not None:
                out[f"{side}_xg_per_shot_last{window}"] = _safe_ratio(xg_for, shots_for)
            if goals_for is not None and xg_for is not None:
                out[f"{side}_goals_minus_xg_last{window}"] = pd.to_numeric(goals_for, errors="coerce") - pd.to_numeric(xg_for, errors="coerce")

            if goals_against is not None and shots_against is not None:
                out[f"{side}_defensive_goal_conversion_allowed_last{window}"] = _safe_ratio(goals_against, shots_against)
            if sot_against is not None and shots_against is not None:
                out[f"{side}_defensive_sot_rate_allowed_last{window}"] = _safe_ratio(sot_against, shots_against)
            if xg_against is not None and shots_against is not None:
                out[f"{side}_xg_against_per_shot_last{window}"] = _safe_ratio(xg_against, shots_against)
    return out


def _build_targets(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()
    out = pd.DataFrame({
        "match_id": df["match_id"].astype(str),
        "target_home_goals": pd.to_numeric(df.get("home_goals"), errors="coerce"),
        "target_away_goals": pd.to_numeric(df.get("away_goals"), errors="coerce"),
    })
    out["target_total_goals"] = out["target_home_goals"] + out["target_away_goals"]
    out["target_1x2"] = [
        _target_outcome(h, a) for h, a in zip(out["target_home_goals"], out["target_away_goals"], strict=False)
    ]
    out["target_btts"] = (
        (out["target_home_goals"] > 0)
        & (out["target_away_goals"] > 0)
    ).where(out[["target_home_goals", "target_away_goals"]].notna().all(axis=1), pd.NA)

    optional_pairs = [
        ("home_xg", "target_home_xg"),
        ("away_xg", "target_away_xg"),
        ("home_npxg", "target_home_npxg"),
        ("away_npxg", "target_away_npxg"),
        ("home_xa", "target_home_xa"),
        ("away_xa", "target_away_xa"),
        ("home_shots", "target_home_shots"),
        ("away_shots", "target_away_shots"),
        ("home_sot", "target_home_sot"),
        ("away_sot", "target_away_sot"),
        ("home_shots_inside_box", "target_home_shots_inside_box"),
        ("away_shots_inside_box", "target_away_shots_inside_box"),
        ("home_shots_outside_box", "target_home_shots_outside_box"),
        ("away_shots_outside_box", "target_away_shots_outside_box"),
        ("home_header_shots", "target_home_header_shots"),
        ("away_header_shots", "target_away_header_shots"),
        ("home_left_foot_shots", "target_home_left_foot_shots"),
        ("away_left_foot_shots", "target_away_left_foot_shots"),
        ("home_right_foot_shots", "target_home_right_foot_shots"),
        ("away_right_foot_shots", "target_away_right_foot_shots"),
        ("home_corners", "target_home_corners"),
        ("away_corners", "target_away_corners"),
        ("home_fouls", "target_home_fouls"),
        ("away_fouls", "target_away_fouls"),
        ("home_yellow_cards", "target_home_yellow_cards"),
        ("away_yellow_cards", "target_away_yellow_cards"),
        ("home_red_cards", "target_home_red_cards"),
        ("away_red_cards", "target_away_red_cards"),
        ("home_possession", "target_home_possession"),
        ("away_possession", "target_away_possession"),
        ("home_field_tilt", "target_home_field_tilt"),
        ("away_field_tilt", "target_away_field_tilt"),
    ]
    for source, target in optional_pairs:
        if source in df.columns:
            out[target] = pd.to_numeric(df[source], errors="coerce")
        else:
            out[target] = pd.NA
    return out



def _feature_contract(snapshots: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in snapshots.columns:
        if col in IDENTITY_COLUMNS:
            role = "identity"
            leakage_status = "safe_identity"
            notes = "Not a target and not a post-match statistic."
        elif col in TARGET_COLUMNS:
            role = "target"
            leakage_status = "post_match_target_not_feature"
            notes = "May be used for training/evaluation labels only."
        else:
            role = "feature"
            leakage_status = "pre_match_feature"
            notes = "Computed from prior matches only or provided as pre-match context."
        non_null = int(snapshots[col].notna().sum())
        rows.append({
            "column": col,
            "role": role,
            "non_null_rows": non_null,
            "coverage_rate": float(non_null / len(snapshots)) if len(snapshots) else 0.0,
            "leakage_status": leakage_status,
            "notes": notes,
        })
    return pd.DataFrame(rows)


def build_model_ready_match_snapshots(
    matches: pd.DataFrame,
    *,
    dataset_name: str = "model_ready_match_snapshots",
) -> ModelReadySnapshotsOutputs:
    """Build leakage-safe match-level snapshots for hybrid Big-5 modelling.

    The output has one row per match. All feature columns are computed from data
    available before kickoff. Targets are kept in the same file for validation
    convenience, but the feature contract marks them as post-match labels so
    downstream models can exclude them.
    """
    required = {"match_id", "date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = sorted(required - set(matches.columns))
    if missing:
        summary = {
            "version": MODEL_READY_SNAPSHOTS_VERSION,
            "dataset_name": dataset_name,
            "status": "blocked",
            "missing_required_columns": missing,
            "input_rows": int(len(matches)),
            "output_rows": 0,
            "principle": "cannot_build_model_ready_snapshots_without_required_match_columns",
        }
        empty = pd.DataFrame()
        return ModelReadySnapshotsOutputs(empty, _feature_contract(empty), summary)

    df = matches.copy()
    df["match_id"] = df["match_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["competition", "season", "stage", "team_scope"]:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)
    for col in ["venue_country", "venue_city", "home_team_country", "away_team_country", "home_team_city", "away_team_city"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    if "neutral" not in df.columns:
        df["neutral"] = 0
    df["neutral"] = pd.to_numeric(df["neutral"], errors="coerce").fillna(0).astype(int)

    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

    identity = df[[c for c in IDENTITY_COLUMNS if c in df.columns]].copy()
    league_context = _build_league_context(df)
    calendar_context = _build_calendar_context(df)
    elo_pre = _build_elo_pre(df)
    external_strength = _build_external_strength_context(df)
    team_features = _wide_team_features(df)
    targets = _build_targets(df)

    snapshots = identity.merge(league_context, on="match_id", how="left", validate="one_to_one")
    snapshots = snapshots.merge(calendar_context, on="match_id", how="left", validate="one_to_one")
    snapshots = snapshots.merge(elo_pre, on="match_id", how="left", validate="one_to_one")
    snapshots = snapshots.merge(external_strength, on="match_id", how="left", validate="one_to_one")
    snapshots = snapshots.merge(team_features, on="match_id", how="left", validate="one_to_one")
    snapshots = snapshots.merge(targets, on="match_id", how="left", validate="one_to_one")
    snapshots = _add_derived_ratio_features(snapshots)
    snapshots = snapshots.sort_values(["date", "match_id"]).reset_index(drop=True)

    contract = _feature_contract(snapshots)
    feature_cols = contract.loc[contract["role"] == "feature", "column"].tolist()
    target_cols = contract.loc[contract["role"] == "target", "column"].tolist()
    xg_features = [c for c in feature_cols if "_xg_" in c or c.endswith("xg_diff_last5")]
    missing_xg = len(xg_features) == 0 or snapshots[xg_features].notna().sum().sum() == 0

    summary = {
        "version": MODEL_READY_SNAPSHOTS_VERSION,
        "extension_version": "v0.50.1_maximum_useful_data_ingestion_snapshots",
        "dataset_name": dataset_name,
        "status": "ok",
        "input_rows": int(len(matches)),
        "output_rows": int(len(snapshots)),
        "date_min": str(snapshots["date"].min().date()) if len(snapshots) and snapshots["date"].notna().any() else None,
        "date_max": str(snapshots["date"].max().date()) if len(snapshots) and snapshots["date"].notna().any() else None,
        "competitions": sorted(snapshots.get("competition", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())[:50],
        "team_scopes": sorted(snapshots.get("team_scope", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())[:10],
        "feature_columns": int(len(feature_cols)),
        "target_columns": int(len(target_cols)),
        "identity_columns": int((contract["role"] == "identity").sum()),
        "xg_features_available": bool(not missing_xg),
        "clubelo_or_external_elo_supported": True,
        "clubelo_features_available": bool("clubelo_available_pre" in snapshots.columns and snapshots["clubelo_available_pre"].fillna(False).any()),
        "calendar_features_available": bool(all(c in snapshots.columns for c in CALENDAR_CONTEXT_COLUMNS)),
        "internal_elo_features_available": bool("home_elo_pre" in snapshots.columns and snapshots["home_elo_pre"].notna().any()),
        "league_context_features_available": bool(all(c in snapshots.columns for c in LEAGUE_CONTEXT_COLUMNS)),
        "derived_conversion_features_available": bool(any("conversion" in c or "xg_per_shot" in c for c in feature_cols)),
        "hybrid_model_policy": "global_big5_model_with_league_features_and_league_level_calibration",
        "leakage_policy": "feature_columns_are_pre_match_or_prior-only; target_columns_are_post_match_labels",
        "raw_data_changed": False,
        "model_logic_changed": False,
        "recommendations": [
            "Use model_ready_match_snapshots.csv as the next model input contract, not raw provider files.",
            "Train global club models on Big 5 snapshots, but always report metrics by league.",
            "Keep xG optional: use it when provider coverage exists, never require it for baseline operation.",
            "Use club data primarily for player/event evidence; keep national-team results national-first.",
        ],
    }
    return ModelReadySnapshotsOutputs(snapshots, contract, summary)
