from __future__ import annotations

import pandas as pd

from mundialytics.identity.normalization import canonical_team_name


def add_pre_match_rolling_features(team_rows: pd.DataFrame, windows: tuple[int, ...] = (3, 5, 10)) -> pd.DataFrame:
    """Add leakage-safe rolling features per team.

    Every rolling stat is shifted by one match, so the current match result is
    never used to predict itself.
    """
    df = team_rows.sort_values(["team", "date", "match_id"]).copy()
    df["team_match_count_pre"] = df.groupby("team").cumcount()
    base_numeric_cols = [
        "goals_for", "goals_against", "xg_for", "xg_against",
        "shots_for", "shots_against", "sot_for", "sot_against",
        "corners_for", "corners_against", "fouls_for", "fouls_against",
        "yellow_cards_for", "yellow_cards_against",
    ]
    inferred_numeric_cols = [
        c for c in df.columns
        if c.endswith(("_for", "_against")) and c not in {"team_for", "team_against"}
    ]
    numeric_cols = sorted(set(base_numeric_cols + inferred_numeric_cols))
    for col in numeric_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        for w in windows:
            df[f"{col}_last{w}"] = (
                df.groupby("team", group_keys=False)[col]
                .apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )
    df["goal_diff_last5"] = df.get("goals_for_last5", 0) - df.get("goals_against_last5", 0)
    df["xg_diff_last5"] = df.get("xg_for_last5", 0) - df.get("xg_against_last5", 0)
    df["shot_diff_last5"] = df.get("shots_for_last5", 0) - df.get("shots_against_last5", 0)
    return df


def apply_rolling_feature_shrinkage(df: pd.DataFrame, *, prior_matches: float = 10.0) -> pd.DataFrame:
    """Shrink noisy rolling features toward dataset medians for low-sample teams.

    This is a simple empirical-Bayes style stabilizer. A team with very few
    pre-match observations should not get extreme rolling attack/defence inputs
    solely from one or two recent games.
    """
    if prior_matches <= 0 or "team_match_count_pre" not in df.columns:
        return df
    out = df.copy()
    rolling_cols = [
        c for c in out.columns
        if c.endswith(("last3", "last5", "last10")) or c in {"goal_diff_last5", "xg_diff_last5", "shot_diff_last5"}
    ]
    if not rolling_cols:
        return out
    counts = pd.to_numeric(out["team_match_count_pre"], errors="coerce").fillna(0).clip(lower=0)
    confidence = counts / (counts + float(prior_matches))
    for col in rolling_cols:
        values = pd.to_numeric(out[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        global_value = float(values.median())
        out[col] = confidence * values.fillna(global_value) + (1.0 - confidence) * global_value
    return out


def build_goal_training_frame(
    team_rows: pd.DataFrame,
    elo_history: pd.DataFrame | None = None,
    *,
    rolling_shrinkage_prior_matches: float = 10.0,
) -> pd.DataFrame:
    df = add_pre_match_rolling_features(team_rows)
    if elo_history is not None and not elo_history.empty:
        h = elo_history[["match_id", "home_team", "away_team", "home_elo_pre", "away_elo_pre"]]
        df = df.merge(h, on="match_id", how="left")
        df["team_elo"] = df.apply(lambda r: r["home_elo_pre"] if r["team"] == r["home_team"] else r["away_elo_pre"], axis=1)
        df["opponent_elo"] = df.apply(lambda r: r["away_elo_pre"] if r["team"] == r["home_team"] else r["home_elo_pre"], axis=1)
        df["elo_diff"] = df["team_elo"] - df["opponent_elo"]
    else:
        df["team_elo"] = 1500.0
        df["opponent_elo"] = 1500.0
        df["elo_diff"] = 0.0

    # Optional external Elo / ClubElo columns can be supplied by canonical data
    # adapters. Internal Elo remains the default and is always available.
    for col in ["external_team_elo", "external_opponent_elo"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["external_elo_diff"] = (
        pd.to_numeric(df["external_team_elo"], errors="coerce")
        - pd.to_numeric(df["external_opponent_elo"], errors="coerce")
    )

    df["neutral"] = df["neutral"].fillna(1).astype(int)
    df["is_home_non_neutral"] = ((df["is_home"] == 1) & (df["neutral"] == 0)).astype(int)
    return apply_rolling_feature_shrinkage(df, prior_matches=rolling_shrinkage_prior_matches)


def fixture_feature_row(home_team: str, away_team: str, match_context: dict, historical_team_features: pd.DataFrame) -> pd.DataFrame:
    home_team = canonical_team_name(home_team)
    away_team = canonical_team_name(away_team)
    rows = []
    for team, opponent, is_home in [(home_team, away_team, 1), (away_team, home_team, 0)]:
        hist = historical_team_features[historical_team_features["team"] == team].sort_values(["date", "match_id"])
        last = hist.iloc[-1].to_dict() if len(hist) else {}
        row = {
            "team": team,
            "opponent": opponent,
            "is_home": is_home,
            "neutral": match_context.get("neutral", 1),
            "competition": match_context.get("competition", "unknown"),
            "stage": match_context.get("stage", "unknown"),
            "team_elo": match_context.get(f"{team}_elo", match_context.get("home_elo" if is_home else "away_elo", 1500)),
            "opponent_elo": match_context.get(f"{opponent}_elo", match_context.get("away_elo" if is_home else "home_elo", 1500)),
            "team_match_count_pre": last.get("team_match_count_pre", 0),
        }
        row["elo_diff"] = row["team_elo"] - row["opponent_elo"]
        external_team_key = "home_external_elo" if is_home else "away_external_elo"
        external_opp_key = "away_external_elo" if is_home else "home_external_elo"
        row["external_team_elo"] = match_context.get(external_team_key, last.get("external_team_elo", pd.NA))
        row["external_opponent_elo"] = match_context.get(external_opp_key, last.get("external_opponent_elo", pd.NA))
        try:
            row["external_elo_diff"] = float(row["external_team_elo"]) - float(row["external_opponent_elo"])
        except (TypeError, ValueError):
            row["external_elo_diff"] = pd.NA
        row["is_home_non_neutral"] = int(is_home == 1 and row["neutral"] == 0)
        # The fitted sklearn ColumnTransformer expects the same feature columns at
        # prediction time. For teams not present in the training history, fill rolling
        # features with training medians instead of omitting the columns. This is
        # important when predicting future slates that include newly promoted clubs
        # or less common national teams.
        rolling_cols = [
            c for c in historical_team_features.columns
            if c.endswith(("last3", "last5", "last10")) or c in ["goal_diff_last5", "xg_diff_last5", "shot_diff_last5"]
        ]
        medians = historical_team_features[rolling_cols].median(numeric_only=True).to_dict() if rolling_cols else {}
        for key in rolling_cols:
            row[key] = last.get(key, medians.get(key, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)
