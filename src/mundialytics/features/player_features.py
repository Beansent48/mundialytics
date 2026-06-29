from __future__ import annotations

import pandas as pd

EVENT_COLUMNS = [
    "shots", "shots_on_target", "fouls_committed", "fouls_drawn", "yellow_cards", "goals", "assists"
]


def _player_key_columns(df: pd.DataFrame) -> list[str]:
    """Return stable identity columns for player aggregation.

    Prefer player_id_global when available so future lineups can match current
    players against historical training rows without relying only on display
    names. Keep player as a fallback for older sample data.
    """
    if "player_id_global" in df.columns:
        return ["player_id_global"]
    return ["player"]


def add_player_rate_features(events: pd.DataFrame) -> pd.DataFrame:
    df = events.sort_values([c for c in ["player_id_global", "player", "date", "match_id"] if c in events.columns]).copy()
    keys = _player_key_columns(df)
    for col in EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[f"{col}_per90"] = df[col] / df["minutes"].clip(lower=1) * 90
        df[f"{col}_last5_per90"] = (
            df.groupby(keys, group_keys=False)[f"{col}_per90"]
            .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
    df["minutes_last5"] = (
        df.groupby(keys, group_keys=False)["minutes"]
        .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    return df


def _minutes_by_team_type(g: pd.DataFrame) -> dict[str, float]:
    if "team_type" not in g.columns:
        return {"club_minutes_sample": 0.0, "national_minutes_sample": 0.0, "unknown_team_type_minutes_sample": float(g["minutes"].sum())}
    tmp = g.copy()
    tmp["team_type"] = tmp["team_type"].fillna("unknown").astype(str)
    sums = tmp.groupby("team_type")["minutes"].sum()
    return {
        "club_minutes_sample": float(sums.get("club", 0.0)),
        "national_minutes_sample": float(sums.get("national_team", 0.0)),
        "unknown_team_type_minutes_sample": float(sums.get("unknown", 0.0)),
    }


def player_baselines(events: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Aggregate historical player baselines without leaking future rows.

    Rows are grouped by player identity by default. The returned table keeps
    metadata needed for operational audits, including minutes split by club vs
    national-team context so current national props can safely use recent club
    evidence while making that cross-context use visible.
    """
    df = events.copy()
    if "minutes" not in df.columns:
        df["minutes"] = 90.0
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).clip(lower=0, upper=130)
    for col in EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    keys = group_cols or _player_key_columns(df)
    missing_keys = [c for c in keys if c not in df.columns]
    if missing_keys:
        raise ValueError(f"player_baselines missing group columns: {missing_keys}")
    rows = []
    for key, g in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key_tuple = (key,)
        else:
            key_tuple = key
        minutes = float(g["minutes"].sum())
        last = g.sort_values([c for c in ["date", "match_id"] if c in g.columns]).iloc[-1]
        row = {
            "player": last.get("player"),
            "team": last.get("team"),
            "position": last.get("position"),
            "minutes_sample": minutes,
            "matches_sample": int(g["match_id"].astype(str).nunique()) if "match_id" in g.columns else int(len(g)),
            "expected_minutes": float(min(max(g["minutes"].tail(5).mean(), 20), 95)) if len(g) else 60.0,
            "start_probability": float(pd.to_numeric(g.get("started", pd.Series([1] * len(g))), errors="coerce").fillna(1).tail(10).mean()) if len(g) else 0.75,
        }
        row.update(_minutes_by_team_type(g))
        for col, value in zip(keys, key_tuple):
            row[col] = value
        for optional in ["player_context_id", "team_scope", "team_type", "competition_context", "gender", "competition"]:
            if optional in g.columns and optional not in row:
                row[optional] = last.get(optional)
        for col in EVENT_COLUMNS:
            row[f"{col}_per90"] = float(g[col].sum() / max(minutes, 1) * 90)
        rows.append(row)
    return pd.DataFrame(rows)
