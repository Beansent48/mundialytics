from __future__ import annotations

import pandas as pd

from mundialytics.identity.normalization import canonical_player_name, canonical_team_name

def add_team_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["team", "home_team", "away_team", "opponent"]:
        if col in out.columns:
            out[f"{col}_canonical"] = out[col].map(canonical_team_name)
    if "team" in out.columns and "team_id" not in out.columns:
        out["team_id"] = out["team"].map(canonical_team_name)
    return out

def add_player_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "player" in out.columns:
        out["player_canonical"] = out["player"].map(canonical_player_name)
    if "player_id" not in out.columns and "player" in out.columns:
        out["player_id"] = out["player"].map(canonical_player_name)
    return out
