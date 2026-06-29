from __future__ import annotations

from pathlib import Path
import pandas as pd

def canonical_fallback_player_id(player: object, team: object | None = None) -> str:
    return f"{str(team or '').strip().lower()}::{str(player or '').strip().lower()}"

def load_identity_map(path: str | Path | None = None) -> pd.DataFrame:
    if path and Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def attach_identity_map_to_lineups(lineups: pd.DataFrame, identity_map: pd.DataFrame | None = None) -> pd.DataFrame:
    out = lineups.copy()
    if "player_id" not in out.columns and "player" in out.columns:
        out["player_id"] = [canonical_fallback_player_id(p, t) for p, t in zip(out["player"], out.get("team", pd.Series([None]*len(out))), strict=False)]
    return out
