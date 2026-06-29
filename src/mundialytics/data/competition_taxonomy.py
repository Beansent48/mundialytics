from __future__ import annotations

import pandas as pd

def enrich_competition_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "team_scope" not in out.columns:
        out["team_scope"] = "club"
    if "competition" not in out.columns:
        out["competition"] = "unknown"
    return out
