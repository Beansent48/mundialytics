from __future__ import annotations

import pandas as pd

def add_local_kickoff_columns(df: pd.DataFrame, timezone: str | None = None) -> pd.DataFrame:
    return df.copy()

def filter_by_matchday_date(df: pd.DataFrame, matchday_date: str | None = None) -> pd.DataFrame:
    if matchday_date is None or "date" not in df.columns:
        return df.copy()
    d = pd.to_datetime(matchday_date).date()
    out = df.copy()
    return out[pd.to_datetime(out["date"], errors="coerce").dt.date == d].copy()
