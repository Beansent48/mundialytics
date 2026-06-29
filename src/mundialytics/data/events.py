from __future__ import annotations

import pandas as pd

def add_basic_event_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["shots", "shots_on_target", "fouls_committed", "fouls_drawn", "yellow_cards", "goals", "assists"]:
        if col not in out.columns:
            out[col] = 0
    return out
