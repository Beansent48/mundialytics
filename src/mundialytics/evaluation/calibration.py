from __future__ import annotations

import pandas as pd

from mundialytics.evaluation.metrics import calibration_table


def print_calibration(df: pd.DataFrame, prob_col: str, outcome_col: str, bins: int = 10) -> pd.DataFrame:
    table = calibration_table(df, prob_col, outcome_col, bins=bins)
    return table
