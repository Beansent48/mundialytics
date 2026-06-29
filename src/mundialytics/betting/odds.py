from __future__ import annotations

import pandas as pd


def decimal_to_implied(odds: float) -> float:
    if odds <= 1:
        raise ValueError("Decimal odds must be > 1")
    return 1.0 / float(odds)


def add_implied_probabilities(odds_df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    df = odds_df.copy()
    df["implied_probability_raw"] = df["odds"].apply(decimal_to_implied)
    if group_cols:
        overround = df.groupby(group_cols)["implied_probability_raw"].transform("sum")
        df["book_overround"] = overround
        df["implied_probability"] = df["implied_probability_raw"] / overround
    else:
        df["book_overround"] = None
        df["implied_probability"] = df["implied_probability_raw"]
    return df


def fair_odds(probability: float, commission: float = 0.0) -> float:
    p = min(max(float(probability), 1e-6), 0.999999)
    odds = 1.0 / p
    if commission > 0:
        # Approximation: require higher gross odds to offset exchange commission on net winnings.
        odds = 1 + (odds - 1) / (1 - commission)
    return odds
