from __future__ import annotations

import pandas as pd


def expected_return(probability: float, decimal_odds: float, commission: float = 0.0) -> float:
    """Expected return per 1 unit stake.

    For exchange-style commission, commission applies to net winnings only.
    """
    net_win = (decimal_odds - 1) * (1 - commission)
    return probability * net_win - (1 - probability)



def shrink_probability(probability: float, sample_size: float | int | None = None, shrink_to: float = 0.5, strength: float = 180.0) -> float:
    """Shrink overconfident probabilities when sample size is small.

    For props, small samples are dangerous. This lightweight empirical-Bayes
    shrinkage prevents demo-level rates from becoming absurd EV picks.
    """
    p = min(max(float(probability), 0.0), 1.0)
    if sample_size is None or pd.isna(sample_size):
        return p
    strength = max(float(strength), 0.0)
    if strength == 0:
        return p
    n = max(float(sample_size), 0.0)
    return float((n * p + strength * shrink_to) / (n + strength))


def evaluate_value(row: pd.Series | dict, min_edge: float = 0.03, min_ev: float = 0.03, commission: float = 0.0) -> dict:
    p_model = shrink_probability(float(row["model_probability"]), row.get("sample_size"), row.get("market_prior", 0.5), row.get("shrink_strength", 50.0))
    odds = float(row["odds"])
    p_impl = float(row.get("implied_probability", 1 / odds))
    edge = p_model - p_impl
    ev = expected_return(p_model, odds, commission=commission)
    return {
        "model_probability_adjusted": p_model,
        "implied_probability": p_impl,
        "edge": edge,
        "expected_return": ev,
        "value_flag": bool(edge >= min_edge and ev >= min_ev),
    }


def add_value_columns(df: pd.DataFrame, min_edge: float = 0.03, min_ev: float = 0.03, commission: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    values = out.apply(lambda r: evaluate_value(r, min_edge=min_edge, min_ev=min_ev, commission=commission), axis=1)
    vdf = pd.DataFrame(list(values))
    for c in ["model_probability_adjusted", "edge", "expected_return", "value_flag"]:
        out[c] = vdf[c].values
    return out
