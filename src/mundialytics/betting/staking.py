from __future__ import annotations


def flat_stake(value_flag: bool, stake: float = 1.0) -> float:
    return float(stake if value_flag else 0.0)


def fractional_kelly(probability: float, odds: float, fraction: float = 0.10, bankroll: float = 100.0) -> float:
    """Conservative fractional Kelly stake for decimal odds."""
    b = odds - 1
    p = probability
    q = 1 - p
    kelly = (b * p - q) / b if b > 0 else 0
    return max(0.0, bankroll * fraction * kelly)
