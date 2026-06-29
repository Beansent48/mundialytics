from __future__ import annotations

import pandas as pd

from mundialytics.reports.match_value import build_match_value_picks

_OUTCOME_TO_SELECTION = {"H": "home", "D": "draw", "A": "away"}


def settle_match_value_picks(picks: pd.DataFrame, prediction_results: pd.DataFrame, *, stake: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Settle 1X2 value picks against backtest actual outcomes.

    Only rows with ``value_flag=True`` are counted as bets. The function expects
    backtest predictions containing match_id or fixture_id plus actual_outcome
    in H/D/A format.
    """
    if picks.empty:
        return picks.copy(), {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": 0.0, "hit_rate": 0.0}
    key = "fixture_id" if "fixture_id" in picks.columns else "match_id"
    if key not in prediction_results.columns:
        raise ValueError(f"Prediction results missing key column {key!r}.")
    if "actual_outcome" not in prediction_results.columns:
        raise ValueError("Prediction results must contain actual_outcome for settlement.")
    actual = prediction_results[[key, "actual_outcome"]].drop_duplicates(subset=[key])
    out = picks.merge(actual, on=key, how="left", validate="many_to_one")
    out["bet"] = out["value_flag"].astype(bool)
    out["stake"] = out["bet"].astype(float) * float(stake)
    out["won"] = out.apply(lambda r: bool(r["bet"]) and _OUTCOME_TO_SELECTION.get(str(r.get("actual_outcome"))) == r.get("selection_type"), axis=1)
    out["profit"] = out.apply(lambda r: (float(r["odds"]) - 1) * r["stake"] if r["won"] else (-r["stake"] if r["bet"] else 0.0), axis=1)
    bets = out[out["bet"]].copy()
    total_stake = float(bets["stake"].sum())
    profit = float(bets["profit"].sum())
    summary = {
        "bets": int(len(bets)),
        "stake": total_stake,
        "profit": profit,
        "roi": float(profit / total_stake) if total_stake else 0.0,
        "hit_rate": float(bets["won"].mean()) if len(bets) else 0.0,
        "avg_edge": float(bets["edge"].mean()) if len(bets) else 0.0,
        "avg_expected_return": float(bets["expected_return"].mean()) if len(bets) else 0.0,
    }
    return out, summary


def run_match_value_backtest(prediction_results: pd.DataFrame, odds: pd.DataFrame, *, min_edge: float = 0.03, min_ev: float = 0.03, stake: float = 1.0) -> tuple[pd.DataFrame, dict]:
    picks = build_match_value_picks(prediction_results, odds, min_edge=min_edge, min_ev=min_ev)
    return settle_match_value_picks(picks, prediction_results, stake=stake)
