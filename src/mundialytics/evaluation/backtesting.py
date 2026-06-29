from __future__ import annotations

import pandas as pd

from mundialytics.betting.value import expected_return


def evaluate_paper_bets(picks: pd.DataFrame, outcomes: pd.DataFrame | None = None, stake: float = 1.0) -> pd.DataFrame:
    """Evaluate paper bets when outcomes are available.

    `outcomes` should contain columns: market_type, match_id, player(optional), line, won(bool).
    """
    df = picks.copy()
    if outcomes is None or outcomes.empty:
        df["stake"] = df["value_flag"].astype(float) * stake
        df["realized_profit"] = None
        return df
    keys = ["match_id", "market_type", "line"]
    if "player" in df.columns and "player" in outcomes.columns:
        keys.append("player")
    df = df.merge(outcomes[keys + ["won"]], on=keys, how="left")
    df["stake"] = df["value_flag"].astype(float) * stake
    df["realized_profit"] = df.apply(
        lambda r: 0 if r["stake"] <= 0 or pd.isna(r.get("won")) else ((r["odds"] - 1) * r["stake"] if r["won"] else -r["stake"]),
        axis=1,
    )
    return df


def paper_summary(paper_df: pd.DataFrame) -> dict:
    bets = paper_df[paper_df["stake"] > 0]
    total_staked = float(bets["stake"].sum()) if not bets.empty else 0.0
    profit = float(pd.to_numeric(bets["realized_profit"], errors="coerce").fillna(0).sum()) if not bets.empty else 0.0
    return {
        "bets": int(len(bets)),
        "total_staked": total_staked,
        "profit": profit,
        "roi": profit / total_staked if total_staked else 0.0,
    }
