from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_PICK_COLUMNS = {
    "created_at", "match_id", "market_type", "selection", "line", "odds",
    "model_probability", "implied_probability", "edge", "expected_return", "stake",
}


def prepare_picks_for_tracking(picks: pd.DataFrame, created_at: str, stake: float = 1.0) -> pd.DataFrame:
    out = picks.copy()
    out["created_at"] = created_at
    if "player" in out.columns and "team" in out.columns:
        out["selection"] = out["player"].where(out["player"].notna() & (out["player"].astype(str).str.strip() != ""), out["team"])
    elif "player" in out.columns:
        out["selection"] = out["player"]
    elif "team" in out.columns:
        out["selection"] = out["team"]
    else:
        out["selection"] = "unknown"
    out["stake"] = out.get("value_flag", True).astype(float) * stake if "value_flag" in out.columns else stake
    out["status"] = "open"
    out["won"] = pd.NA
    out["profit"] = pd.NA
    return out


def append_picks(picks: pd.DataFrame, ledger_path: str | Path, created_at: str, stake: float = 1.0) -> pd.DataFrame:
    ledger_path = Path(ledger_path)
    new = prepare_picks_for_tracking(picks, created_at, stake=stake)
    if ledger_path.exists():
        old = pd.read_csv(ledger_path)
        ledger = pd.concat([old, new], ignore_index=True)
    else:
        ledger = new
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(ledger_path, index=False)
    return ledger


def settle_ledger(ledger: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Settle open picks using outcome rows with match_id, market_type, selection, line, won."""
    keys = ["match_id", "market_type", "selection", "line"]
    merged = ledger.merge(outcomes[keys + ["won"]], on=keys, how="left", suffixes=("", "_settled"))
    mask = merged["won_settled"].notna()
    merged.loc[mask, "won"] = merged.loc[mask, "won_settled"].astype(bool)
    merged.loc[mask, "status"] = "settled"
    merged.loc[mask, "profit"] = merged.loc[mask].apply(
        lambda r: (float(r["odds"]) - 1) * float(r["stake"]) if bool(r["won"]) else -float(r["stake"]), axis=1
    )
    return merged.drop(columns=["won_settled"], errors="ignore")


def ledger_summary(ledger: pd.DataFrame) -> dict:
    status = ledger.get("status", pd.Series(dtype=str))
    settled = ledger[status == "settled"].copy()
    open_ = ledger[status == "open"].copy()
    total_stake = pd.to_numeric(ledger.get("stake", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    settled_stake = pd.to_numeric(settled.get("stake", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    open_stake = pd.to_numeric(open_.get("stake", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    profit = pd.to_numeric(settled.get("profit", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    return {
        "total_picks": int(len(ledger)),
        "settled_picks": int(len(settled)),
        "open_picks": int(len(open_)),
        "total_stake": float(total_stake),
        "settled_stake": float(settled_stake),
        "open_stake": float(open_stake),
        "profit": float(profit),
        "roi_on_settled": float(profit / settled_stake) if settled_stake else 0.0,
    }
