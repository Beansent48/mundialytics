#!/usr/bin/env python3
"""Build join-ready training features from OddsPapi snapshot odds.

This creates wide features per match/market/line with one column per snapshot label.
Use for market-aware calibration, not as replacement for the pure sports model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from mundialytics.betting.odds_contract import norm_key


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["market_key", "side", "scope", "snapshot_label"]:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].astype("string").fillna("").map(norm_key)
    for col in ["bookmaker_odds", "line"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "snapshot_label" not in work.columns or work["snapshot_label"].eq("").all():
        work["snapshot_label"] = "single_snapshot"
    work = work.dropna(subset=["bookmaker_odds"])
    work = work[work["bookmaker_odds"].gt(1.0)].copy()
    work["implied_probability_raw"] = 1.0 / work["bookmaker_odds"]
    return work


def build_snapshot_feature_table(odds: pd.DataFrame) -> pd.DataFrame:
    if odds.empty:
        return pd.DataFrame()
    for col in ["match_id", "bookmaker", "market_key", "scope", "subject_team", "subject_player", "line", "side", "snapshot_label"]:
        if col not in odds.columns:
            odds[col] = ""
    group_cols = ["match_id", "bookmaker", "market_key", "scope", "subject_team", "subject_player", "line", "side"]
    base = odds[group_cols].drop_duplicates().copy()
    odds_pivot = odds.pivot_table(index=group_cols, columns="snapshot_label", values="bookmaker_odds", aggfunc="last").reset_index()
    imp_pivot = odds.pivot_table(index=group_cols, columns="snapshot_label", values="implied_probability_raw", aggfunc="last").reset_index()
    # Flatten columns.
    odds_pivot.columns = [str(c) if c in group_cols else f"odds_{c}" for c in odds_pivot.columns]
    imp_pivot.columns = [str(c) if c in group_cols else f"implied_prob_{c}" for c in imp_pivot.columns]
    out = odds_pivot.merge(imp_pivot, on=group_cols, how="outer")
    if "odds_t1h" in out.columns and "odds_closing" in out.columns:
        out["odds_move_t1h_to_closing"] = out["odds_closing"] - out["odds_t1h"]
        out["odds_move_pct_t1h_to_closing"] = (out["odds_closing"] / out["odds_t1h"]) - 1.0
    if "implied_prob_t1h" in out.columns and "implied_prob_closing" in out.columns:
        out["implied_prob_move_t1h_to_closing"] = out["implied_prob_closing"] - out["implied_prob_t1h"]
    return out.reset_index(drop=True)


def build_1x2_devig_features(odds: pd.DataFrame) -> pd.DataFrame:
    sub = odds[odds["market_key"].eq("1x2") & odds["side"].isin(["home", "draw", "away"])].copy()
    if sub.empty:
        return pd.DataFrame()
    idx = ["match_id", "bookmaker", "snapshot_label"]
    piv = sub.pivot_table(index=idx, columns="side", values="implied_probability_raw", aggfunc="last").reset_index()
    for c in ["home", "draw", "away"]:
        if c not in piv.columns:
            piv[c] = np.nan
    piv["odds_1x2_overround"] = piv[["home", "draw", "away"]].sum(axis=1)
    for c in ["home", "draw", "away"]:
        piv[f"odds_p_{c}_devig"] = piv[c] / piv["odds_1x2_overround"]
    wide = piv.pivot_table(index=["match_id", "bookmaker"], columns="snapshot_label", values=["odds_p_home_devig", "odds_p_draw_devig", "odds_p_away_devig", "odds_1x2_overround"], aggfunc="last").reset_index()
    wide.columns = ["_".join([str(x) for x in c if str(x) != ""]).strip("_") if isinstance(c, tuple) else str(c) for c in wide.columns]
    return wide


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build training features from OddsPapi snapshot odds.")
    parser.add_argument("--snapshot-odds", required=True)
    parser.add_argument("--out-dir", default="outputs/odds_training_features_from_snapshots_current")
    args = parser.parse_args(argv)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    odds = _clean(pd.read_csv(_resolve(args.snapshot_odds), low_memory=False))
    line_features = build_snapshot_feature_table(odds)
    match_1x2 = build_1x2_devig_features(odds)
    line_features.to_csv(out_dir / "odds_features_market_lines_by_snapshot.csv", index=False)
    match_1x2.to_csv(out_dir / "odds_features_match_1x2_by_snapshot.csv", index=False)
    summary = {
        "version": "v0.46_odds_training_features_from_snapshots",
        "input_rows": int(len(odds)),
        "market_line_feature_rows": int(len(line_features)),
        "match_1x2_feature_rows": int(len(match_1x2)),
        "snapshot_labels": sorted(odds["snapshot_label"].dropna().unique().tolist()),
        "markets": sorted(odds["market_key"].dropna().unique().tolist()),
        "leakage_policy": "Features come from pre-match snapshots selected upstream. Keep training split temporal.",
    }
    (out_dir / "odds_training_features_from_snapshots_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
