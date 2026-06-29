#!/usr/bin/env python3
"""Build leakage-safe training features from normalized bookmaker odds.

Input must be a Mundialytics `historical_odds_input.csv` produced by the OddsPapi adapter.
Use snapshots from before kickoff only (`snapshot_policy=pre_kickoff` or `closing` upstream).
The script does not train a model; it creates join-ready features to merge with historical
match rows or line-level model datasets.
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


def _clean_odds(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["bookmaker_odds", "line"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in ["market_key", "side", "scope", "bookmaker", "match_id", "subject_team", "subject_player"]:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].astype("string").fillna("").map(norm_key if col in {"market_key", "side", "scope"} else lambda x: str(x).strip())
    work = work.dropna(subset=["bookmaker_odds"])
    work = work[work["bookmaker_odds"].gt(1.0)]
    work["implied_probability_raw"] = 1.0 / work["bookmaker_odds"]
    return work


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["match_id", "bookmaker", "market_key", "side", "scope", "subject_team", "subject_player"]:
        if col not in work.columns:
            work[col] = ""
    if "line" not in work.columns:
        work["line"] = np.nan
    return work


def build_match_1x2_features(odds: pd.DataFrame) -> pd.DataFrame:
    odds = _ensure_feature_columns(odds)
    sub = odds[odds["market_key"].eq("1x2") & odds["side"].isin(["home", "draw", "away"])].copy()
    if sub.empty:
        return pd.DataFrame()
    # Last duplicate per bookmaker/side/snapshot is enough after upstream snapshot selection.
    idx_cols = ["match_id", "bookmaker"]
    pivot = sub.pivot_table(index=idx_cols, columns="side", values="implied_probability_raw", aggfunc="mean").reset_index()
    for c in ["home", "draw", "away"]:
        if c not in pivot.columns:
            pivot[c] = np.nan
    pivot["odds_1x2_overround"] = pivot[["home", "draw", "away"]].sum(axis=1)
    for c in ["home", "draw", "away"]:
        pivot[f"odds_p_{c}_devig"] = pivot[c] / pivot["odds_1x2_overround"]
    pivot["odds_favourite_side"] = pivot[["odds_p_home_devig", "odds_p_draw_devig", "odds_p_away_devig"]].idxmax(axis=1).str.replace("odds_p_", "").str.replace("_devig", "")
    pivot["odds_home_fair"] = 1.0 / pivot["odds_p_home_devig"]
    pivot["odds_draw_fair"] = 1.0 / pivot["odds_p_draw_devig"]
    pivot["odds_away_fair"] = 1.0 / pivot["odds_p_away_devig"]
    return pivot


def build_line_features(odds: pd.DataFrame) -> pd.DataFrame:
    odds = _ensure_feature_columns(odds)
    sub = odds[~odds["market_key"].eq("1x2")].copy()
    if sub.empty:
        return pd.DataFrame()
    group_cols = ["match_id", "bookmaker", "market_key", "scope", "subject_team", "subject_player", "line"]
    pivot = sub.pivot_table(index=group_cols, columns="side", values="implied_probability_raw", aggfunc="mean").reset_index()
    if "over" in pivot.columns and "under" in pivot.columns:
        pivot["odds_ou_overround"] = pivot["over"] + pivot["under"]
        pivot["odds_p_over_devig"] = pivot["over"] / pivot["odds_ou_overround"]
        pivot["odds_p_under_devig"] = pivot["under"] / pivot["odds_ou_overround"]
        pivot["odds_over_fair"] = 1.0 / pivot["odds_p_over_devig"]
        pivot["odds_under_fair"] = 1.0 / pivot["odds_p_under_devig"]
    if "yes" in pivot.columns and "no" in pivot.columns:
        pivot["odds_yes_no_overround"] = pivot["yes"] + pivot["no"]
        pivot["odds_p_yes_devig"] = pivot["yes"] / pivot["odds_yes_no_overround"]
        pivot["odds_p_no_devig"] = pivot["no"] / pivot["odds_yes_no_overround"]
    return pivot.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build training features from historical odds input.")
    parser.add_argument("--historical-odds", required=True)
    parser.add_argument("--out-dir", default="outputs/odds_training_features_current")
    parser.add_argument("--min-bookmakers-per-match", type=int, default=1)
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    odds = _clean_odds(pd.read_csv(_resolve(args.historical_odds), low_memory=False))
    match_features = build_match_1x2_features(odds)
    line_features = build_line_features(odds)

    if not match_features.empty and args.min_bookmakers_per_match > 1:
        counts = match_features.groupby("match_id")["bookmaker"].nunique().reset_index(name="bookmakers_per_match")
        keep = counts[counts["bookmakers_per_match"].ge(args.min_bookmakers_per_match)]["match_id"]
        match_features = match_features[match_features["match_id"].isin(keep)]

    match_features.to_csv(out_dir / "odds_features_match_1x2.csv", index=False)
    line_features.to_csv(out_dir / "odds_features_market_lines.csv", index=False)

    summary = {
        "version": "v0.43_odds_training_features",
        "input_rows": int(len(odds)),
        "match_1x2_feature_rows": int(len(match_features)),
        "market_line_feature_rows": int(len(line_features)),
        "markets": sorted(odds["market_key"].dropna().unique().tolist()),
        "leakage_policy": "Use only pre-kickoff snapshots. This script assumes upstream historical_odds_input was generated with closing/pre_kickoff, never post-kickoff.",
    }
    (out_dir / "odds_training_features_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
