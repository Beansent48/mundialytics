#!/usr/bin/env python3
"""Create settled bookmaker-style over/under line signals for event markets.

Input: team_match_market_stats.csv with real targets such as corners_for, saves_for,
shots_for, shots_on_target_for, yellow_cards_for, fouls_for.
Optional input: goalkeeper_match_stats.csv with player-level real goalkeeper saves.
Output: settled_event_line_signals.csv consumable by backtest_pick_policy.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.statistical_core.event_line_backtest import build_settled_event_line_signals, build_goalkeeper_save_line_signals
from mundialytics.betting.pick_policy import standardize_settled_line_signals, evaluate_threshold_performance_by_market


def _resolve(p: str | None) -> Path | None:
    if not p:
        return None
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build settled over/under event line signals from team/player match stats")
    ap.add_argument("--team-match-stats", default="data/processed/team_match_market_stats.csv")
    ap.add_argument("--goalkeeper-match-stats", default="data/processed/goalkeeper_match_stats.csv", help="Optional player-level goalkeeper saves CSV")
    ap.add_argument("--out-dir", default="outputs/event_line_backtest_current")
    ap.add_argument("--min-history", type=int, default=3)
    args = ap.parse_args(argv)

    stats_path = _resolve(args.team_match_stats)
    gk_path = _resolve(args.goalkeeper_match_stats)
    out_dir = _resolve(args.out_dir) or ROOT / "outputs/event_line_backtest_current"
    out_dir.mkdir(parents=True, exist_ok=True)
    if stats_path is None or not stats_path.exists():
        print(f"No existe team-match-stats: {stats_path}")
        return 1
    stats = pd.read_csv(stats_path, low_memory=False)
    signals = build_settled_event_line_signals(stats, min_history=args.min_history)
    gk_signals = pd.DataFrame()
    if gk_path and gk_path.exists():
        gk = pd.read_csv(gk_path, low_memory=False)
        gk_signals = build_goalkeeper_save_line_signals(gk, team_match_stats=stats, min_history=args.min_history)
        if not gk_signals.empty:
            signals = pd.concat([signals, gk_signals], ignore_index=True, sort=False)
    else:
        print(f"Aviso: goalkeeper-match-stats no existe: {gk_path}. Solo se evaluarán saves de equipo si existen en team-match-stats.")
    standardized = standardize_settled_line_signals(signals)
    signals_path = out_dir / "settled_event_line_signals.csv"
    standardized.to_csv(signals_path, index=False)
    threshold = evaluate_threshold_performance_by_market(standardized)
    threshold.to_csv(out_dir / "event_line_threshold_performance.csv", index=False)
    summary = {
        "version": "v0.38_extra_stats_relational_event_line_backtest",
        "team_match_stats": str(stats_path),
        "goalkeeper_match_stats": str(gk_path) if gk_path else "",
        "rows_in": int(len(stats)),
        "gk_signals_rows": int(len(gk_signals)),
        "signals_rows": int(len(standardized)),
        "matches": int(standardized["match_id"].nunique()) if not standardized.empty else 0,
        "markets": standardized["market"].value_counts().to_dict() if not standardized.empty else {},
        "signal_groups": standardized["signal_group"].value_counts().to_dict() if not standardized.empty and "signal_group" in standardized.columns else {},
        "target_quality": standardized["target_quality"].value_counts().to_dict() if not standardized.empty and "target_quality" in standardized.columns else {},
        "output": str(signals_path),
        "model_family": "relational_rolling_market_model_v038",
        "notes": [
            "Corners use corner history plus a modest shot-volume adjustment.",
            "Team goalkeeper saves use save history plus opponent SOT pressure when available.",
            "Player goalkeeper saves use real goalkeeper rows only, from provider player stats or StatsBomb raw events.",
            "Derived saves remain flagged and should be analysed separately from real saves.",
        ],
    }
    (out_dir / "event_line_backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("MUNDIALYTICS EVENT LINE BACKTEST v0.38")
    print(f"Input rows: {len(stats)}")
    print(f"Player GK signals: {len(gk_signals)}")
    print(f"Signals: {len(standardized)} | matches: {summary['matches']}")
    print(f"Output: {signals_path}")
    if not standardized.empty:
        print("Signal groups:")
        print(standardized["signal_group"].value_counts().head(40).to_string())
        if "target_quality" in standardized.columns:
            print("Target quality:")
            print(standardized["target_quality"].value_counts().head(20).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
