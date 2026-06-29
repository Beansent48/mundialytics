#!/usr/bin/env python3
"""Audit which betting markets can be trained/evaluated from current data.

This script is intentionally conservative: if a target column is missing, the market is
reported as NOT_TRAINABLE instead of being proxied or invented.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd


MATCH_MARKETS = {
    "1x2": {
        "level": "match",
        "required_prediction_columns": ["p_home_win", "p_draw", "p_away_win"],
        "required_actual_columns": ["actual_home_goals", "actual_away_goals"],
    },
    "match_goals_over_under": {
        "level": "match",
        "required_prediction_columns": ["p_over_05", "p_over_15", "p_over_25", "p_over_35"],
        "required_actual_columns": ["actual_home_goals", "actual_away_goals"],
    },
    "btts": {
        "level": "match",
        "required_prediction_columns": ["p_btts"],
        "required_actual_columns": ["actual_home_goals", "actual_away_goals"],
    },
}

EVENT_MARKETS = {
    # Team and match markets typically offered by bookmakers.
    "team_shots": {"level": "team", "target_columns": ["shots", "shots_for"]},
    "match_shots": {"level": "match", "target_columns": ["shots", "shots_for", "total_shots"]},
    "team_shots_on_target": {"level": "team", "target_columns": ["shots_on_target", "shots_on_target_for"]},
    "match_shots_on_target": {"level": "match", "target_columns": ["shots_on_target", "shots_on_target_for", "total_shots_on_target"]},
    "team_fouls": {"level": "team", "target_columns": ["fouls_committed", "fouls_for"]},
    "match_fouls": {"level": "match", "target_columns": ["fouls_committed", "fouls_for", "total_fouls"]},
    "team_yellow_cards": {"level": "team", "target_columns": ["yellow_cards", "yellow_cards_for"]},
    "match_yellow_cards": {"level": "match", "target_columns": ["yellow_cards", "yellow_cards_for", "total_yellow_cards"]},
    "team_red_cards": {"level": "team", "target_columns": ["red_cards", "red_cards_for"]},
    "match_red_cards": {"level": "match", "target_columns": ["red_cards", "red_cards_for", "total_red_cards"]},
    # Player props offered by many books.
    "player_shots": {"level": "player", "target_columns": ["shots"]},
    "player_shots_on_target": {"level": "player", "target_columns": ["shots_on_target"]},
    "player_fouls_committed": {"level": "player", "target_columns": ["fouls_committed"]},
    "player_yellow_card": {"level": "player", "target_columns": ["yellow_cards"]},
    "player_red_card": {"level": "player", "target_columns": ["red_cards"]},
    "player_goals": {"level": "player", "target_columns": ["goals"]},
    "player_assists": {"level": "player", "target_columns": ["assists"]},
    # High-value missing markets: never proxy these silently.
    "team_corners": {"level": "team", "target_columns": ["corners", "corner_kicks", "corners_for"]},
    "match_corners": {"level": "match", "target_columns": ["corners", "corner_kicks", "corners_for", "total_corners"]},
    "team_goalkeeper_saves": {"level": "team", "target_columns": ["saves_for", "goalkeeper_saves", "shots_saved"]},
    "goalkeeper_saves": {"level": "player", "target_columns": ["saves", "goalkeeper_saves", "shots_saved"]},
}

BOOKMAKER_SIDE_RULES = {
    "over_under_markets_must_have_both_sides": [
        "goals", "team_goals", "corners", "team_corners", "shots", "team_shots",
        "shots_on_target", "team_shots_on_target", "yellow_cards", "team_yellow_cards",
        "cards", "team_cards", "fouls", "team_fouls", "goalkeeper_saves",
        "player_shots", "player_shots_on_target", "player_fouls_committed", "player_yellow_card",
    ],
    "binary_markets_must_have_both_sides": ["btts_yes", "btts_no"],
    "result_markets_must_have_all_sides": ["1x2_home", "1x2_draw", "1x2_away"],
}



def _resolve(p: str | None) -> Path | None:
    if not p:
        return None
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def _columns(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    try:
        return set(pd.read_csv(path, nrows=1).columns)
    except Exception:
        return set()



def _present_targets_with_data(path: Path | None, targets: list[str]) -> list[str]:
    """Return target columns that exist and contain at least one non-null value."""
    if path is None or not path.exists():
        return []
    try:
        cols = _columns(path)
        use = [c for c in targets if c in cols]
        if not use:
            return []
        df = pd.read_csv(path, usecols=use)
        return [c for c in use if df[c].notna().any()]
    except Exception:
        # Fallback to schema presence if reading values fails.
        return [c for c in targets if c in _columns(path)]

def _n_rows(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    try:
        # Quick enough for our CSVs and avoids loading all columns into memory.
        return int(sum(1 for _ in open(path, "rb")) - 1)
    except Exception:
        try:
            return int(len(pd.read_csv(path, usecols=[0])))
        except Exception:
            return 0


def _status_from_required(required: list[str], cols: set[str]) -> tuple[str, list[str]]:
    missing = [c for c in required if c not in cols]
    return ("trainable" if not missing else "not_trainable_missing_columns", missing)


def _status_any_target(targets: list[str], cols: set[str]) -> tuple[str, list[str], list[str]]:
    present = [c for c in targets if c in cols]
    missing = [c for c in targets if c not in cols]
    return ("trainable" if present else "not_trainable_missing_target", present, missing)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit trainable/evaluable markets from current Mundialytics data.")
    ap.add_argument("--historical-events", default="outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv")
    ap.add_argument("--match-backtest", default="outputs/evaluation_current/match_backtest_predictions.csv")
    ap.add_argument("--extra-match-stats", default="data/processed/team_match_market_stats.csv", help="Optional team-match market stats CSV with corners/saves/shots/cards targets")
    ap.add_argument("--goalkeeper-match-stats", default="data/processed/goalkeeper_match_stats.csv", help="Optional player-level goalkeeper saves CSV")
    ap.add_argument("--out-dir", default="outputs/market_coverage_audit_current")
    args = ap.parse_args(argv)

    events_path = _resolve(args.historical_events)
    match_path = _resolve(args.match_backtest)
    extra_stats_path = _resolve(args.extra_match_stats)
    goalkeeper_stats_path = _resolve(args.goalkeeper_match_stats)
    out_dir = _resolve(args.out_dir) or ROOT / "outputs/market_coverage_audit_current"
    out_dir.mkdir(parents=True, exist_ok=True)

    event_cols = _columns(events_path)
    extra_cols = _columns(extra_stats_path)
    goalkeeper_cols = _columns(goalkeeper_stats_path)
    # Event markets can be trained/evaluated from player-event rows, team-match market stats, or player goalkeeper stats.
    event_cols_combined = set(event_cols) | set(extra_cols) | set(goalkeeper_cols)
    match_cols = _columns(match_path)
    rows = []

    for market, spec in MATCH_MARKETS.items():
        required = spec["required_prediction_columns"] + spec["required_actual_columns"]
        status, missing = _status_from_required(required, match_cols)
        rows.append({
            "market": market,
            "level": spec["level"],
            "data_source": str(match_path) if match_path else "",
            "status": status,
            "present_targets": ";".join([c for c in required if c in match_cols]),
            "missing_columns": ";".join(missing),
            "rows_available": _n_rows(match_path),
            "decision": "evaluate_model" if status == "trainable" else "do_not_use_until_data_exists",
        })

    for market, spec in EVENT_MARKETS.items():
        targets = spec["target_columns"]
        present_events = _present_targets_with_data(events_path, targets)
        present_extra = _present_targets_with_data(extra_stats_path, targets)
        present_goalkeeper = _present_targets_with_data(goalkeeper_stats_path, targets)
        present = list(dict.fromkeys(present_events + present_extra + present_goalkeeper))
        missing = [c for c in targets if c not in present]
        status = "trainable" if present else "not_trainable_missing_target"
        source_path = goalkeeper_stats_path if present_goalkeeper else (extra_stats_path if present_extra else events_path)
        rows.append({
            "market": market,
            "level": spec["level"],
            "data_source": str(source_path) if source_path else "",
            "status": status,
            "present_targets": ";".join(present),
            "missing_columns": ";".join(missing),
            "rows_available": _n_rows(source_path),
            "decision": "train_or_evaluate_model" if status == "trainable" else "do_not_proxy_or_invent",
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "market_coverage_audit.csv", index=False)
    summary = {
        "version": "v0.38_market_coverage_audit",
        "status": "completed",
        "historical_events_path": str(events_path),
        "match_backtest_path": str(match_path),
        "extra_match_stats_path": str(extra_stats_path),
        "goalkeeper_match_stats_path": str(goalkeeper_stats_path),
        "trainable_markets": df[df["status"].eq("trainable")]["market"].tolist(),
        "not_trainable_markets": df[~df["status"].eq("trainable")][["market", "missing_columns"]].to_dict(orient="records"),
        "bookmaker_side_rules": BOOKMAKER_SIDE_RULES,
        "hard_rules": [
            "Every over/under market must generate/evaluate BOTH over and under sides when a real target exists.",
            "BTTS must generate/evaluate BOTH Yes and No.",
            "Corners must remain not_available until a real corners target exists.",
            "Goalkeeper saves should prefer real provider saves or raw goalkeeper-save events.",
            "A lower-quality goalkeeper saves target may be used only when explicitly derived and flagged as derived_saves_from_sot_minus_goals.",
            "Do not mix real and derived goalkeeper saves without checking saves_data_quality_flag/data_quality_flag.",
            "Do not select a market as a pick candidate only because it is the best aggregate policy; inspect side-level calibration: over vs under, yes vs no, home/draw/away.",
        ],
    }
    (out_dir / "market_coverage_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("MUNDIALYTICS MARKET COVERAGE AUDIT")
    print(f"Output dir: {out_dir}")
    print(df[["market", "level", "status", "present_targets", "missing_columns", "decision"]].to_string(index=False))
    print("\nImportant: missing markets are deliberately kept not_available. Derived saves are allowed only when explicitly requested and flagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
