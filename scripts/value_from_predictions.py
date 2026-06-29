from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.reports.match_value import build_match_value_picks


def main() -> None:
    p = argparse.ArgumentParser(description="Create 1X2 value picks from fixture predictions and decimal odds.")
    p.add_argument("--predictions", required=True, help="CSV produced by predict_fixtures.py")
    p.add_argument("--odds", required=True, help="CSV with fixture_id/match_id, bookmaker, market_type, selection, odds")
    p.add_argument("--out", default="outputs/match_value_picks.csv")
    p.add_argument("--min-edge", type=float, default=0.03)
    p.add_argument("--min-ev", type=float, default=0.03)
    p.add_argument("--commission", type=float, default=0.0)
    args = p.parse_args()

    pred_path = ROOT / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions)
    odds_path = ROOT / args.odds if not Path(args.odds).is_absolute() else Path(args.odds)
    predictions = pd.read_csv(pred_path)
    odds = pd.read_csv(odds_path)
    picks = build_match_value_picks(
        predictions,
        odds,
        min_edge=args.min_edge,
        min_ev=args.min_ev,
        commission=args.commission,
    )
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    picks.to_csv(out, index=False)
    print(f"Saved match value picks to {out}")
    if picks.empty:
        print("No match-winner odds were available.")
    else:
        print(picks[["date", "home_team", "away_team", "selection", "odds", "model_probability", "implied_probability", "edge", "expected_return", "value_flag"]].to_string(index=False))


if __name__ == "__main__":
    main()
