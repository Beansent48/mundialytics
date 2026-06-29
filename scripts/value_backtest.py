from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.evaluation.value_backtest import run_match_value_backtest


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest 1X2 value picks from prediction results and historical odds.")
    p.add_argument("--predictions", required=True, help="Backtest CSV with match_id/fixture_id and actual_outcome")
    p.add_argument("--odds", required=True, help="Odds CSV with matching match_id/fixture_id")
    p.add_argument("--out", default="outputs/value_backtest.csv")
    p.add_argument("--summary-out", default=None)
    p.add_argument("--min-edge", type=float, default=0.03)
    p.add_argument("--min-ev", type=float, default=0.03)
    p.add_argument("--stake", type=float, default=1.0)
    args = p.parse_args()

    pred_path = ROOT / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions)
    odds_path = ROOT / args.odds if not Path(args.odds).is_absolute() else Path(args.odds)
    preds = pd.read_csv(pred_path)
    odds = pd.read_csv(odds_path)
    settled, summary = run_match_value_backtest(preds, odds, min_edge=args.min_edge, min_ev=args.min_ev, stake=args.stake)

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    settled.to_csv(out, index=False)
    summary_out = ROOT / args.summary_out if args.summary_out and not Path(args.summary_out).is_absolute() else (Path(args.summary_out) if args.summary_out else out.with_suffix(".summary.json"))
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved value backtest to {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
