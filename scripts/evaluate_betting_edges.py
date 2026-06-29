from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from mundialytics.statistical_core.betting_value import BettingValueEngine


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="match_predictions.csv")
    ap.add_argument("--odds", required=True)
    ap.add_argument("--team-stats", default=None)
    ap.add_argument("--player-events", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    match_predictions = pd.read_csv(args.predictions)
    odds = pd.read_csv(args.odds)
    team_stats = pd.read_csv(args.team_stats) if args.team_stats else pd.DataFrame()
    player_events = pd.read_csv(args.player_events) if args.player_events else pd.DataFrame()
    out = BettingValueEngine().evaluate(odds, match_predictions, team_stats, player_events)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
