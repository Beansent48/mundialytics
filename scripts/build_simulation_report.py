from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from mundialytics.statistical_core.reporting import build_daily_html_report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-predictions", required=True)
    ap.add_argument("--team-stats", default=None)
    ap.add_argument("--player-events", default=None)
    ap.add_argument("--betting-edges", default=None)
    ap.add_argument("--tournament-simulation", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build_daily_html_report(
        args.out,
        pd.read_csv(args.match_predictions),
        pd.read_csv(args.team_stats) if args.team_stats else pd.DataFrame(),
        pd.read_csv(args.player_events) if args.player_events else pd.DataFrame(),
        pd.read_csv(args.betting_edges) if args.betting_edges else pd.DataFrame(),
        pd.read_csv(args.tournament_simulation) if args.tournament_simulation else pd.DataFrame(),
        {"status": "report_only"},
    )
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
