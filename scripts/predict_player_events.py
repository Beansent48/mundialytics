from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from mundialytics.statistical_core.match_model import MatchOutcomeModel
from mundialytics.statistical_core.team_stats_model import TeamStatsModel
from mundialytics.statistical_core.player_event_model import PlayerEventModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--lineups", default=None)
    ap.add_argument("--squads", default=None)
    ap.add_argument("--historical-events", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fixtures = pd.read_csv(args.fixtures)
    hist = pd.read_csv(args.historical_events) if args.historical_events else pd.DataFrame()
    lineups = pd.read_csv(args.lineups) if args.lineups else pd.DataFrame()
    squads = pd.read_csv(args.squads) if args.squads else pd.DataFrame()
    mp, _ = MatchOutcomeModel().fit(hist).predict_fixtures(fixtures)
    ts = TeamStatsModel().fit(hist).predict_fixtures(fixtures, mp)
    out, warnings = PlayerEventModel().fit(hist).predict(fixtures, lineups, squads, ts)
    if not out.empty:
        out["runtime_warnings"] = ";".join(warnings)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
