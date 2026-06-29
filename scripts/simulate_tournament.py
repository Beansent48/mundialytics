from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from mundialytics.statistical_core.match_model import MatchOutcomeModel
from mundialytics.statistical_core.tournament_simulator import TournamentSimulationConfig, TournamentSimulator


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--match-predictions", default=None)
    ap.add_argument("--historical-events", default=None)
    ap.add_argument("--player-events", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-simulations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    fixtures = pd.read_csv(args.fixtures)
    if args.match_predictions:
        mp = pd.read_csv(args.match_predictions)
    else:
        hist = pd.read_csv(args.historical_events) if args.historical_events else pd.DataFrame()
        mp, _ = MatchOutcomeModel().fit(hist).predict_fixtures(fixtures)
    pe = pd.read_csv(args.player_events) if args.player_events else pd.DataFrame()
    out, details = TournamentSimulator(TournamentSimulationConfig(args.n_simulations, args.seed)).simulate(fixtures, mp, pe)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    details.to_csv(str(Path(args.out).with_name(Path(args.out).stem + "_details.csv")), index=False)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
