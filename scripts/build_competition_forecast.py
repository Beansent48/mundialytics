from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mundialytics.statistical_core.scorer_model import CompetitionForecastEngine, ScorerForecastConfig  # noqa: E402
from mundialytics.statistical_core.schemas import read_csv_optional, write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build football.meets.data-style competition forecasts from existing matchday outputs.")
    p.add_argument("--player-events", required=True, help="player_event_predictions.csv")
    p.add_argument("--tournament-simulation", default=None, help="tournament_simulation.csv")
    p.add_argument("--match-predictions", default=None, help="match_predictions.csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-simulations", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    player = pd.read_csv(args.player_events)
    tournament = read_csv_optional(args.tournament_simulation)
    matches = read_csv_optional(args.match_predictions)
    engine = CompetitionForecastEngine(ScorerForecastConfig(n_simulations=args.n_simulations, seed=args.seed))
    top, awards, summary = engine.build_outputs(player, tournament, matches)
    top.to_csv(out / "top_scorer_predictions.csv", index=False)
    awards.to_csv(out / "award_predictions.csv", index=False)
    summary.to_csv(out / "competition_summary.csv", index=False)
    write_json(out / "competition_forecast_audit.json", engine.audit)
    print(f"Competition forecast complete: {out}")
    print(f"Top scorer rows: {len(top)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
