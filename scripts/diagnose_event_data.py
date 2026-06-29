from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.event_quality import EventReadinessThresholds, assert_event_data_ready, diagnose_player_event_dataset


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    p = argparse.ArgumentParser(description="Strict diagnostic for player-event data coverage. Fails if props data is missing.")
    p.add_argument("--player-events", required=True)
    p.add_argument("--lineups", default=None)
    p.add_argument("--out", default="outputs/event_data_diagnostic.json")
    p.add_argument("--markets", nargs="+", default=["player_shots", "player_shots_on_target", "player_fouls_committed", "player_fouls_drawn", "player_yellow_card"])
    p.add_argument("--min-matches", type=int, default=50)
    p.add_argument("--min-player-rows", type=int, default=500)
    p.add_argument("--min-total-events-per-market", type=int, default=10)
    p.add_argument("--min-minutes-coverage", type=float, default=0.80)
    p.add_argument("--strict", action="store_true", help="Exit with non-zero code if event data is not ready.")
    args = p.parse_args()

    pe_path = _resolve(args.player_events)
    if pe_path is None or not pe_path.exists():
        raise SystemExit(f"player-events file not found: {pe_path}")
    pe = pd.read_csv(pe_path)
    lu = None
    if args.lineups:
        lu_path = _resolve(args.lineups)
        if lu_path is None or not lu_path.exists():
            raise SystemExit(f"lineups file not found: {lu_path}")
        lu = pd.read_csv(lu_path)
    report = diagnose_player_event_dataset(
        pe,
        lineups=lu,
        required_markets=args.markets,
        thresholds=EventReadinessThresholds(
            min_matches=args.min_matches,
            min_player_rows=args.min_player_rows,
            min_total_events_per_market=args.min_total_events_per_market,
            min_minutes_coverage=args.min_minutes_coverage,
        ),
    )
    out = _resolve(args.out)
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict:
        assert_event_data_ready(report)


if __name__ == "__main__":
    main()
