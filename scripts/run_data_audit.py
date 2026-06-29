
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mundialytics.data_quality import (  # noqa: E402
    DATA_AUDIT_VERSION,
    audit_data_sources,
    write_data_audit_outputs,
)


OPTION_TO_DATASET = {
    "fixtures": "fixtures",
    "actual_results": "actual_results",
    "lineups": "lineups",
    "squads": "squads",
    "player_events": "player_events",
    "odds": "odds",
    "predictions": "predictions",
    "scorelines": "scorelines",
    "dynamic_lines": "dynamic_lines",
    "matchday_summary": "matchday_summary",
    "tournament_simulation": "tournament_simulation",
    "tournament_report": "tournament_report",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an offline Mundialytics data quality audit. No models, APIs or betting actions are executed."
    )
    parser.add_argument("--fixtures", default=None, help="CSV with match fixtures")
    parser.add_argument("--actual-results", default=None, help="CSV with finished match results")
    parser.add_argument("--lineups", default=None, help="CSV with current lineups")
    parser.add_argument("--squads", default=None, help="CSV with current squads")
    parser.add_argument("--player-events", default=None, help="CSV with historical player events")
    parser.add_argument("--odds", default=None, help="Optional CSV with odds snapshots or sample odds")
    parser.add_argument("--predictions", default=None, help="Optional match_predictions.csv")
    parser.add_argument("--scorelines", default=None, help="Optional scoreline_distribution.csv")
    parser.add_argument("--dynamic-lines", default=None, help="Optional dynamic_market_lines.csv")
    parser.add_argument("--matchday-summary", default=None, help="Optional matchday_summary.csv")
    parser.add_argument("--tournament-simulation", default=None, help="Optional tournament_simulation.csv")
    parser.add_argument("--tournament-report", default=None, help="Optional tournament_report.csv")
    parser.add_argument("--out-dir", required=True, help="Directory where audit outputs will be written")
    parser.add_argument("--run-label", default="data_audit", help="Human-readable run label saved in data_audit_summary.json")
    parser.add_argument("--clean-out-dir", action="store_true", help="Delete out-dir before writing outputs")
    return parser


def _read_optional_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None or str(path).strip() == "":
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_dir = Path(args.out_dir)
    if args.clean_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources: dict[str, pd.DataFrame] = {}
    source_paths: dict[str, str] = {}
    for option_name, dataset_name in OPTION_TO_DATASET.items():
        value = getattr(args, option_name)
        if value is not None and str(value).strip():
            sources[dataset_name] = _read_optional_csv(value)
            source_paths[dataset_name] = str(value)

    outputs = audit_data_sources(
        sources=sources,
        source_paths=source_paths,
        run_label=args.run_label,
    )
    written = write_data_audit_outputs(outputs, out_dir)

    print(f"{DATA_AUDIT_VERSION}: status={outputs.summary['status']} outputs={len(written)} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
