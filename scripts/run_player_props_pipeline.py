from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="One-command player props pipeline using real StatsBomb Open Data event files.")
    p.add_argument("--statsbomb-data", default="data/raw/statsbomb/open-data/data", help="StatsBomb data directory containing events/, matches/, competitions.json")
    p.add_argument("--team-scope", default="club", choices=["club", "national"])
    p.add_argument("--limit", type=int, default=None, help="Optional cap for quick tests. Do not use for serious validation.")
    p.add_argument("--out-dir", default="outputs/player_props_statsbomb")
    p.add_argument("--min-matches", type=int, default=50)
    p.add_argument("--min-player-rows", type=int, default=500)
    p.add_argument("--min-train-matches", type=int, default=50)
    p.add_argument("--test-matches", type=int, default=300)
    p.add_argument("--min-total-events-per-market", type=int, default=10)
    p.add_argument("--min-minutes-coverage", type=float, default=0.80)
    p.add_argument("--skip-backtest", action="store_true", help="Only build/diagnose event data; useful for tiny smoke tests.")
    p.add_argument("--run-calibration", action="store_true", help="After backtest, search calibration methods and output calibrated prop probabilities.")
    p.add_argument("--calibration-fraction", type=float, default=0.5)
    p.add_argument("--min-calibration-market-rows", type=int, default=200)
    args = p.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pe = out_dir / "statsbomb_player_events.csv"
    te = out_dir / "statsbomb_team_events.csv"
    lu = out_dir / "statsbomb_lineups.csv"
    ts = out_dir / "statsbomb_tactical_shifts.csv"
    diag = out_dir / "event_data_diagnostic.json"

    build_cmd = [
        sys.executable, "scripts/build_event_datasets.py", "statsbomb",
        "--input", args.statsbomb_data,
        "--team-scope", args.team_scope,
        "--player-events-out", str(pe),
        "--team-events-out", str(te),
        "--lineups-out", str(lu),
        "--tactical-out", str(ts),
        "--diagnostic-out", str(diag),
    ]
    if args.limit:
        build_cmd += ["--limit", str(args.limit)]
    _run(build_cmd)

    _run([
        sys.executable, "scripts/diagnose_event_data.py",
        "--player-events", str(pe),
        "--lineups", str(lu),
        "--out", str(diag),
        "--min-matches", str(args.min_matches),
        "--min-player-rows", str(args.min_player_rows),
        "--min-total-events-per-market", str(args.min_total_events_per_market),
        "--min-minutes-coverage", str(args.min_minutes_coverage),
        "--strict",
    ])

    validation_dir = out_dir / "validation"
    summary_path = validation_dir / "player_props_backtest_summary.json"
    predictions_path = validation_dir / "player_props_backtest_predictions.csv"
    calibration_dir = out_dir / "calibration"
    if not args.skip_backtest:
        _run([
            sys.executable, "scripts/validate_player_props.py",
            "--player-events", str(pe),
            "--lineups", str(lu),
            "--out-dir", str(validation_dir),
            "--min-train-matches", str(args.min_train_matches),
            "--test-matches", str(args.test_matches),
        ])
        if args.run_calibration:
            _run([
                sys.executable, "scripts/calibrate_player_props.py",
                "--predictions", str(predictions_path),
                "--out-dir", str(calibration_dir),
                "--calibration-fraction", str(args.calibration_fraction),
                "--min-market-rows", str(args.min_calibration_market_rows),
            ])

    report = {
        "status": "PLAYER_PROPS_PIPELINE_COMPLETED" if not args.skip_backtest else "PLAYER_PROPS_EVENT_DATA_BUILT_AND_DIAGNOSED",
        "player_events": str(pe),
        "team_events": str(te),
        "lineups": str(lu),
        "tactical_shifts": str(ts),
        "diagnostic": str(diag),
        "validation_summary": str(summary_path) if summary_path.exists() else None,
        "validation_predictions": str(predictions_path) if predictions_path.exists() else None,
        "calibration_report": str(calibration_dir / "calibration_report.json") if (calibration_dir / "calibration_report.json").exists() else None,
        "calibration_results": str(calibration_dir / "calibration_search_results.csv") if (calibration_dir / "calibration_search_results.csv").exists() else None,
    }
    (out_dir / "player_props_pipeline_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
