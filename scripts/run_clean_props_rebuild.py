from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(map(str, cmd)))
    env = os.environ.copy()
    # Keep sklearn/numpy subprocesses deterministic and prevent BLAS thread hangs
    # observed in CI-like Windows/Linux shells during repeated calibration runs.
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True, env=env)


def _add_name_args(cmd: list[str], flag: str, values: list[str] | None) -> list[str]:
    if values:
        cmd.append(flag)
        cmd.extend(values)
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Strict clean rebuild for player-prop validation/calibration. "
            "It filters bad event rows, audits data, validates props, audits predictions, "
            "calibrates, and runs a real temporal anti-overfitting check."
        )
    )
    p.add_argument("--player-events", required=True)
    p.add_argument("--lineups", required=True)
    p.add_argument("--out-dir", default="outputs/player_props_statsbomb_clean")
    p.add_argument("--include-competitions", nargs="*", default=None)
    p.add_argument("--exclude-competitions", nargs="*", default=["StatsBomb Open Data"])
    p.add_argument("--expected-domain", choices=["club", "national", "mixed"], default="mixed")
    p.add_argument("--min-train-matches", type=int, default=100)
    p.add_argument("--test-matches", type=int, default=300)
    p.add_argument("--min-calibration-market-rows", type=int, default=500)
    p.add_argument("--calibration-fraction", type=float, default=0.5)
    p.add_argument("--max-date-null-rate", type=float, default=0.01)
    p.add_argument("--feature-player-events", default=None, help="Optional broader historical player events for cross-context features, e.g. club history in national-team props.")
    p.add_argument("--hierarchical-calibration", action="store_true", default=True, help="Run v0.17 adaptive hierarchical calibration diagnostics.")
    p.add_argument("--min-hierarchical-group-rows", type=int, default=200)
    p.add_argument("--hierarchical-selection-mode", choices=["adaptive", "narrowest"], default="adaptive")
    args = p.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_events = out_dir / "statsbomb_player_events_clean.csv"
    clean_lineups = out_dir / "statsbomb_lineups_clean.csv"
    filter_report = out_dir / "filter_report.json"
    event_audit = out_dir / "audit_clean_events.json"
    validation_dir = out_dir / "validation"
    predictions = validation_dir / "player_props_backtest_predictions.csv"
    prediction_audit = out_dir / "audit_predictions.json"
    calibration_dir = out_dir / "calibration"
    temporal_dir = out_dir / "calibration_temporal_check"
    policy_path = out_dir / "player_props_policy.json"

    filter_cmd = [
        sys.executable, "scripts/filter_player_events_for_props.py",
        "--player-events", args.player_events,
        "--lineups", args.lineups,
        "--out-player-events", str(clean_events),
        "--out-lineups", str(clean_lineups),
        "--report", str(filter_report),
        "--require-valid-date",
    ]
    _add_name_args(filter_cmd, "--exclude-competitions", args.exclude_competitions)
    _add_name_args(filter_cmd, "--include-competitions", args.include_competitions)
    _run(filter_cmd)

    audit_events_cmd = [
        sys.executable, "scripts/audit_props_pipeline.py",
        "--player-events", str(clean_events),
        "--lineups", str(clean_lineups),
        "--out", str(event_audit),
        "--require-valid-date",
        "--max-date-null-rate", str(args.max_date_null_rate),
        "--expected-domain", args.expected_domain,
        "--strict",
    ]
    _add_name_args(audit_events_cmd, "--exclude-competitions", args.exclude_competitions)
    _add_name_args(audit_events_cmd, "--include-competitions", args.include_competitions)
    _run(audit_events_cmd)

    validate_cmd = [
        sys.executable, "scripts/validate_player_props.py",
        "--player-events", str(clean_events),
        "--lineups", str(clean_lineups),
        "--out-dir", str(validation_dir),
        "--min-train-matches", str(args.min_train_matches),
        "--test-matches", str(args.test_matches),
        "--require-valid-date",
        "--max-prediction-date-null-rate", str(args.max_date_null_rate),
    ]
    if args.feature_player_events:
        validate_cmd.extend(["--feature-player-events", args.feature_player_events])
    _run(validate_cmd)

    audit_pred_cmd = [
        sys.executable, "scripts/audit_props_pipeline.py",
        "--predictions", str(predictions),
        "--out", str(prediction_audit),
        "--require-valid-date",
        "--max-date-null-rate", str(args.max_date_null_rate),
        "--expected-domain", args.expected_domain,
        "--strict",
    ]
    _add_name_args(audit_pred_cmd, "--exclude-competitions", args.exclude_competitions)
    _add_name_args(audit_pred_cmd, "--include-competitions", args.include_competitions)
    _run(audit_pred_cmd)

    calibrate_cmd = [
        sys.executable, "scripts/calibrate_player_props.py",
        "--predictions", str(predictions),
        "--out-dir", str(calibration_dir),
        "--calibration-fraction", str(args.calibration_fraction),
        "--min-market-rows", str(args.min_calibration_market_rows),
        "--require-valid-date",
        "--max-date-null-rate", str(args.max_date_null_rate),
    ]
    if args.hierarchical_calibration:
        calibrate_cmd.extend(["--hierarchical", "--min-hierarchical-group-rows", str(args.min_hierarchical_group_rows), "--hierarchical-selection-mode", args.hierarchical_selection_mode])
    _add_name_args(calibrate_cmd, "--exclude-competitions", args.exclude_competitions)
    _add_name_args(calibrate_cmd, "--include-competitions", args.include_competitions)
    _run(calibrate_cmd)

    temporal_cmd = [
        sys.executable, "scripts/temporal_calibration_check.py",
        "--predictions", str(predictions),
        "--out-dir", str(temporal_dir),
        "--calibration-fraction", str(args.calibration_fraction),
        "--min-market-rows", str(args.min_calibration_market_rows),
        "--require-valid-date",
        "--max-date-null-rate", str(args.max_date_null_rate),
    ]
    if args.hierarchical_calibration:
        temporal_cmd.extend(["--hierarchical", "--min-hierarchical-group-rows", str(args.min_hierarchical_group_rows), "--hierarchical-selection-mode", args.hierarchical_selection_mode])
    _run(temporal_cmd)

    policy_cmd = [
        sys.executable, "scripts/finalize_player_props_policy.py",
        "--temporal-report", str(temporal_dir / "temporal_calibration_report.json"),
        "--hierarchical-report", str(temporal_dir / "hierarchical_temporal_calibration_report.json"),
        "--out", str(policy_path),
    ]
    _run(policy_cmd)

    report = {
        "status": "CLEAN_PROPS_REBUILD_COMPLETE",
        "out_dir": str(out_dir),
        "clean_player_events": str(clean_events),
        "clean_lineups": str(clean_lineups),
        "filter_report": str(filter_report),
        "event_audit": str(event_audit),
        "validation_summary": str(validation_dir / "player_props_backtest_summary.json"),
        "validation_predictions": str(predictions),
        "prediction_audit": str(prediction_audit),
        "calibration_report": str(calibration_dir / "calibration_report.json"),
        "calibration_results": str(calibration_dir / "calibration_search_results.csv"),
        "temporal_report": str(temporal_dir / "temporal_calibration_report.json"),
        "hierarchical_temporal_report": str(temporal_dir / "hierarchical_temporal_calibration_report.json"),
        "player_props_policy": str(policy_path),
        "hierarchical_selection_mode": args.hierarchical_selection_mode,
        "feature_player_events": args.feature_player_events,
    }
    (out_dir / "clean_rebuild_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
