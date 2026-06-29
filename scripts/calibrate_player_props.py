from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.competition_taxonomy import enrich_competition_metadata

from mundialytics.evaluation.prop_calibration import (
    incoherence_checks,
    reliability_table,
    run_market_calibration_search,
)
from mundialytics.evaluation.hierarchical_prop_calibration import run_hierarchical_calibration_search


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    p = argparse.ArgumentParser(description="Search player-prop probability calibration methods and diagnostics.")
    p.add_argument("--predictions", required=True, help="player_props_backtest_predictions.csv from validate_player_props.py/run_player_props_pipeline.py")
    p.add_argument("--out-dir", default="outputs/player_props_calibration")
    p.add_argument("--calibration-fraction", type=float, default=0.5, help="Fraction of backtest matches used to fit calibrators; rest is final evaluation")
    p.add_argument("--min-market-rows", type=int, default=200)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--require-valid-date", action="store_true", help="Fail if prediction dates are missing/unparseable above --max-date-null-rate.")
    p.add_argument("--max-date-null-rate", type=float, default=0.01)
    p.add_argument("--exclude-competitions", nargs="*", default=None, help="Optional competitions that must not be present in predictions.")
    p.add_argument("--include-competitions", nargs="*", default=None, help="Optional allowlist of competitions for predictions.")
    p.add_argument("--hierarchical", action="store_true", help="Also run v0.17 hierarchical/adaptive calibration by competition/domain with safe fallbacks.")
    p.add_argument("--min-hierarchical-group-rows", type=int, default=200)
    p.add_argument("--hierarchical-selection-mode", choices=["adaptive", "narrowest"], default="adaptive")
    args = p.parse_args()

    pred_path = _resolve(args.predictions)
    out_dir = _resolve(args.out_dir)
    assert pred_path is not None and out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = enrich_competition_metadata(pd.read_csv(pred_path), overwrite=True)
    if args.require_valid_date:
        required_meta = ["date", "competition", "team_scope", "team_type", "competition_context", "gender", "player_id_global", "player_context_id", "position", "expected_minutes_source"]
        missing_meta = [c for c in required_meta if c not in pred.columns]
        if missing_meta:
            raise ValueError(f"Predictions missing required audit metadata columns for calibration: {missing_meta}")
        leaky = int(pred["expected_minutes_source"].astype(str).str.contains("LEAKY|observed_test_minutes", case=False, regex=True).sum())
        if leaky:
            raise ValueError(f"Predictions use observed test minutes in expected_minutes_source rows={leaky}; rebuild without leakage before calibration.")

    def _parse_names(values):
        if not values:
            return []
        out = []
        for v in values:
            for part in str(v).split(','):
                part = part.strip()
                if part:
                    out.append(part)
        return out

    preflight = {"rows_before": int(len(pred)), "warnings": [], "filters": []}
    if "date" in pred.columns:
        parsed_dates = pd.to_datetime(pred["date"], errors="coerce")
        date_null_rate = float(parsed_dates.isna().mean()) if len(pred) else 0.0
        preflight["date_null_rate"] = date_null_rate
        if args.require_valid_date and date_null_rate > args.max_date_null_rate:
            raise ValueError(f"Prediction date_null_rate={date_null_rate:.3f} exceeds max={args.max_date_null_rate:.3f}. Rebuild predictions from a clean dated player_events file before calibrating.")
    elif args.require_valid_date:
        raise ValueError("Predictions have no date column. Rebuild predictions before temporal calibration.")

    if "competition" in pred.columns and args.exclude_competitions:
        excluded = {x.casefold() for x in _parse_names(args.exclude_competitions)}
        present = set(pred["competition"].dropna().astype(str).str.casefold().unique())
        bad = sorted(present & excluded)
        if bad:
            raise ValueError(f"Excluded competitions present in predictions: {bad}")
    if "competition" in pred.columns and args.include_competitions:
        included = {x.casefold() for x in _parse_names(args.include_competitions)}
        present = set(pred["competition"].dropna().astype(str).str.casefold().unique())
        outside = sorted(present - included)
        if outside:
            raise ValueError(f"Predictions contain competitions outside include list: {outside[:20]}")

    incoh = incoherence_checks(pred)

    results, calibrated, report = run_market_calibration_search(
        pred,
        calibration_fraction=args.calibration_fraction,
        min_market_rows=args.min_market_rows,
    )

    # JSON-friendly params.
    results_csv = results.copy()
    if "params" in results_csv.columns:
        results_csv["params_json"] = results_csv["params"].apply(lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
        results_csv = results_csv.drop(columns=["params"])

    results_path = out_dir / "calibration_search_results.csv"
    calibrated_path = out_dir / "calibrated_player_prop_predictions.csv"
    report_path = out_dir / "calibration_report.json"
    incoh_path = out_dir / "incoherence_report.json"
    reliability_raw_path = out_dir / "reliability_raw_by_market.csv"
    reliability_cal_path = out_dir / "reliability_calibrated_by_market.csv"
    hierarchical_payload = None
    if args.hierarchical:
        h_results, h_calibrated, h_report = run_hierarchical_calibration_search(
            pred,
            calibration_fraction=args.calibration_fraction,
            min_group_rows=args.min_hierarchical_group_rows,
            min_market_rows=args.min_market_rows,
            selection_mode=args.hierarchical_selection_mode,
        )
        h_results_csv = h_results.copy()
        if "params" in h_results_csv.columns:
            h_results_csv["params_json"] = h_results_csv["params"].apply(lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True) if isinstance(x, dict) else "")
            h_results_csv = h_results_csv.drop(columns=["params"])
        h_results_path = out_dir / "hierarchical_calibration_results.csv"
        h_calibrated_path = out_dir / "hierarchical_calibrated_player_prop_predictions.csv"
        h_report_path = out_dir / "hierarchical_calibration_report.json"
        h_results_csv.to_csv(h_results_path, index=False)
        h_calibrated.to_csv(h_calibrated_path, index=False)
        h_report_path.write_text(json.dumps(h_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        hierarchical_payload = {
            "hierarchical_results_csv": str(h_results_path),
            "hierarchical_calibrated_predictions_csv": str(h_calibrated_path),
            "hierarchical_report_json": str(h_report_path),
            "selection_counts": h_report.get("selection_counts", {}),
            "fallback_reasons": h_report.get("fallback_reasons", {}),
            "markets": h_report.get("markets", {}),
        }

    results_csv.to_csv(results_path, index=False)
    calibrated.to_csv(calibrated_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    incoh_path.write_text(json.dumps(incoh, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    raw_rel_parts = []
    for market, g in pred.groupby("market_type"):
        rel = reliability_table(g, prob_col="probability", actual_col="actual", bins=args.bins)
        rel.insert(0, "market_type", market)
        rel.insert(1, "probability_type", "raw")
        raw_rel_parts.append(rel)
    raw_rel = pd.concat(raw_rel_parts, ignore_index=True) if raw_rel_parts else pd.DataFrame()
    raw_rel.to_csv(reliability_raw_path, index=False)

    cal_rel_parts = []
    if not calibrated.empty:
        for market, g in calibrated.groupby("market_type"):
            rel = reliability_table(g, prob_col="calibrated_probability", actual_col="actual", bins=args.bins)
            rel.insert(0, "market_type", market)
            rel.insert(1, "probability_type", "calibrated")
            rel.insert(2, "method", str(g["calibration_method"].iloc[0]))
            cal_rel_parts.append(rel)
    cal_rel = pd.concat(cal_rel_parts, ignore_index=True) if cal_rel_parts else pd.DataFrame()
    cal_rel.to_csv(reliability_cal_path, index=False)

    # Compact console summary.
    best = []
    if not results.empty:
        for market, g in results.sort_values(["log_loss", "brier"], na_position="last").groupby("market_type"):
            row = g.iloc[0].to_dict()
            row.pop("params", None)
            best.append(row)

    payload = {
        "status": "CALIBRATION_COMPLETE",
        "predictions": str(pred_path),
        "preflight": preflight,
        "outputs": {
            "calibration_search_results_csv": str(results_path),
            "calibrated_predictions_csv": str(calibrated_path),
            "calibration_report_json": str(report_path),
            "incoherence_report_json": str(incoh_path),
            "reliability_raw_csv": str(reliability_raw_path),
            "reliability_calibrated_csv": str(reliability_cal_path),
            **({"hierarchical": hierarchical_payload} if hierarchical_payload else {}),
        },
        "best_by_market": best,
        "warnings": incoh.get("warnings", []),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
