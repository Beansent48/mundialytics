from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.competition_taxonomy import enrich_competition_metadata

from mundialytics.evaluation.prop_calibration import run_market_calibration_search, reliability_table
from mundialytics.evaluation.hierarchical_prop_calibration import run_hierarchical_calibration_search


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def repair_with_events(pred: pd.DataFrame, events_path: Path | None) -> tuple[pd.DataFrame, dict]:
    report = {"used_player_events": False, "date_null_rate_before": None, "date_null_rate_after": None}
    if "date" not in pred.columns:
        pred["date"] = pd.NA
    report["date_null_rate_before"] = float(pd.to_datetime(pred["date"], errors="coerce").isna().mean())
    if events_path is None:
        report["date_null_rate_after"] = report["date_null_rate_before"]
        return pred, report
    events = pd.read_csv(events_path)
    if "match_id" not in events.columns or "match_id" not in pred.columns:
        report["date_null_rate_after"] = report["date_null_rate_before"]
        return pred, report
    pred = pred.copy()
    pred["match_id"] = pred["match_id"].astype(str)
    events["match_id"] = events["match_id"].astype(str)
    meta_cols = [c for c in ["match_id", "date", "competition", "team_scope"] if c in events.columns]
    meta = events[meta_cols].drop_duplicates("match_id", keep="first").rename(columns={c: f"{c}_meta" for c in meta_cols if c != "match_id"})
    pred = pred.merge(meta, on="match_id", how="left")
    for col in ["date", "competition", "team_scope"]:
        mcol = f"{col}_meta"
        if mcol in pred.columns:
            if col not in pred.columns:
                pred[col] = pred[mcol]
            else:
                cur = pred[col]
                pred[col] = cur.where(cur.notna() & (cur.astype(str).str.strip() != ""), pred[mcol])
            pred = pred.drop(columns=[mcol])
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
    pred = enrich_competition_metadata(pred, overwrite=True)
    report["used_player_events"] = True
    report["date_null_rate_after"] = float(pred["date"].isna().mean())
    return pred, report


def main() -> None:
    p = argparse.ArgumentParser(description="Temporal anti-overfitting check for player prop calibration.")
    p.add_argument("--predictions", required=True, help="Raw player_props_backtest_predictions.csv")
    p.add_argument("--player-events", default=None, help="Optional statsbomb_player_events.csv to repair missing dates/competition")
    p.add_argument("--out-dir", default="outputs/player_props_calibration_temporal_check")
    p.add_argument("--calibration-fraction", type=float, default=0.5)
    p.add_argument("--min-market-rows", type=int, default=500)
    p.add_argument("--require-valid-date", action="store_true", help="Fail if repaired predictions still have too many missing dates.")
    p.add_argument("--max-date-null-rate", type=float, default=0.01)
    p.add_argument("--hierarchical", action="store_true", help="Also run v0.16 hierarchical temporal calibration by competition/domain.")
    p.add_argument("--min-hierarchical-group-rows", type=int, default=200)
    p.add_argument("--hierarchical-selection-mode", choices=["adaptive", "narrowest"], default="adaptive")
    args = p.parse_args()

    pred_path = _resolve(args.predictions)
    events_path = _resolve(args.player_events)
    out_dir = _resolve(args.out_dir)
    assert pred_path is not None and out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = enrich_competition_metadata(pd.read_csv(pred_path), overwrite=True)
    pred, repair_report = repair_with_events(pred, events_path)
    if args.require_valid_date:
        required_meta = ["date", "competition", "team_scope", "team_type", "competition_context", "gender", "player_id_global", "player_context_id", "position", "expected_minutes_source"]
        missing_meta = [c for c in required_meta if c not in pred.columns]
        if missing_meta:
            raise ValueError(f"Predictions missing required audit metadata columns for temporal calibration: {missing_meta}")
        leaky = int(pred["expected_minutes_source"].astype(str).str.contains("LEAKY|observed_test_minutes", case=False, regex=True).sum())
        if leaky:
            raise ValueError(f"Predictions use observed test minutes in expected_minutes_source rows={leaky}; rebuild without leakage before temporal calibration.")
    if args.require_valid_date and repair_report.get("date_null_rate_after", 1.0) > args.max_date_null_rate:
        raise ValueError(
            f"date_null_rate_after={repair_report.get('date_null_rate_after'):.3f} exceeds max={args.max_date_null_rate:.3f}. "
            "Do not run temporal calibration on this file; rebuild predictions from clean dated player events."
        )

    repaired_path = out_dir / "predictions_with_metadata.csv"
    pred.to_csv(repaired_path, index=False)

    results, calibrated, report = run_market_calibration_search(
        pred,
        calibration_fraction=args.calibration_fraction,
        min_market_rows=args.min_market_rows,
    )
    results_csv = results.copy()
    if "params" in results_csv.columns:
        results_csv["params_json"] = results_csv["params"].apply(lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
        results_csv = results_csv.drop(columns=["params"])
    results_path = out_dir / "temporal_calibration_all_methods.csv"
    calibrated_path = out_dir / "temporal_calibrated_predictions.csv"
    best_path = out_dir / "temporal_calibration_best_by_market.csv"
    rel_path = out_dir / "temporal_reliability_best_by_market.csv"
    results_csv.to_csv(results_path, index=False)
    calibrated.to_csv(calibrated_path, index=False)

    best_rows = []
    if not results.empty:
        for market, g in results.sort_values(["log_loss", "brier"], na_position="last").groupby("market_type"):
            row = g.iloc[0].drop(labels=["params"], errors="ignore").to_dict()
            best_rows.append(row)
    best = pd.DataFrame(best_rows)
    best.to_csv(best_path, index=False)

    rel_parts = []
    if not calibrated.empty:
        for market, g in calibrated.groupby("market_type"):
            rel = reliability_table(g, prob_col="calibrated_probability", actual_col="actual", bins=10)
            rel.insert(0, "market_type", market)
            rel.insert(1, "method", str(g["calibration_method"].iloc[0]))
            rel_parts.append(rel)
    rel = pd.concat(rel_parts, ignore_index=True) if rel_parts else pd.DataFrame()
    rel.to_csv(rel_path, index=False)

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
        h_results_path = out_dir / "hierarchical_temporal_calibration_results.csv"
        h_calibrated_path = out_dir / "hierarchical_temporal_calibrated_predictions.csv"
        h_report_path = out_dir / "hierarchical_temporal_calibration_report.json"
        h_results_csv.to_csv(h_results_path, index=False)
        h_calibrated.to_csv(h_calibrated_path, index=False)
        h_report_path.write_text(json.dumps(h_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        hierarchical_payload = {
            "results": str(h_results_path),
            "calibrated_predictions": str(h_calibrated_path),
            "report": str(h_report_path),
            "selection_counts": h_report.get("selection_counts", {}),
            "fallback_reasons": h_report.get("fallback_reasons", {}),
            "markets": h_report.get("markets", {}),
        }

    payload = {
        "status": "TEMPORAL_CALIBRATION_CHECK_COMPLETE",
        "predictions": str(pred_path),
        "player_events": str(events_path) if events_path else None,
        "repair_report": repair_report,
        "outputs": {
            "predictions_with_metadata": str(repaired_path),
            "all_methods": str(results_path),
            "best_by_market": str(best_path),
            "calibrated_predictions": str(calibrated_path),
            "reliability": str(rel_path),
            **({"hierarchical": hierarchical_payload} if hierarchical_payload else {}),
        },
        "best_by_market": best.to_dict(orient="records"),
        "calibration_report": report,
        "warnings": [],
    }
    if repair_report.get("date_null_rate_after", 1.0) > 0.05:
        payload["warnings"].append("date coverage is still weak; temporal split may be unreliable")
    (out_dir / "temporal_calibration_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
