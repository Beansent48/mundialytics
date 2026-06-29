from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.events import add_basic_event_metrics, merge_player_events_with_lineups
from mundialytics.data.competition_taxonomy import enrich_competition_metadata, competition_domain_report
from mundialytics.evaluation.player_props import PlayerPropBacktestConfig, backtest_player_props


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    p = argparse.ArgumentParser(description="Validate player-event prop markets from processed player_events CSV.")
    p.add_argument("--player-events", required=True, help="Processed player_events CSV from build_event_datasets.py")
    p.add_argument("--lineups", default=None, help="Optional processed lineups CSV to enforce minutes/substitution context")
    p.add_argument("--feature-player-events", default=None, help="Optional broader historical player_events CSV for cross-context features, e.g. club history for national-team props. Rows are cut off before the test period to avoid leakage.")
    p.add_argument("--out-dir", default="outputs/player_props_validation")
    p.add_argument("--min-train-matches", type=int, default=20)
    p.add_argument("--test-matches", type=int, default=200)
    p.add_argument("--markets", nargs="+", default=["player_shots", "player_shots_on_target", "player_fouls_committed", "player_yellow_card"])
    p.add_argument("--line", default="1+")
    p.add_argument("--require-valid-date", action="store_true", help="Drop rows with missing/unparseable date before temporal backtest")
    p.add_argument("--exclude-competitions", nargs="*", default=None, help="Competition names to exclude before backtest")
    p.add_argument("--include-competitions", nargs="*", default=None, help="If supplied, keep only these competition names before backtest")
    p.add_argument("--max-prediction-date-null-rate", type=float, default=0.01)
    p.add_argument("--allow-observed-test-minutes", action="store_true", help="Diagnostic only: use actual test minutes as expected_minutes. This leaks information and is disabled by default.")
    args = p.parse_args()

    out_dir = _resolve(args.out_dir)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    pe = pd.read_csv(_resolve(args.player_events))
    pe = enrich_competition_metadata(pe, overwrite=True)
    if args.lineups:
        lu = pd.read_csv(_resolve(args.lineups))
        lu = enrich_competition_metadata(lu, overwrite=True)
        pe = merge_player_events_with_lineups(pe, lu)
        pe = enrich_competition_metadata(pe, overwrite=True)
    pe = add_basic_event_metrics(pe)

    if "date" in pe.columns:
        pe["date"] = pd.to_datetime(pe["date"], errors="coerce")
        if args.require_valid_date:
            before = len(pe)
            pe = pe[pe["date"].notna()].copy()
            print(f"Filtered valid dates: {before}->{len(pe)}")

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

    if "competition" in pe.columns and args.exclude_competitions:
        excluded = {x.casefold() for x in _parse_names(args.exclude_competitions)}
        before = len(pe)
        pe = pe[~pe["competition"].astype(str).str.casefold().isin(excluded)].copy()
        print(f"Excluded competitions {sorted(excluded)}: {before}->{len(pe)}")
    if "competition" in pe.columns and args.include_competitions:
        included = {x.casefold() for x in _parse_names(args.include_competitions)}
        before = len(pe)
        pe = pe[pe["competition"].astype(str).str.casefold().isin(included)].copy()
        print(f"Included competitions {sorted(included)}: {before}->{len(pe)}")

    feature_events = None
    if args.feature_player_events:
        feature_events = pd.read_csv(_resolve(args.feature_player_events))
        feature_events = enrich_competition_metadata(feature_events, overwrite=True)
        feature_events = add_basic_event_metrics(feature_events)
        if "date" in feature_events.columns:
            feature_events["date"] = pd.to_datetime(feature_events["date"], errors="coerce")
            if args.require_valid_date:
                feature_events = feature_events[feature_events["date"].notna()].copy()

    pred, summary = backtest_player_props(
        pe,
        PlayerPropBacktestConfig(
            min_train_matches=args.min_train_matches,
            test_matches=args.test_matches,
            markets=tuple(args.markets),
            line=args.line,
            use_observed_test_minutes=bool(args.allow_observed_test_minutes),
        ),
        feature_events=feature_events,
    )
    # Hard validation of prediction metadata. A temporal/calibration backtest is not trustworthy
    # if match dates disappeared on the way to the predictions CSV.
    prediction_date_null_rate = None
    if "date" in pred.columns and len(pred):
        prediction_date_null_rate = float(pd.to_datetime(pred["date"], errors="coerce").isna().mean())
    meta_cols_required = ["date", "competition", "team_scope", "team_type", "competition_context", "gender", "position", "started", "player_id_global", "player_context_id"]
    missing_meta = [c for c in meta_cols_required if c not in pred.columns]
    if missing_meta:
        raise ValueError(f"Backtest predictions missing required audit metadata columns: {missing_meta}")
    summary["prediction_metadata"] = {
        "date_null_rate": prediction_date_null_rate,
        "date_min": str(pd.to_datetime(pred["date"], errors="coerce").min().date()) if "date" in pred.columns and pd.to_datetime(pred["date"], errors="coerce").notna().any() else None,
        "date_max": str(pd.to_datetime(pred["date"], errors="coerce").max().date()) if "date" in pred.columns and pd.to_datetime(pred["date"], errors="coerce").notna().any() else None,
        "competitions_top": {str(k): int(v) for k, v in pred["competition"].fillna("<NA>").value_counts().head(25).items()},
        "team_scope_counts": {str(k): int(v) for k, v in pred["team_scope"].fillna("<NA>").value_counts().head(25).items()},
        "domain_counts": competition_domain_report(pred),
        "expected_minutes_source_counts": {str(k): int(v) for k, v in pred.get("expected_minutes_source", pd.Series(dtype=object)).fillna("<NA>").value_counts().head(25).items()},
        "cross_context_feature_used_counts": {str(k): int(v) for k, v in pred.get("cross_context_feature_used", pd.Series(dtype=object)).fillna(False).value_counts().head(25).items()},
        "feature_training": summary.get("feature_training", {}),
        "uses_observed_test_minutes": bool(args.allow_observed_test_minutes),
    }
    if args.require_valid_date and prediction_date_null_rate is not None and prediction_date_null_rate > args.max_prediction_date_null_rate:
        raise ValueError(f"Backtest predictions lost dates: date_null_rate={prediction_date_null_rate:.3f}. Rebuild from clean player_events before calibrating.")

    pred_path = out_dir / "player_props_backtest_predictions.csv"
    summary_path = out_dir / "player_props_backtest_summary.json"
    pred.to_csv(pred_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"predictions_csv": str(pred_path), "summary_json": str(summary_path), "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
