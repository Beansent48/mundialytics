from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.inference.safe_props import DEFAULT_MARKETS, predict_props_for_lineups


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    p = argparse.ArgumentParser(description="Generate safe calibrated player props ONLY for supplied current lineups.")
    p.add_argument("--lineups", required=True, help="CSV with current eligible players: match_id,date,team,opponent,player,position,expected_minutes,started")
    p.add_argument("--player-events", required=True, help="Historical statsbomb_player_events.csv used for training rates")
    p.add_argument("--calibration-predictions", default=None, help="Historical player_props_backtest_predictions.csv used to fit probability calibrators")
    p.add_argument("--calibration-results", default=None, help="Optional calibration_search_results.csv to choose best method by market")
    p.add_argument("--out", default="outputs/safe_lineup_props.csv")
    p.add_argument("--markets", nargs="+", default=list(DEFAULT_MARKETS))
    p.add_argument("--line", default="1+")
    p.add_argument("--min-calibration-rows", type=int, default=500)
    p.add_argument("--strict-lineup-contract", action="store_true", help="Require current-lineup schema and valid dates/minutes; recommended for real matchday use.")
    p.add_argument("--disable-hierarchical-calibration", action="store_true")
    p.add_argument("--min-hierarchical-group-rows", type=int, default=200)
    p.add_argument("--calibration-policy", default=None, help="Optional player_props_policy.json from finalize_player_props_policy.py")
    p.add_argument("--identity-map", default=None, help="Optional provider ID ↔ historical player identity map CSV")
    args = p.parse_args()

    lineups_path = _resolve(args.lineups)
    events_path = _resolve(args.player_events)
    cal_pred_path = _resolve(args.calibration_predictions)
    cal_results_path = _resolve(args.calibration_results)
    policy_path = _resolve(args.calibration_policy)
    identity_map_path = _resolve(args.identity_map)
    out_path = _resolve(args.out)
    assert lineups_path is not None and events_path is not None and out_path is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lineups = pd.read_csv(lineups_path)
    events = pd.read_csv(events_path)
    cal_pred = pd.read_csv(cal_pred_path) if cal_pred_path and cal_pred_path.exists() else None
    cal_results = pd.read_csv(cal_results_path) if cal_results_path and cal_results_path.exists() else None
    market_policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path and policy_path.exists() else None

    preds = predict_props_for_lineups(
        events,
        lineups,
        markets=args.markets,
        line=args.line,
        calibration_predictions=cal_pred,
        calibration_results=cal_results,
        min_calibration_rows=args.min_calibration_rows,
        strict_lineup_contract=bool(args.strict_lineup_contract),
        use_hierarchical_calibration=not bool(args.disable_hierarchical_calibration),
        min_hierarchical_group_rows=args.min_hierarchical_group_rows,
        market_calibration_policy=market_policy,
        identity_map=str(identity_map_path) if identity_map_path else None,
    )
    preds.to_csv(out_path, index=False)
    payload = {
        "status": "SAFE_LINEUP_PROPS_COMPLETE",
        "lineups": str(lineups_path),
        "player_events": str(events_path),
        "calibration_predictions": str(cal_pred_path) if cal_pred_path else None,
        "calibration_results": str(cal_results_path) if cal_results_path else None,
        "calibration_policy": str(policy_path) if policy_path else None,
        "identity_map": str(identity_map_path) if identity_map_path else None,
        "identity_map_status_counts": preds.get("identity_map_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not preds.empty and "identity_map_status" in preds.columns else {},
        "provider_players": int(preds.get("canonical_player_id", pd.Series(dtype=str)).nunique()) if not preds.empty and "canonical_player_id" in preds.columns else 0,
        "out": str(out_path),
        "input_lineup_players": int(lineups["player"].nunique()) if "player" in lineups.columns else None,
        "output_rows": int(len(preds)),
        "markets": sorted(preds["market_type"].unique().tolist()) if not preds.empty else [],
        "warnings_count": int((preds.get("warnings", pd.Series(dtype=str)).fillna("").astype(str) != "").sum()) if not preds.empty else 0,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
