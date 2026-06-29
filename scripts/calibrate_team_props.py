from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.features.team_match_stats import TEAM_PROP_MARKETS, predict_team_props_simple


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _metrics(y, pred) -> dict:
    y = pd.to_numeric(y, errors="coerce")
    pred = pd.to_numeric(pred, errors="coerce")
    mask = y.notna() & pred.notna()
    if not mask.any():
        return {"n": 0}
    return {
        "n": int(mask.sum()),
        "actual_mean": float(y[mask].mean()),
        "pred_mean": float(pred[mask].mean()),
        "bias": float(pred[mask].mean() - y[mask].mean()),
        "mae": float(mean_absolute_error(y[mask], pred[mask])),
        "rmse": float(mean_squared_error(y[mask], pred[mask]) ** 0.5),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Temporal calibration check for team prop expected-count predictions.")
    p.add_argument("--team-match-stats", required=True)
    p.add_argument("--out-dir", default="outputs/team_props_calibration")
    p.add_argument("--markets", nargs="+", default=TEAM_PROP_MARKETS)
    p.add_argument("--calibration-fraction", type=float, default=0.5)
    p.add_argument("--test-matches", type=int, default=200)
    p.add_argument("--scale-floor", type=float, default=0.25, help="Guardrail for simple calibration scale.")
    p.add_argument("--scale-cap", type=float, default=4.0, help="Guardrail for simple calibration scale.")
    args = p.parse_args()
    df = pd.read_csv(_resolve(args.team_match_stats))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    matches = df[["match_id", "date"]].drop_duplicates().sort_values(["date", "match_id"]).tail(args.test_matches)
    if matches.empty:
        raise SystemExit("No matches available for team prop calibration.")
    test_ids = set(matches["match_id"])
    eval_rows = df[df["match_id"].isin(test_ids)].copy()
    # Predict each selected match using only rows before that match by passing the
    # selected team rows as pseudo-fixtures. This is conservative and leakage-safe.
    pseudo_fixtures = eval_rows.rename(columns={"team": "home_team", "opponent": "away_team"})[["match_id", "date", "competition", "team_scope", "team_type", "competition_context", "gender", "home_team", "away_team"]].copy()
    pseudo_fixtures["fixture_id"] = pseudo_fixtures["match_id"].astype(str) + "__" + pseudo_fixtures["home_team"].astype(str)
    pseudo_fixtures["neutral"] = 1
    preds = predict_team_props_simple(df[~df["match_id"].isin(test_ids)].copy(), pseudo_fixtures, markets=args.markets)
    # The pseudo fixture creates both team perspectives; keep rows matching the original team/opponent pair.
    merged = eval_rows.merge(preds, left_on=["match_id", "team", "opponent"], right_on=["match_id", "team", "opponent"], suffixes=("", "_pred"), how="left")
    # Split the evaluation rows into calibration/test halves by date and learn simple scale factors.
    merged = merged.sort_values(["date", "match_id", "team"])
    cut = int(len(merged) * args.calibration_fraction)
    cal = merged.iloc[:cut].copy()
    hold = merged.iloc[cut:].copy()
    market_reports = {}
    for market in args.markets:
        exp_col = f"expected_{market}"
        if market not in merged.columns or exp_col not in merged.columns:
            market_reports[market] = {"status": "unavailable"}
            continue
        y_cal = pd.to_numeric(cal[market], errors="coerce")
        p_cal = pd.to_numeric(cal[exp_col], errors="coerce")
        mask = y_cal.notna() & p_cal.notna() & (p_cal > 0)
        raw_scale = float(y_cal[mask].sum() / p_cal[mask].sum()) if mask.any() and p_cal[mask].sum() > 0 else 1.0
        scale = max(args.scale_floor, min(args.scale_cap, raw_scale))
        raw = _metrics(hold[market], hold[exp_col])
        calibrated_pred = pd.to_numeric(hold[exp_col], errors="coerce") * scale
        calibrated = _metrics(hold[market], calibrated_pred)
        market_reports[market] = {"raw_scale": raw_scale, "scale": scale, "scale_guardrail_applied": scale != raw_scale, "raw": raw, "calibrated": calibrated}
    out_dir = _resolve(args.out_dir)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "team_prop_temporal_predictions.csv", index=False)
    report = {
        "status": "TEAM_PROPS_TEMPORAL_CALIBRATION_COMPLETE",
        "team_match_stats": str(_resolve(args.team_match_stats)),
        "n_rows": int(len(df)),
        "n_eval_rows": int(len(merged)),
        "markets": market_reports,
        "outputs": {"predictions": str(out_dir / "team_prop_temporal_predictions.csv")},
        "note": "Scale calibration is a baseline. Next upgrade: Negative Binomial with hierarchical calibration.",
    }
    (out_dir / "team_prop_calibration_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
