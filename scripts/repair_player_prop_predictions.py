from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _null_rate(s: pd.Series) -> float:
    if len(s) == 0:
        return 1.0
    return float(s.isna().mean())


def main() -> None:
    p = argparse.ArgumentParser(description="Repair player-prop backtest predictions with match metadata from player_events CSV.")
    p.add_argument("--predictions", required=True, help="player_props_backtest_predictions.csv or calibrated_player_prop_predictions.csv")
    p.add_argument("--player-events", required=True, help="statsbomb_player_events.csv containing date/competition/team_scope")
    p.add_argument("--out", required=True, help="Repaired predictions CSV")
    p.add_argument("--report", default=None, help="Optional repair report JSON")
    args = p.parse_args()

    pred_path = _resolve(args.predictions)
    events_path = _resolve(args.player_events)
    out_path = _resolve(args.out)
    report_path = _resolve(args.report) if args.report else None
    assert pred_path is not None and events_path is not None and out_path is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(pred_path)
    events = pd.read_csv(events_path)
    if "match_id" not in pred.columns or "match_id" not in events.columns:
        raise ValueError("Both predictions and player-events must contain match_id")

    pred["match_id"] = pred["match_id"].astype(str)
    events["match_id"] = events["match_id"].astype(str)

    if "date" not in pred.columns:
        pred["date"] = pd.NA
    before_date_null = _null_rate(pred["date"])

    meta_cols = [c for c in ["match_id", "date", "competition", "team_scope", "source"] if c in events.columns]
    meta = events[meta_cols].drop_duplicates("match_id", keep="first")
    rename = {c: f"{c}_meta" for c in meta.columns if c != "match_id"}
    meta = meta.rename(columns=rename)
    out = pred.merge(meta, on="match_id", how="left")

    for col in ["date", "competition", "team_scope", "source"]:
        mcol = f"{col}_meta"
        if mcol in out.columns:
            if col not in out.columns:
                out[col] = out[mcol]
            else:
                current = out[col]
                out[col] = current.where(current.notna() & (current.astype(str).str.strip() != ""), out[mcol])
            out = out.drop(columns=[mcol])

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype("string")
    after_date_null = _null_rate(pd.to_datetime(out["date"], errors="coerce")) if "date" in out.columns else 1.0

    out.to_csv(out_path, index=False)

    report = {
        "status": "PREDICTIONS_REPAIRED",
        "input_predictions": str(pred_path),
        "player_events": str(events_path),
        "output": str(out_path),
        "rows": int(len(out)),
        "date_null_rate_before": before_date_null,
        "date_null_rate_after": after_date_null,
        "matched_metadata_rows": int(out["competition"].notna().sum()) if "competition" in out.columns else None,
        "competitions_top": out["competition"].value_counts().head(30).to_dict() if "competition" in out.columns else {},
        "team_scope_counts": out["team_scope"].value_counts(dropna=False).to_dict() if "team_scope" in out.columns else {},
        "warnings": [],
    }
    if after_date_null > 0.05:
        report["warnings"].append(f"high_date_null_rate_after={after_date_null:.3f}")
    if report_path:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
