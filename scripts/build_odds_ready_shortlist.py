#!/usr/bin/env python3
"""Build an odds-ready shortlist/template from settled model line signals.

This does not fetch odds and does not alter model performance. It only produces a
small, provider-agnostic contract so a future odds API can be plugged in safely.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.betting.odds_contract import (
    ODDS_INPUT_COLUMNS,
    MODEL_LINE_COLUMNS,
    classify_fair_odds_bucket,
    confidence_from_row,
    min_acceptable_odds_from_probability,
    standard_model_line_frame,
    write_contract_files,
)


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def _csv_dtype_hints() -> dict[str, str]:
    return {
        "match_id": "string", "date": "string", "home_team": "string", "away_team": "string",
        "team": "string", "player": "string", "goalkeeper": "string", "competition": "string",
        "market": "string", "scope": "string", "selection": "string", "signal_group": "string",
        "target_quality": "string", "data_quality_flag": "string", "saves_data_quality_flag": "string",
        "model_family": "string", "expected_components": "string",
    }


def _parse_csv_list(text: str | None, default: set[str] | None = None) -> set[str]:
    if not text:
        return default or set()
    return {x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()}


def _load_decision_filter(path: Path, decisions: set[str], min_fair_bucket: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if decisions and "decision" in df.columns:
        df = df[df["decision"].astype(str).isin(decisions)].copy()
    # Keep all buckets by default; caller usually sets min_model_probability/fair odds row filters.
    needed_cols = [c for c in ["signal_group", "target_quality", "fair_odds_bucket", "decision", "reason_codes", "test_n", "test_hit_rate", "test_calibration_gap", "test_avg_fair_odds"] if c in df.columns]
    df = df[needed_cols].drop_duplicates().copy()
    if "confidence_label" not in df.columns:
        df["confidence_label"] = df.apply(confidence_from_row, axis=1)
    return df


def build_shortlist(
    line_signals: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    *,
    min_model_probability: float,
    min_fair_odds: float,
    max_fair_odds: float,
    max_rows_per_signal_group: int,
    min_ev: float,
    min_edge: float,
) -> pd.DataFrame:
    if line_signals.empty:
        return pd.DataFrame(columns=MODEL_LINE_COLUMNS)
    work = line_signals.copy()
    if "signal_group" not in work.columns:
        if "market" in work.columns and "selection" in work.columns:
            work["signal_group"] = work["market"].astype(str) + "_" + work["selection"].astype(str)
        else:
            work["signal_group"] = ""
    if "target_quality" not in work.columns:
        work["target_quality"] = ""
    work["model_probability"] = pd.to_numeric(work.get("model_probability"), errors="coerce")
    if "fair_odds" not in work.columns:
        work["fair_odds"] = 1.0 / work["model_probability"].clip(1e-6, 1 - 1e-6)
    else:
        work["fair_odds"] = pd.to_numeric(work["fair_odds"], errors="coerce")
    work = work[
        work["model_probability"].ge(float(min_model_probability))
        & work["fair_odds"].ge(float(min_fair_odds))
        & work["fair_odds"].le(float(max_fair_odds))
    ].copy()
    if work.empty:
        return standard_model_line_frame(work)
    if not decision_matrix.empty:
        # Join by signal group + target quality + bucket when possible; this preserves market-side decisions.
        work["fair_odds_bucket"] = work["fair_odds"].map(classify_fair_odds_bucket)
        join_cols = ["signal_group"]
        if "target_quality" in decision_matrix.columns:
            join_cols.append("target_quality")
        if "fair_odds_bucket" in decision_matrix.columns:
            join_cols.append("fair_odds_bucket")
        dm = decision_matrix.drop_duplicates(join_cols).copy()
        work = work.merge(dm, on=join_cols, how="inner", suffixes=("", "_decision"))
    if work.empty:
        return standard_model_line_frame(work)
    # Keep strongest/most relevant rows by signal group to avoid massive templates.
    sort_cols = ["signal_group", "model_probability"]
    work = work.sort_values(sort_cols, ascending=[True, False], kind="mergesort")
    if max_rows_per_signal_group > 0:
        work = work.groupby("signal_group", dropna=False, group_keys=False).head(int(max_rows_per_signal_group)).copy()
    work["min_acceptable_odds"] = work["model_probability"].map(lambda p: min_acceptable_odds_from_probability(p, min_ev=min_ev, min_edge=min_edge))
    if "decision_reason_codes" not in work.columns:
        if "reason_codes" in work.columns:
            work["decision_reason_codes"] = work["reason_codes"]
        else:
            work["decision_reason_codes"] = ""
    if "decision" not in work.columns:
        work["decision"] = "candidate_unscored"
    if "confidence_label" not in work.columns:
        work["confidence_label"] = work.apply(confidence_from_row, axis=1)
    out = standard_model_line_frame(work)
    return out


def build_odds_template(model_lines: pd.DataFrame) -> pd.DataFrame:
    if model_lines.empty:
        return pd.DataFrame(columns=ODDS_INPUT_COLUMNS)
    rows = model_lines.copy()
    out = pd.DataFrame({
        "snapshot_time_utc": "",
        "bookmaker": "",
        "provider": "",
        "provider_event_id": "",
        "internal_match_id": rows["match_id"],
        "match_id": rows["match_id"],
        "date": rows["date"],
        "home_team": rows["home_team"],
        "away_team": rows["away_team"],
        "market_key": rows["market_key"],
        "market": rows["market"],
        "scope": rows["scope"],
        "subject_team": rows["subject_team"],
        "subject_player": rows["subject_player"],
        "line": rows["line"],
        "side": rows["side"],
        "bookmaker_odds": "",
        "is_live": False,
        "source_url": "",
        "notes": "",
    })
    return out[ODDS_INPUT_COLUMNS].drop_duplicates().sort_values(["date", "match_id", "market_key", "scope", "line", "side"], kind="mergesort")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create odds-ready model lines and a compact historical odds template.")
    parser.add_argument("--line-signals", required=True, help="settled_event_line_signals.csv or compatible model lines")
    parser.add_argument("--decision-matrix", required=True, help="market_side_decision_matrix.csv from analyze_market_distribution_lab.py")
    parser.add_argument("--out-dir", default="outputs/odds_ready_current")
    parser.add_argument("--decisions", default="candidate", help="Comma-separated decisions to include, e.g. candidate,needs_calibration")
    parser.add_argument("--min-model-probability", type=float, default=0.52)
    parser.add_argument("--min-fair-odds", type=float, default=1.25, help="Avoid ultra-low fair odds in the odds template.")
    parser.add_argument("--max-fair-odds", type=float, default=3.50)
    parser.add_argument("--max-rows-per-signal-group", type=int, default=5000)
    parser.add_argument("--min-ev", type=float, default=0.03)
    parser.add_argument("--min-edge", type=float, default=0.02)
    parser.add_argument("--line-max-rows", type=int, default=0, help="Optional row cap for quick tests before filtering.")
    args = parser.parse_args(argv)

    line_path = _resolve(args.line_signals)
    matrix_path = _resolve(args.decision_matrix)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions = _parse_csv_list(args.decisions, {"candidate"})
    decision_matrix = _load_decision_filter(matrix_path, decisions=decisions, min_fair_bucket=None)
    line_df = pd.read_csv(line_path, low_memory=False, dtype=_csv_dtype_hints())
    raw_rows = len(line_df)
    if args.line_max_rows and args.line_max_rows > 0 and raw_rows > args.line_max_rows:
        line_df = line_df.head(int(args.line_max_rows)).copy()
    model_lines = build_shortlist(
        line_df,
        decision_matrix,
        min_model_probability=args.min_model_probability,
        min_fair_odds=args.min_fair_odds,
        max_fair_odds=args.max_fair_odds,
        max_rows_per_signal_group=args.max_rows_per_signal_group,
        min_ev=args.min_ev,
        min_edge=args.min_edge,
    )
    odds_template = build_odds_template(model_lines)
    model_lines.to_csv(out_dir / "model_market_lines.csv", index=False)
    odds_template.to_csv(out_dir / "odds_needed_template.csv", index=False)
    write_contract_files(out_dir)
    summary = {
        "version": "v0.40_odds_ready_layer",
        "line_signals_rows_raw": raw_rows,
        "model_market_lines_rows": int(len(model_lines)),
        "odds_needed_template_rows": int(len(odds_template)),
        "decisions_included": sorted(decisions),
        "min_model_probability": args.min_model_probability,
        "min_fair_odds": args.min_fair_odds,
        "max_fair_odds": args.max_fair_odds,
        "min_ev_for_min_acceptable_odds": args.min_ev,
        "min_edge_for_min_acceptable_odds": args.min_edge,
        "markets": model_lines["market_key"].value_counts().to_dict() if not model_lines.empty else {},
        "signal_groups": model_lines["signal_group"].value_counts().to_dict() if not model_lines.empty else {},
        "warning": "This only prepares odds/API integration. It does not fetch odds, place bets, or change model predictions.",
    }
    (out_dir / "odds_ready_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("MUNDIALYTICS ODDS-READY SHORTLIST v0.40")
    print(f"Input line rows: {raw_rows}")
    print(f"Model market lines: {len(model_lines)}")
    print(f"Odds needed template: {len(odds_template)}")
    print(f"Output dir: {out_dir}")
    print("Generated:")
    print(f"- {out_dir / 'model_market_lines.csv'}")
    print(f"- {out_dir / 'odds_needed_template.csv'}")
    print(f"- {out_dir / 'odds_ready_contract.json'}")
    print("OJO: no se han descargado cuotas ni se ha tocado el modelo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
