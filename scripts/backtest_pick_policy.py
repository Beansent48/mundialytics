#!/usr/bin/env python3
"""Backtest and tune a transparent pick policy from historical predictions.

This script is deliberately honest about what it can and cannot prove:
- With historical bookmaker odds: evaluates real value picks via ROI/profit/EV.
- Without historical odds: evaluates only model signal calibration/hit-rate. That is useful,
  but it is not proof of betting profitability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.betting.pick_policy import (
    add_chronological_split,
    attach_odds,
    backtest_best_policy,
    build_match_pick_signals,
    standardize_settled_line_signals,
    evaluate_policy_grid,
    summary_from_outputs,
    evaluate_model_performance_by_market,
    build_market_takeaways,
    evaluate_threshold_performance_by_market,
)


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def _write_odds_template(signals: pd.DataFrame, out_path: Path) -> None:
    """Write a user-fillable odds template matching generated signal rows."""
    cols = [
        "match_id", "date", "home_team", "away_team", "market",
        "selection", "line", "book_odds",
    ]
    if signals.empty:
        pd.DataFrame(columns=cols).to_csv(out_path, index=False)
        return
    work = signals.copy()
    for col in cols:
        if col not in work.columns:
            work[col] = ""
    template = work[cols].drop_duplicates().sort_values(["date", "match_id", "market", "line", "selection"], kind="mergesort")
    template.to_csv(out_path, index=False)




def _csv_dtype_hints() -> dict[str, str]:
    """Avoid expensive/misleading mixed-type inference on huge line-signal files."""
    return {
        "match_id": "string",
        "date": "string",
        "home_team": "string",
        "away_team": "string",
        "team": "string",
        "player": "string",
        "goalkeeper": "string",
        "competition": "string",
        "competition_context": "string",
        "team_type": "string",
        "gender": "string",
        "market": "string",
        "scope": "string",
        "selection": "string",
        "over_under": "string",
        "data_source": "string",
        "data_quality_flag": "string",
        "saves_data_quality_flag": "string",
        "target_quality": "string",
        "expected_components": "string",
        "model_family": "string",
    }


def _prefilter_line_signals(
    df: pd.DataFrame,
    *,
    min_prob: float,
    markets: str | None,
    target_quality: str,
    max_rows: int,
) -> pd.DataFrame:
    """Filter giant settled line-signal files before policy-grid evaluation.

    The event-line generator can easily produce millions of rows because it emits both
    over and under for many lines. For pick-policy training, rows below the minimum
    probability threshold can never be selected, so filtering them here is lossless
    with respect to the tested policy grid.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "model_probability" in work.columns:
        prob = pd.to_numeric(work["model_probability"], errors="coerce")
        work = work[prob.ge(float(min_prob))].copy()
    if markets:
        wanted = {m.strip().lower() for m in str(markets).replace(";", ",").split(",") if m.strip()}
        if wanted and "market" in work.columns:
            work = work[work["market"].astype(str).str.lower().isin(wanted)].copy()
    if target_quality and str(target_quality).lower() not in {"all", ""} and "target_quality" in work.columns:
        wanted_q = {q.strip().lower() for q in str(target_quality).replace(";", ",").split(",") if q.strip()}
        work = work[work["target_quality"].astype(str).str.lower().isin(wanted_q)].copy()
    if max_rows and int(max_rows) > 0 and len(work) > int(max_rows):
        # Deterministic stratified cap: keep strongest probabilities per signal group/market where possible.
        sort_cols = [c for c in ["signal_group", "market", "model_probability"] if c in work.columns]
        if "model_probability" in work.columns:
            work["_prob_sort"] = pd.to_numeric(work["model_probability"], errors="coerce")
            group_col = "signal_group" if "signal_group" in work.columns else ("market" if "market" in work.columns else None)
            if group_col:
                n_groups = max(1, work[group_col].nunique(dropna=False))
                per_group = max(1, int(max_rows) // n_groups)
                work = (
                    work.sort_values("_prob_sort", ascending=False, kind="mergesort")
                    .groupby(group_col, dropna=False, group_keys=False)
                    .head(per_group)
                    .head(int(max_rows))
                    .copy()
                )
            else:
                work = work.sort_values("_prob_sort", ascending=False, kind="mergesort").head(int(max_rows)).copy()
            work = work.drop(columns=["_prob_sort"], errors="ignore")
        else:
            work = work.head(int(max_rows)).copy()
    return work


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/backtest Mundialytics pick policy on historical predictions.")
    parser.add_argument("--match-backtest", required=True, help="CSV with historical match predictions and actual results, e.g. outputs/evaluation_current/match_backtest_predictions.csv")
    parser.add_argument("--historical-odds", default=None, help="Optional odds CSV. Without it this is signal-only, not betting ROI.")
    parser.add_argument("--line-signals", action="append", default=[], help="Optional settled line-signal CSV(s) for sportsbook-style markets beyond 1X2/goals/BTTS: corners, shots, shots_on_target, cards, fouls, goalkeeper_saves. Must contain model_probability and actual_win or settled_stat.")
    parser.add_argument("--line-min-model-prob", type=float, default=0.52, help="Prefilter line-signals before evaluation. Rows below this probability can never be selected by the default policy grid, so 0.52 is effectively lossless and much faster.")
    parser.add_argument("--line-markets", default=None, help="Optional comma-separated market filter for huge line-signal files, e.g. corners,team_corners,goalkeeper_saves.")
    parser.add_argument("--line-target-quality", default="all", help="Optional comma-separated target_quality filter: real_target,match_total,derived_target,unknown_quality,all.")
    parser.add_argument("--line-max-rows", type=int, default=0, help="Optional deterministic cap after filtering for quick experiments. 0 means no cap.")
    parser.add_argument("--no-full-signals-csv", action="store_true", help="Do not write giant pick_policy_signals.csv. Useful when line-signals has millions of rows.")
    parser.add_argument("--out-dir", default="outputs/pick_policy_backtest_current")
    parser.add_argument("--min-picks", type=int, default=30, help="Minimum validation picks needed for a policy to be selectable")
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--validation-frac", type=float, default=0.20)
    parser.add_argument("--top-print", type=int, default=12)
    parser.add_argument("--write-odds-template", action="store_true", help="Write a historical_odds_template.csv with all signal rows to fill bookmaker prices later")
    args = parser.parse_args(argv)

    match_path = _resolve(args.match_backtest)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    match_predictions = pd.read_csv(match_path)
    base_signals = build_match_pick_signals(match_predictions)
    extra_signals = []
    for line_path_text in args.line_signals:
        line_path = _resolve(line_path_text)
        if not line_path.exists():
            print(f"Aviso: line-signals no existe: {line_path}")
            continue
        try:
            line_df = pd.read_csv(line_path, low_memory=False, dtype=_csv_dtype_hints())
        except Exception as exc:
            print(f"Aviso: no se pudo leer line-signals {line_path}: {exc}")
            continue
        before_rows = len(line_df)
        line_df = _prefilter_line_signals(
            line_df,
            min_prob=args.line_min_model_prob,
            markets=args.line_markets,
            target_quality=args.line_target_quality,
            max_rows=args.line_max_rows,
        )
        print(f"Line-signals loaded: {line_path} | rows {before_rows} -> {len(line_df)} after prefilter")
        standardized = standardize_settled_line_signals(line_df)
        if standardized.empty:
            print(f"Aviso: line-signals sin filas evaluables: {line_path}")
        else:
            extra_signals.append(standardized)
    frames = [df for df in [base_signals, *extra_signals] if df is not None and not df.empty]
    signals = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if signals.empty:
        print("No se han podido crear señales históricas. Revisa columnas de match_backtest o line-signals.")
        return 1

    if args.write_odds_template:
        template_path = out_dir / "historical_odds_template.csv"
        _write_odds_template(signals, template_path)
        print(f"Odds template written: {template_path}")

    require_odds = False
    if args.historical_odds:
        odds_path = _resolve(args.historical_odds)
        if not odds_path.exists():
            print(f"Aviso: historical-odds no existe: {odds_path}")
            print("Se continuará en modo SIGNAL-ONLY. No se medirá ROI real.")
            template_path = out_dir / "historical_odds_template.csv"
            _write_odds_template(signals, template_path)
            print(f"Plantilla para rellenar cuotas creada en: {template_path}")
            signals = attach_odds(signals, None)
        else:
            odds = pd.read_csv(odds_path)
            signals = attach_odds(signals, odds)
            require_odds = signals.get("book_odds", pd.Series(dtype=float)).notna().any()
            if not require_odds:
                print("Aviso: se pasó historical-odds, pero no se pudo adjuntar ninguna cuota. Se hará evaluación signal-only.")
    else:
        signals = attach_odds(signals, None)

    signals = add_chronological_split(signals, train_frac=args.train_frac, validation_frac=args.validation_frac)
    market_summary, line_summary = evaluate_model_performance_by_market(signals)
    threshold_summary = evaluate_threshold_performance_by_market(signals)
    # Useful filtered views: these make it obvious whether unders and BTTS No work.
    selection_perf = line_summary[line_summary["selection"].astype(str).ne("all")].copy() if not line_summary.empty else pd.DataFrame()
    selection_threshold = threshold_summary[threshold_summary.get("eval_level", pd.Series(dtype=str)).isin(["selection", "signal_group", "line_selection"])].copy() if not threshold_summary.empty else pd.DataFrame()
    market_takeaways = build_market_takeaways(market_summary)
    leaderboard, best_policy = evaluate_policy_grid(signals, min_picks=args.min_picks, require_odds=require_odds)
    selected = backtest_best_policy(signals, best_policy, require_odds=require_odds)
    summary = summary_from_outputs(signals, leaderboard, selected, require_odds=require_odds, best_policy=best_policy)
    best_by_signal_group = []
    if not leaderboard.empty and "allowed_signal_group" in leaderboard.columns:
        valid = leaderboard[leaderboard.get("selection_valid", False).eq(True)].copy() if "selection_valid" in leaderboard.columns else leaderboard.copy()
        for group, g in valid.groupby("allowed_signal_group", dropna=False):
            if not g.empty:
                best_by_signal_group.append(g.iloc[0].to_dict())

    if not args.no_full_signals_csv:
        signals.to_csv(out_dir / "pick_policy_signals.csv", index=False)
    else:
        pd.DataFrame({"note": ["full signals CSV skipped by --no-full-signals-csv"], "signals_rows": [len(signals)]}).to_csv(out_dir / "pick_policy_signals_SKIPPED.csv", index=False)
    market_summary.to_csv(out_dir / "market_model_performance.csv", index=False)
    line_summary.to_csv(out_dir / "market_line_performance.csv", index=False)
    threshold_summary.to_csv(out_dir / "market_threshold_performance.csv", index=False)
    selection_perf.to_csv(out_dir / "market_selection_performance.csv", index=False)
    selection_threshold.to_csv(out_dir / "market_selection_threshold_performance.csv", index=False)
    (out_dir / "market_model_takeaways.json").write_text(json.dumps(market_takeaways, indent=2, default=str), encoding="utf-8")
    leaderboard.to_csv(out_dir / "pick_policy_leaderboard.csv", index=False)
    pd.DataFrame(best_by_signal_group).to_csv(out_dir / "pick_policy_best_by_signal_group.csv", index=False)
    selected.to_csv(out_dir / "pick_policy_selected_picks.csv", index=False)
    (out_dir / "pick_policy_best.json").write_text(json.dumps(best_policy, indent=2, default=str), encoding="utf-8")
    (out_dir / "pick_policy_best_by_signal_group.json").write_text(json.dumps(best_by_signal_group, indent=2, default=str), encoding="utf-8")
    (out_dir / "pick_policy_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("MUNDIALYTICS PICK POLICY + EXTRA MARKET/SIDE MODEL BACKTEST v0.38.4")
    print(f"Output dir: {out_dir}")
    print(f"Signals: {len(signals)} rows | matches: {signals['match_id'].nunique()}")
    if require_odds:
        print("Mode: REAL VALUE BACKTEST con cuotas históricas")
    else:
        print("Mode: SIGNAL-ONLY (sin cuotas históricas; NO mide ROI real)")
    print("\nBest policy:")
    print(json.dumps(best_policy, indent=2, default=str))
    print("\nTop policies:")
    cols = [
        "policy_id", "allowed_markets", "allowed_signal_group", "min_model_probability", "min_fair_odds", "max_fair_odds",
        "validation_n_picks", "validation_hit_rate", "validation_avg_model_probability", "validation_calibration_gap",
        "validation_roi", "validation_profit", "test_n_picks", "test_hit_rate", "test_roi", "test_profit",
    ]
    cols = [c for c in cols if c in leaderboard.columns]
    print(leaderboard[cols].head(args.top_print).to_string(index=False))
    print("\nMarket model performance, test split:")
    if not market_summary.empty:
        cols_m = ["market", "n", "hit_rate", "avg_model_probability", "calibration_gap", "brier", "log_loss"]
        test_m = market_summary[market_summary["split"].eq("test")][cols_m]
        print(test_m.to_string(index=False))
    print("\nHigh-confidence market performance, test split:")
    if not threshold_summary.empty:
        cols_t = ["eval_level", "market", "selection", "signal_group", "line", "min_model_probability", "n", "hit_rate", "avg_model_probability", "calibration_gap", "avg_fair_odds"]
        cols_t = [c for c in cols_t if c in threshold_summary.columns]
        test_t = threshold_summary[(threshold_summary["split"].eq("test")) & (threshold_summary["min_model_probability"].ge(0.70))][cols_t]
        # Keep console readable: show the most useful side-aware rows first.
        if "eval_level" in test_t.columns:
            preferred = test_t[test_t["eval_level"].isin(["signal_group", "selection"])].copy()
            if not preferred.empty:
                test_t = preferred
        print(test_t.head(80).to_string(index=False))
    print("\nSelected-picks summary:")
    print(json.dumps(summary["selected_picks_summary"], indent=2, default=str))
    if best_by_signal_group:
        print("\nBest policy by signal group (sample):")
        show_cols = ["allowed_signal_group", "policy_id", "min_model_probability", "min_fair_odds", "max_fair_odds", "validation_n_picks", "validation_hit_rate", "test_n_picks", "test_hit_rate"]
        bb = pd.DataFrame(best_by_signal_group)
        show_cols = [c for c in show_cols if c in bb.columns]
        print(bb[show_cols].head(20).to_string(index=False))
    if not require_odds:
        print("\nOJO: esto entrena/elige una política de señal estadística. Para entrenar picks de apuestas de verdad necesitamos historical_odds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
