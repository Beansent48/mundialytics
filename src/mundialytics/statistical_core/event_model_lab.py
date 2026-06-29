from __future__ import annotations

import json
import math
import shutil
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from mundialytics.statistical_core.event_evaluation import EventEvaluationConfig, evaluate_event_models_temporal, write_event_evaluation_outputs
from mundialytics.statistical_core.schemas import write_json


DEFAULT_EVENT_TRIALS: list[tuple[str, dict[str, Any]]] = [
    ("baseline_v023", {"team_model_config": {}, "player_model_config": {}}),
    ("team_own60_player_share60", {"team_model_config": {"own_weight": 0.60}, "player_model_config": {"share_weight": 0.60}}),
    ("team_own65_player_rate45", {"team_model_config": {"own_weight": 0.65}, "player_model_config": {"share_weight": 0.55}}),
    ("team_shrink8_low8", {"team_model_config": {"own_weight": 0.60, "profile_shrinkage_k": 8.0, "low_sample_blend_k": 8.0}, "player_model_config": {"share_weight": 0.60}}),
    ("team_shrink15_low12", {"team_model_config": {"own_weight": 0.60, "profile_shrinkage_k": 15.0, "low_sample_blend_k": 12.0}, "player_model_config": {"share_weight": 0.60}}),
    ("recency_slow_player_rate", {"team_model_config": {"own_weight": 0.60, "recency_half_life_days": 720.0, "profile_shrinkage_k": 8.0}, "player_model_config": {"share_weight": 0.50}}),
    ("recency_fast_cards_conservative", {"team_model_config": {"own_weight": 0.60, "recency_half_life_days": 180.0, "profile_shrinkage_k": 8.0}, "player_model_config": {"share_weight": 0.55, "yellow_card_cap": 0.50}}),
    ("player_share70", {"team_model_config": {"own_weight": 0.60, "profile_shrinkage_k": 8.0}, "player_model_config": {"share_weight": 0.70}}),
    ("player_rate60", {"team_model_config": {"own_weight": 0.60, "profile_shrinkage_k": 8.0}, "player_model_config": {"share_weight": 0.40}}),
]


def run_event_model_lab(
    historical_events: pd.DataFrame,
    out_dir: str | Path,
    n_trials: int | None = None,
    clean_out_dir: bool = False,
    max_test_matches: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = Path(out_dir)
    if clean_out_dir and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    trials = DEFAULT_EVENT_TRIALS[: int(n_trials)] if n_trials else DEFAULT_EVENT_TRIALS
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_score = float("inf")
    for idx, (name, config) in enumerate(trials, start=1):
        trial_id = f"trial_{idx:03d}"
        try:
            cfg = EventEvaluationConfig(
                test_fraction=0.25,
                min_train_matches=50,
                max_test_matches=max_test_matches,
                team_model_config=dict(config.get("team_model_config", {})),
                player_model_config=dict(config.get("player_model_config", {})),
            )
            team_scored, team_lines, player_scored, player_lines, summary = evaluate_event_models_temporal(historical_events, cfg)
            team_metrics = summary.get("team_event_performance", {}).get("count_metrics_by_market", {})
            player_metrics = summary.get("player_prop_performance", {}).get("prop_metrics_by_market", {})
            shots_mae = _get(team_metrics, "shots", "mae")
            sot_mae = _get(team_metrics, "shots_on_target", "mae")
            fouls_mae = _get(team_metrics, "fouls", "mae")
            cards_mae = _get(team_metrics, "yellow_cards", "mae")
            player_shots_brier = _get(player_metrics, "player_shots", "brier_1plus")
            player_sot_brier = _get(player_metrics, "player_shots_on_target", "brier_1plus")
            player_fouls_brier = _get(player_metrics, "player_fouls_committed", "brier_1plus")
            player_cards_brier = _get(player_metrics, "player_yellow_card", "brier_1plus")
            # Lower is better. Scale count MAE roughly by market magnitude so one event does not dominate.
            objective = (
                0.10 * shots_mae / 10.0
                + 0.10 * sot_mae / 3.5
                + 0.10 * fouls_mae / 11.0
                + 0.15 * cards_mae / 1.8
                + 0.20 * player_shots_brier
                + 0.15 * player_sot_brier
                + 0.10 * player_fouls_brier
                + 0.10 * player_cards_brier
            )
            row = {
                "trial_id": trial_id,
                "trial_name": name,
                "objective": objective,
                "team_shots_mae": shots_mae,
                "team_sot_mae": sot_mae,
                "team_fouls_mae": fouls_mae,
                "team_yellow_cards_mae": cards_mae,
                "player_shots_brier": player_shots_brier,
                "player_sot_brier": player_sot_brier,
                "player_fouls_brier": player_fouls_brier,
                "player_yellow_card_brier": player_cards_brier,
                "event_model_config_json": json.dumps(config, sort_keys=True),
                "status": summary.get("status"),
            }
            rows.append(row)
            trial_dir = out / "trials" / trial_id
            paths = write_event_evaluation_outputs(trial_dir, team_scored, team_lines, player_scored, player_lines, summary)
            if objective < best_score:
                best_score = objective
                best_payload = {
                    "status": "completed",
                    "version": "v0.24_event_model_lab",
                    "trial_id": trial_id,
                    "trial_name": name,
                    "objective": objective,
                    "event_model_config": config,
                    "summary": summary,
                    "artifact_paths": paths,
                }
        except Exception as exc:  # pragma: no cover
            failed.append({"trial_id": trial_id, "trial_name": name, "error": str(exc), "traceback": traceback.format_exc(), "event_model_config": config})
    leaderboard = pd.DataFrame(rows).sort_values("objective", ascending=True).reset_index(drop=True) if rows else pd.DataFrame()
    if not leaderboard.empty:
        leaderboard.to_csv(out / "event_experiment_leaderboard.csv", index=False)
    write_json(out / "failed_event_experiments.json", failed)
    if best_payload is None:
        best_payload = {"status": "no_successful_trials", "failed_experiments": failed}
    write_json(out / "best_event_model_config.json", best_payload)
    report = build_event_model_lab_report(out / "event_model_lab_report.html", leaderboard, best_payload, failed)
    write_json(out / "event_model_lab_audit.json", {"status": "completed" if rows else "failed", "trials_run": len(rows), "trials_failed": len(failed), "best_trial": best_payload.get("trial_id"), "report": str(report)})
    best_payload["report"] = str(report)
    write_json(out / "best_event_model_config.json", best_payload)
    return leaderboard, best_payload


def _get(d: dict[str, Any], market: str, metric: str) -> float:
    try:
        v = float(d.get(market, {}).get(metric))
    except Exception:
        return float("inf")
    return v if math.isfinite(v) else float("inf")


def build_event_model_lab_report(path: str | Path, leaderboard: pd.DataFrame, best_payload: dict[str, Any], failed: list[dict[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    def esc(s: str) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>Mundialytics Event Model Lab v0.24</title>"]
    html.append("<style>body{font-family:Arial,sans-serif;margin:28px} table{border-collapse:collapse;width:100%;font-size:13px} th,td{border:1px solid #ddd;padding:6px} th{background:#f4f4f4}.warn{background:#fff4d6;border:1px solid #d7aa28;padding:10px}</style></head><body>")
    html.append("<h1>Mundialytics Event Model Lab v0.24</h1>")
    html.append("<p>Automatic evaluation loop for team events and player props: shots, shots on target, fouls and yellow cards.</p>")
    best = {k: v for k, v in best_payload.items() if k not in {"summary"}}
    html.append("<h2>Best event model</h2><pre>" + esc(json.dumps(best, indent=2, ensure_ascii=False)) + "</pre>")
    if not leaderboard.empty:
        cols = [c for c in ["trial_id", "trial_name", "objective", "team_shots_mae", "team_sot_mae", "team_fouls_mae", "team_yellow_cards_mae", "player_shots_brier", "player_sot_brier", "player_fouls_brier", "player_yellow_card_brier"] if c in leaderboard.columns]
        html.append("<h2>Leaderboard</h2>")
        html.append(leaderboard[cols].to_html(index=False, float_format=lambda x: f"{x:.6f}"))
    if failed:
        html.append("<h2>Failed experiments</h2><div class='warn'><pre>" + esc(json.dumps(failed, indent=2, ensure_ascii=False)[:12000]) + "</pre></div>")
    html.append("</body></html>")
    out.write_text("\n".join(html), encoding="utf-8")
    return out
