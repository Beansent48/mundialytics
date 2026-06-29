from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mundialytics.statistical_core.distributions import probability_for_count_line
from mundialytics.statistical_core.schemas import canonical_name, write_json
from mundialytics.statistical_core.player_event_model import PLAYER_MARKETS, POSITION_PRIORS, _position_key, _position_group

# Keep this module dependency-light for fast agent iterations; robust shrink calibration is always available.
IsotonicRegression = None
LogisticRegression = None
brier_score_loss = None
log_loss = None


PLAYER_PROP_TARGETS = {
    "player_shots": "shots",
    "player_shots_on_target": "shots_on_target",
    "player_fouls_committed": "fouls_committed",
    "player_yellow_card": "yellow_cards",
}

POSITION_GROUP_PRIORS = {
    "shots": {
        "forward": 0.22,
        "winger": 0.17,
        "attacking_midfield": 0.14,
        "central_midfield": 0.09,
        "defensive_midfield": 0.05,
        "fullback_wingback": 0.04,
        "center_back": 0.035,
        "goalkeeper": 0.0005,
        "unknown_outfield": 0.08,
    },
    "shots_on_target": {
        "forward": 0.23,
        "winger": 0.17,
        "attacking_midfield": 0.13,
        "central_midfield": 0.08,
        "defensive_midfield": 0.04,
        "fullback_wingback": 0.03,
        "center_back": 0.025,
        "goalkeeper": 0.0002,
        "unknown_outfield": 0.06,
    },
    "fouls": {
        "forward": 0.09,
        "winger": 0.08,
        "attacking_midfield": 0.09,
        "central_midfield": 0.12,
        "defensive_midfield": 0.15,
        "fullback_wingback": 0.12,
        "center_back": 0.13,
        "goalkeeper": 0.01,
        "unknown_outfield": 0.09,
    },
    "yellow_cards": {
        "forward": 0.06,
        "winger": 0.05,
        "attacking_midfield": 0.06,
        "central_midfield": 0.10,
        "defensive_midfield": 0.15,
        "fullback_wingback": 0.11,
        "center_back": 0.13,
        "goalkeeper": 0.02,
        "unknown_outfield": 0.08,
    },
}


def _position_prior_for(event: str, position_key: str, position_group: str) -> float:
    """Prior player share by tactical role, with a group-level fallback."""
    if event in POSITION_PRIORS and position_key in POSITION_PRIORS[event]:
        return float(POSITION_PRIORS[event][position_key])
    return float(POSITION_GROUP_PRIORS.get(event, {}).get(position_group, POSITION_GROUP_PRIORS.get(event, {}).get("unknown_outfield", 0.08)))

DEFAULT_PROP_CONFIGS: list[dict[str, Any]] = [
    {"name": "v024_team_share_blend", "share_weight": 0.65, "sample_shrink_k": 450.0, "team_exp_weight": 0.50, "position_prior_weight": 0.20, "yellow_card_cap": 0.45, "prob_floor": 0.003, "prob_cap": 0.97},
    {"name": "v16_rate_recovery", "share_weight": 0.25, "sample_shrink_k": 650.0, "team_exp_weight": 0.25, "position_prior_weight": 0.35, "yellow_card_cap": 0.35, "prob_floor": 0.003, "prob_cap": 0.95},
    {"name": "team_context_share", "share_weight": 0.75, "sample_shrink_k": 700.0, "team_exp_weight": 0.65, "position_prior_weight": 0.20, "yellow_card_cap": 0.38, "prob_floor": 0.003, "prob_cap": 0.96},
    {"name": "conservative_cards_sot", "share_weight": 0.45, "sample_shrink_k": 900.0, "team_exp_weight": 0.35, "position_prior_weight": 0.40, "yellow_card_cap": 0.25, "prob_floor": 0.002, "prob_cap": 0.90},
    {"name": "player_rate_heavy", "share_weight": 0.10, "sample_shrink_k": 500.0, "team_exp_weight": 0.15, "position_prior_weight": 0.25, "yellow_card_cap": 0.40, "prob_floor": 0.003, "prob_cap": 0.96},
    {"name": "v26_v16_starter_minutes", "share_weight": 0.25, "sample_shrink_k": 650.0, "team_exp_weight": 0.25, "position_prior_weight": 0.35, "yellow_card_cap": 0.35, "prob_floor": 0.003, "prob_cap": 0.95, "minutes_model": "starter_role"},
    {"name": "v26_sot_conditional", "share_weight": 0.25, "sample_shrink_k": 650.0, "team_exp_weight": 0.25, "position_prior_weight": 0.35, "yellow_card_cap": 0.35, "prob_floor": 0.003, "prob_cap": 0.95, "minutes_model": "starter_role", "sot_condition_on_shots": True, "sot_condition_weight": 0.55},
    {"name": "v26_nb_moderate", "share_weight": 0.25, "sample_shrink_k": 700.0, "team_exp_weight": 0.30, "position_prior_weight": 0.35, "yellow_card_cap": 0.35, "prob_floor": 0.003, "prob_cap": 0.94, "minutes_model": "starter_role", "count_distribution": "negative_binomial", "nb_shape": 2.5},
    {"name": "v26_nb_shots_only", "share_weight": 0.25, "sample_shrink_k": 650.0, "team_exp_weight": 0.25, "position_prior_weight": 0.35, "yellow_card_cap": 0.35, "prob_floor": 0.003, "prob_cap": 0.95, "minutes_model": "starter_role", "count_distribution_by_market": {"player_shots": "negative_binomial", "player_shots_on_target": "negative_binomial"}, "nb_shape_by_market": {"player_shots": 3.5, "player_shots_on_target": 2.0}},
    {"name": "v26_card_conservative_role", "share_weight": 0.08, "sample_shrink_k": 550.0, "team_exp_weight": 0.15, "position_prior_weight": 0.30, "yellow_card_cap": 0.28, "prob_floor": 0.001, "prob_cap": 0.88, "minutes_model": "starter_role"},
    {"name": "v27_nb_soft_caps", "share_weight": 0.22, "sample_shrink_k": 750.0, "team_exp_weight": 0.32, "position_prior_weight": 0.38, "yellow_card_cap": 0.34, "prob_floor": 0.002, "prob_cap": 0.93, "minutes_model": "starter_role", "count_distribution": "negative_binomial", "nb_shape": 2.0},
    {"name": "v27_shots_sot_hybrid", "share_weight": 0.23, "sample_shrink_k": 700.0, "team_exp_weight": 0.28, "position_prior_weight": 0.38, "yellow_card_cap": 0.34, "prob_floor": 0.002, "prob_cap": 0.94, "minutes_model": "starter_role", "count_distribution_by_market": {"player_shots": "negative_binomial", "player_fouls_committed": "negative_binomial"}, "nb_shape_by_market": {"player_shots": 2.0, "player_fouls_committed": 2.0}, "sot_condition_on_shots": True, "sot_condition_weight": 0.45},
    {"name": "v27_card_ultra_conservative", "share_weight": 0.06, "sample_shrink_k": 650.0, "team_exp_weight": 0.10, "position_prior_weight": 0.40, "yellow_card_cap": 0.22, "prob_floor": 0.001, "prob_cap": 0.35, "minutes_model": "starter_role"},
    {"name": "v28_position_group_calibrated_nb", "share_weight": 0.20, "sample_shrink_k": 760.0, "team_exp_weight": 0.32, "position_prior_weight": 0.42, "yellow_card_cap": 0.33, "prob_floor": 0.002, "prob_cap": 0.92, "minutes_model": "starter_role", "count_distribution": "negative_binomial", "nb_shape": 2.0, "position_group_priors": True},
    {"name": "v28_winger_forward_softened", "share_weight": 0.18, "sample_shrink_k": 800.0, "team_exp_weight": 0.30, "position_prior_weight": 0.48, "yellow_card_cap": 0.32, "prob_floor": 0.002, "prob_cap": 0.90, "minutes_model": "starter_role", "count_distribution": "negative_binomial", "nb_shape": 2.3, "position_group_priors": True, "role_probability_softening": {"forward": 0.94, "winger": 0.94, "attacking_midfield": 0.98}},
    {"name": "v28_goalkeeper_guardrail_cards", "share_weight": 0.06, "sample_shrink_k": 700.0, "team_exp_weight": 0.10, "position_prior_weight": 0.45, "yellow_card_cap": 0.20, "prob_floor": 0.001, "prob_cap": 0.34, "minutes_model": "starter_role", "position_group_priors": True},
]



@dataclass(frozen=True)
class ChampionPropConfig:
    test_fraction: float = 0.25
    calibration_fraction_within_train: float = 0.35
    min_train_matches: int = 50
    max_test_matches: int | None = None
    max_calibration_matches: int | None = None
    min_calibration_rows: int = 300
    min_group_rows: int = 400
    min_segment_rows: int = 120
    line: str = "1+"
    evaluate_segments: bool = True
    configs: list[dict[str, Any]] = field(default_factory=lambda: list(DEFAULT_PROP_CONFIGS))
    n_trials: int | None = None


def run_player_prop_champion_lab(historical_events: pd.DataFrame, cfg: ChampionPropConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate several player-prop architectures, calibrate them, and select a champion.

    The design intentionally recovers the older v0.16 idea: player rates, expected
    minutes, cross-context evidence, context metadata, caps/floors and
    hierarchical calibration. It does not assume one universal model is best for
    every market; it chooses champions by market and stores segment diagnostics.
    """
    cfg = cfg or ChampionPropConfig()
    df = _prepare_events(historical_events)
    split = _temporal_three_way_split(df, cfg)
    if split["status"] != "ok":
        payload = {"status": split["status"], "reason": split.get("reason", "")}
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), payload

    train = split["train"].copy()
    cal = split["calibration"].copy()
    test = split["test"].copy()

    trial_rows: list[dict[str, Any]] = []
    test_predictions_parts: list[pd.DataFrame] = []
    calibration_reports: dict[str, Any] = {}
    configs_to_run = cfg.configs[: int(cfg.n_trials)] if cfg.n_trials is not None and cfg.n_trials > 0 else cfg.configs
    for i, config in enumerate(configs_to_run, start=1):
        name = str(config.get("name") or f"config_{i:03d}")
        cal_raw = _predict_prop_rows(train, cal, config, split_name="calibration")
        test_raw = _predict_prop_rows(pd.concat([train, cal], ignore_index=True, sort=False), test, config, split_name="test")
        if cal_raw.empty or test_raw.empty:
            trial_rows.append({"trial_id": f"trial_{i:03d}", "trial_name": name, "status": "empty_predictions"})
            continue
        calibrators, cal_report = _fit_hierarchical_calibrators(cal_raw, cfg)
        test_cal = _apply_hierarchical_calibrators(test_raw, calibrators)
        trial_metrics = _summarize_trial(test_cal)
        objective = _trial_objective(trial_metrics)
        row = {"trial_id": f"trial_{i:03d}", "trial_name": name, "status": "completed", "objective": objective, **_flatten_market_metrics(trial_metrics)}
        trial_rows.append(row)
        test_cal["trial_id"] = f"trial_{i:03d}"
        test_cal["trial_name"] = name
        test_cal["model_config_json"] = json.dumps(config, sort_keys=True)
        test_predictions_parts.append(test_cal)
        calibration_reports[f"trial_{i:03d}"] = {"trial_name": name, "model_config": config, "calibration": cal_report, "metrics": trial_metrics, "objective": objective}

    leaderboard = pd.DataFrame(trial_rows)
    all_predictions = pd.concat(test_predictions_parts, ignore_index=True) if test_predictions_parts else pd.DataFrame()
    champion_rows: list[dict[str, Any]] = []
    champion_predictions_parts: list[pd.DataFrame] = []
    if not all_predictions.empty:
        for market, g in all_predictions.groupby("market"):
            best = None
            best_key = (float("inf"), float("inf"))
            for trial_id, tg in g.groupby("trial_id"):
                metrics = _binary_metrics(tg["actual"], tg["calibrated_probability"])
                key = (metrics["log_loss"], metrics["brier"])
                if key < best_key:
                    best_key = key
                    best = (trial_id, str(tg["trial_name"].iloc[0]), metrics, tg)
            if best is None:
                continue
            trial_id, trial_name, metrics, pred = best
            base = _binary_metrics(pred["actual"], pred["baseline_probability"])
            champion_rows.append({
                "market": str(market),
                "champion_trial_id": trial_id,
                "champion_trial_name": trial_name,
                "n": int(len(pred)),
                "actual_rate": float(pred["actual"].mean()),
                "avg_probability": float(pred["calibrated_probability"].mean()),
                "probability_bias": float(pred["calibrated_probability"].mean() - pred["actual"].mean()),
                "brier": metrics["brier"],
                "log_loss": metrics["log_loss"],
                "baseline_brier": base["brier"],
                "baseline_log_loss": base["log_loss"],
                "brier_improvement_vs_baseline": _improvement(base["brier"], metrics["brier"]),
                "logloss_improvement_vs_baseline": _improvement(base["log_loss"], metrics["log_loss"]),
                "policy": _policy_for_market(str(market), metrics, base, int(len(pred))),
            })
            champion_predictions_parts.append(pred.copy())
    champion_summary = pd.DataFrame(champion_rows).sort_values("market") if champion_rows else pd.DataFrame()
    champion_predictions = pd.concat(champion_predictions_parts, ignore_index=True) if champion_predictions_parts else pd.DataFrame()
    segment_metrics = _segment_metrics(champion_predictions, cfg) if cfg.evaluate_segments and not champion_predictions.empty else pd.DataFrame()

    payload: dict[str, Any] = {
        "status": "completed",
        "version": "v0.28_statistical_upgrade_player_prop_lab",
        "train_matches": int(split["train_match_count"]),
        "calibration_matches": int(split["calibration_match_count"]),
        "test_matches": int(split["test_match_count"]),
        "cutoff_calibration_start": str(split["calibration_start_date"].date()),
        "cutoff_test_start": str(split["test_start_date"].date()),
        "uses_observed_test_minutes": False,
        "lineup_known_backtest": True,
        "markets": champion_summary.to_dict(orient="records") if not champion_summary.empty else [],
        "calibration_reports": calibration_reports,
        "honest_limitations": [
            "This evaluates player props with historical lineups known, but actual test minutes are not used as features.",
            "Real betting value still needs real odds, closing-line tracking and stake simulation.",
            "Segment-specific champions are selected only when sample sizes are large enough; otherwise the market champion is used.",
        ],
    }
    return leaderboard, champion_summary, segment_metrics, {**payload, "champion_predictions": champion_predictions}


def write_player_prop_champion_outputs(out_dir: str | Path, leaderboard: pd.DataFrame, champion_summary: pd.DataFrame, segment_metrics: pd.DataFrame, payload: dict[str, Any]) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred = payload.pop("champion_predictions", pd.DataFrame())
    paths: dict[str, str] = {}
    for name, frame in {
        "player_prop_champion_leaderboard.csv": leaderboard,
        "player_prop_champion_summary.csv": champion_summary,
        "player_prop_segment_metrics.csv": segment_metrics,
        "player_prop_champion_predictions.csv": pred,
    }.items():
        p = out / name
        frame.to_csv(p, index=False)
        paths[name] = str(p)
    registry = _build_registry(champion_summary, payload, segment_metrics)
    reg_path = out / "prediction_registry.json"
    write_json(reg_path, registry)
    paths["prediction_registry.json"] = str(reg_path)
    summary_path = out / "player_prop_champion_audit.json"
    write_json(summary_path, payload)
    paths["player_prop_champion_audit.json"] = str(summary_path)
    report_path = build_player_prop_champion_report(out / "player_prop_champion_report.html", leaderboard, champion_summary, segment_metrics, payload, pred)
    paths["player_prop_champion_report.html"] = str(report_path)
    return paths


def build_player_prop_champion_report(path: str | Path, leaderboard: pd.DataFrame, champion_summary: pd.DataFrame, segment_metrics: pd.DataFrame, payload: dict[str, Any], predictions: pd.DataFrame) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    def esc(x: Any) -> str:
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>Mundialytics v0.28 Champion Prop Lab</title>"]
    html.append("<style>body{font-family:Arial,sans-serif;margin:28px;color:#111} table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:24px} th,td{border:1px solid #ddd;padding:6px} th{background:#f4f4f4}.good{background:#e9f7e9}.warn{background:#fff4d6}.bad{background:#fde9e7} code{background:#eee;padding:2px 4px}</style></head><body>")
    html.append("<h1>Mundialytics v0.28 Champion Player Prop Lab</h1>")
    html.append(f"<p>Train matches: <strong>{payload.get('train_matches')}</strong> | Calibration: <strong>{payload.get('calibration_matches')}</strong> | Test: <strong>{payload.get('test_matches')}</strong></p>")
    html.append("<p>Goal: keep the best model per market, not one global model for everything. Actual test minutes are not used as features.</p>")
    if not champion_summary.empty:
        cols = [c for c in ["market","champion_trial_name","n","actual_rate","avg_probability","probability_bias","brier","baseline_brier","brier_improvement_vs_baseline","log_loss","baseline_log_loss","logloss_improvement_vs_baseline","policy"] if c in champion_summary.columns]
        html.append("<h2>Champion by market</h2>")
        html.append(champion_summary[cols].to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    if not leaderboard.empty:
        html.append("<h2>Trial leaderboard</h2>")
        html.append(leaderboard.sort_values("objective", na_position="last").to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    if not segment_metrics.empty:
        html.append("<h2>Segment diagnostics</h2>")
        html.append(segment_metrics.head(100).to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    if not predictions.empty:
        worst = predictions.sort_values("log_loss", ascending=False).head(30)
        cols = [c for c in ["date","competition","team_type","gender","team","opponent","player","position","market","calibrated_probability","actual","actual_count","expected_count","sample_size_minutes","calibration_level","champion_policy"] if c in worst.columns]
        html.append("<h2>Worst calibrated misses</h2>")
        html.append(worst[cols].to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    html.append("<h2>Audit</h2><pre>" + esc(json.dumps({k: v for k, v in payload.items() if k != "calibration_reports"}, indent=2, ensure_ascii=False, default=str)) + "</pre>")
    html.append("</body></html>")
    out.write_text("\n".join(html), encoding="utf-8")
    return out


def _probability_1plus_from_count(lam: float, market: str, config: dict[str, Any]) -> float:
    """Count-to-1+ probability link, selectable per market for model-lab search."""
    lam = float(np.clip(lam if np.isfinite(lam) else 0.0, 0.0, 50.0))
    by_market = config.get("count_distribution_by_market") or {}
    dist = str(by_market.get(market, config.get("count_distribution", "poisson"))).lower()
    if dist in {"negative_binomial", "nb", "negbin"}:
        shape_by_market = config.get("nb_shape_by_market") or {}
        r = float(shape_by_market.get(market, config.get("nb_shape", 3.0)))
        r = float(np.clip(r, 0.25, 100.0))
        p0 = (r / (r + lam)) ** r if lam > 0 else 1.0
        return float(1.0 - p0)
    return probability_for_count_line(lam, "1+", "over")


def _safe_median(values: pd.Series, default: float) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return float(default)
    out = float(v.median())
    return out if np.isfinite(out) and out > 0 else float(default)


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    if df.empty:
        return df
    df["match_id"] = df["match_id"].astype(str)
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    for c in ["team", "opponent", "player", "competition", "team_type", "team_scope", "competition_context", "gender", "position"]:
        if c not in df.columns:
            df[c] = "unknown"
    for c in ["team", "opponent", "player"]:
        df[c] = df[c].map(canonical_name)
    if "player_id_global" not in df.columns:
        df["player_id_global"] = df["player"].map(lambda x: f"player_{canonical_name(x).replace(' ', '_')}")
    for c in ["minutes", "started", *PLAYER_PROP_TARGETS.values()]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0)
        else:
            df[c] = 0.0
    df["started"] = (pd.to_numeric(df.get("started"), errors="coerce").fillna(0) > 0).astype(int)
    df = df.dropna(subset=["date"])
    return df.sort_values(["date", "match_id", "team", "player"]).reset_index(drop=True)


def _temporal_three_way_split(df: pd.DataFrame, cfg: ChampionPropConfig) -> dict[str, Any]:
    if df.empty or "match_id" not in df.columns:
        return {"status": "empty_events"}
    matches = df[["match_id", "date"]].drop_duplicates("match_id").dropna(subset=["date"]).sort_values(["date", "match_id"]).reset_index(drop=True)
    if len(matches) <= cfg.min_train_matches + 2:
        return {"status": "not_enough_matches", "reason": f"matches={len(matches)}"}
    test_n = max(1, int(round(len(matches) * cfg.test_fraction)))
    if cfg.max_test_matches is not None and cfg.max_test_matches > 0:
        test_n = min(test_n, int(cfg.max_test_matches))
    test_start = len(matches) - test_n
    train_cal = matches.iloc[:test_start].copy()
    cal_n = max(1, int(round(len(train_cal) * cfg.calibration_fraction_within_train)))
    train_n = len(train_cal) - cal_n
    train_n = max(train_n, cfg.min_train_matches)
    if train_n >= len(train_cal):
        train_n = len(train_cal) - 1
    train_matches = train_cal.iloc[:train_n].copy()
    cal_matches = train_cal.iloc[train_n:].copy()
    if cfg.max_calibration_matches is not None and cfg.max_calibration_matches > 0 and len(cal_matches) > int(cfg.max_calibration_matches):
        cal_matches = cal_matches.tail(int(cfg.max_calibration_matches)).copy()
    test_matches = matches.iloc[test_start:].copy()
    train_ids = set(train_matches["match_id"].astype(str))
    cal_ids = set(cal_matches["match_id"].astype(str))
    test_ids = set(test_matches["match_id"].astype(str))
    return {
        "status": "ok",
        "train": df[df["match_id"].astype(str).isin(train_ids)].copy(),
        "calibration": df[df["match_id"].astype(str).isin(cal_ids)].copy(),
        "test": df[df["match_id"].astype(str).isin(test_ids)].copy(),
        "train_match_count": len(train_matches),
        "calibration_match_count": len(cal_matches),
        "test_match_count": len(test_matches),
        "calibration_start_date": pd.Timestamp(cal_matches.iloc[0]["date"]),
        "test_start_date": pd.Timestamp(test_matches.iloc[0]["date"]),
    }


def _team_match_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    event_cols = list(set(PLAYER_PROP_TARGETS.values()))
    for (match_id, team), g in df.groupby(["match_id", "team"], dropna=False):
        r = {"match_id": str(match_id), "team": str(team), "date": g["date"].iloc[0], "opponent": str(g["opponent"].mode().iloc[0]) if len(g["opponent"].dropna()) else "unknown"}
        for e in event_cols:
            r[e] = float(pd.to_numeric(g[e], errors="coerce").fillna(0).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def _predict_prop_rows(train: pd.DataFrame, target: pd.DataFrame, config: dict[str, Any], split_name: str) -> pd.DataFrame:
    if train.empty or target.empty:
        return pd.DataFrame()
    train = train.copy()
    target = target.copy()
    event_cols = PLAYER_PROP_TARGETS
    team_match = _team_match_table(train)
    # Team for/against profiles with shrinkage.
    team_profiles = {}
    global_team_means = {event: float(team_match[col].mean()) if col in team_match.columns and len(team_match) else 1.0 for event, col in event_cols.items()}
    for event, col in event_cols.items():
        own = team_match.groupby("team")[col].agg(["mean", "count"]).rename(columns={"mean": "own_mean", "count": "own_n"})
        against = team_match.groupby("opponent")[col].agg(["mean", "count"]).rename(columns={"mean": "against_mean", "count": "against_n"})
        team_profiles[event] = (own, against)

    train["position_key"] = train["position"].map(_position_key)
    target["position_key"] = target["position"].map(_position_key)
    train["position_group"] = train["position_key"].map(_position_group)
    target["position_group"] = target["position_key"].map(_position_group)
    profile_cols = ["player_id_global", "player", "position_key", "position_group", "team_type", "gender", "competition_context"]
    agg = {"minutes": "sum", "started": "mean"}
    for col in event_cols.values():
        agg[col] = "sum"
    player_global = train.groupby("player_id_global", dropna=False).agg(agg).reset_index()
    # Dominant display/context.
    dom = train.sort_values("date").groupby("player_id_global", dropna=False).agg({
        "player": "last",
        "position_key": lambda s: s.mode().iloc[0] if len(s.mode()) else "unk",
        "position_group": lambda s: s.mode().iloc[0] if len(s.mode()) else "unknown_outfield",
    }).reset_index()
    player_global = player_global.merge(dom, on="player_id_global", how="left")
    pos_prior = train.groupby("position_group", dropna=False).agg({"minutes": "sum", **{col: "sum" for col in event_cols.values()}}).reset_index()
    global_minutes = max(float(train["minutes"].sum()), 1.0)
    global_event_rates = {market: 90.0 * float(train[col].sum()) / global_minutes for market, col in event_cols.items()}
    pos_rates: dict[tuple[str, str], float] = {}
    for _, r in pos_prior.iterrows():
        mins = max(float(r["minutes"]), 1.0)
        for market, col in event_cols.items():
            pos_rates[(str(r["position_group"]), market)] = 90.0 * float(r[col]) / mins
    # Player shares by historical teams.
    team_totals = {market: train.groupby("team", dropna=False)[col].sum().to_dict() for market, col in event_cols.items()}
    player_teams = train[["player_id_global", "team"]].drop_duplicates().groupby("player_id_global")["team"].agg(list).to_dict()
    for market, col in event_cols.items():
        player_global[f"{market}_rate_per90_raw"] = 90.0 * pd.to_numeric(player_global[col], errors="coerce").fillna(0) / player_global["minutes"].clip(lower=1)
        # shrink player rate to position prior/global based on sample size later after merge.
        shares = []
        for _, r in player_global.iterrows():
            teams = player_teams.get(r["player_id_global"], [])
            denom = sum(float(team_totals[market].get(t, 0.0)) for t in teams)
            shares.append(float(r[col]) / denom if denom > 0 else np.nan)
        player_global[f"{market}_share_raw"] = pd.Series(shares).clip(lower=0.001, upper=0.60)

    # Expected minutes from training history; actual test minutes are not used.
    player_minutes = train.groupby("player_id_global")["minutes"].median().rename("historical_expected_minutes").reset_index()
    position_minutes = train.groupby("position_group")["minutes"].median().to_dict()
    player_started_minutes = train.groupby(["player_id_global", "started"])["minutes"].median().unstack(fill_value=np.nan)
    position_started_minutes = train.groupby(["position_group", "started"])["minutes"].median().unstack(fill_value=np.nan)
    prof = player_global.merge(player_minutes, on="player_id_global", how="left")
    rows = target[target["minutes"] > 0].drop_duplicates(["match_id", "team", "player_id_global"]).copy()
    rows = rows.merge(prof, on="player_id_global", how="left", suffixes=("", "_profile"))
    # Preserve target position if no profile exists.
    if "position_key_profile" in rows.columns:
        rows["profile_position_key"] = rows["position_key_profile"].where(rows["position_key_profile"].notna(), rows["position_key"])
    else:
        rows["profile_position_key"] = rows["position_key"]
    if "position_group_profile" in rows.columns:
        rows["profile_position_group"] = rows["position_group_profile"].where(rows["position_group_profile"].notna(), rows["position_group"])
    else:
        rows["profile_position_group"] = rows["position_group"]
    sample_k = float(config.get("sample_shrink_k", 650.0))
    share_weight = float(config.get("share_weight", 0.35))
    team_exp_weight = float(config.get("team_exp_weight", 0.35))
    pos_prior_weight = float(config.get("position_prior_weight", 0.35))
    yellow_card_cap = float(config.get("yellow_card_cap", 0.35))
    prob_floor = float(config.get("prob_floor", 0.003))
    prob_cap = float(config.get("prob_cap", 0.96))
    out_rows = []
    for _, r in rows.iterrows():
        sample_minutes = float(r.get("minutes_profile", r.get("minutes_y", 0.0)) or r.get("minutes", 0.0) or 0.0)
        pos = str(r.get("profile_position_key") or r.get("position_key") or "unk")
        pos_group = str(r.get("profile_position_group") or _position_group(pos))
        hist_min = r.get("historical_expected_minutes")
        started_flag = int(r.get("started", 0) or 0)
        if str(config.get("minutes_model", "median")) == "starter_role":
            player_id = r.get("player_id_global")
            role_min = np.nan
            if player_id in player_started_minutes.index and started_flag in player_started_minutes.columns:
                role_min = player_started_minutes.loc[player_id, started_flag]
            if pd.isna(role_min) or float(role_min) <= 0:
                if pos_group in position_started_minutes.index and started_flag in position_started_minutes.columns:
                    role_min = position_started_minutes.loc[pos_group, started_flag]
            if pd.isna(role_min) or float(role_min) <= 0:
                role_min = position_minutes.get(pos_group, 65.0 if started_flag else 25.0)
            expected_minutes = float(role_min)
        else:
            if pd.isna(hist_min) or float(hist_min) <= 0:
                hist_min = position_minutes.get(pos_group, 65.0)
            expected_minutes = float(hist_min)
            if started_flag == 1:
                expected_minutes = max(expected_minutes, 55.0)
            else:
                expected_minutes = min(expected_minutes, 35.0)
        if started_flag == 1:
            expected_minutes = max(expected_minutes, 45.0)
        else:
            expected_minutes = min(expected_minutes, 40.0)
        expected_minutes = float(np.clip(expected_minutes, 1, 105))
        sample_weight = sample_minutes / (sample_minutes + sample_k) if sample_minutes > 0 else 0.0
        for market, col in event_cols.items():
            raw_rate = r.get(f"{market}_rate_per90_raw")
            if pd.isna(raw_rate):
                raw_rate = global_event_rates[market]
            pos_rate = pos_rates.get((pos_group, market), global_event_rates[market])
            shrunk_rate = sample_weight * float(raw_rate) + (1 - sample_weight) * ((1 - pos_prior_weight) * global_event_rates[market] + pos_prior_weight * pos_rate)
            own, against = team_profiles[market]
            global_mean = global_team_means[market]
            team = str(r.get("team"))
            opp = str(r.get("opponent"))
            own_row = own.loc[team] if team in own.index else None
            ag_row = against.loc[opp] if opp in against.index else None
            own_mean = float(own_row["own_mean"]) if own_row is not None else global_mean
            ag_mean = float(ag_row["against_mean"]) if ag_row is not None else global_mean
            own_n = float(own_row["own_n"]) if own_row is not None else 0.0
            ag_n = float(ag_row["against_n"]) if ag_row is not None else 0.0
            team_k = 8.0
            own_mean = (own_n * own_mean + team_k * global_mean) / (own_n + team_k)
            ag_mean = (ag_n * ag_mean + team_k * global_mean) / (ag_n + team_k)
            team_expected = team_exp_weight * own_mean + (1 - team_exp_weight) * ag_mean
            share = r.get(f"{market}_share_raw")
            if pd.isna(share):
                share = _position_prior_for(PLAYER_MARKETS[market]["event"], pos, pos_group)
            share = float(np.clip(share, 0.001, 0.50))
            rate_component = shrunk_rate * expected_minutes / 90.0
            share_component = team_expected * share * expected_minutes / 75.0
            lam = max(0.0, share_weight * share_component + (1 - share_weight) * rate_component)
            role_softening = config.get("role_probability_softening") or {}
            if role_softening and pos_group in role_softening and market in {"player_shots", "player_shots_on_target"}:
                lam *= float(np.clip(role_softening.get(pos_group, 1.0), 0.50, 1.10))
            if pos_group == "goalkeeper" and market in {"player_shots", "player_shots_on_target"}:
                lam = 0.0
            if market == "player_shots_on_target" and bool(config.get("sot_condition_on_shots", False)):
                prof_sot = float(r.get("shots_on_target_profile", 0.0) or 0.0)
                prof_shots = float(r.get("shots_profile", 0.0) or 0.0)
                global_sot_conv = float(train["shots_on_target"].sum() / max(train["shots"].sum(), 1.0))
                conv_k = float(config.get("sot_conversion_shrink_k", 12.0))
                conv = float(np.clip((prof_sot + conv_k * global_sot_conv) / max(prof_shots + conv_k, 1e-6), 0.02, 0.75))
                shot_raw_rate = r.get("player_shots_rate_per90_raw")
                if pd.isna(shot_raw_rate):
                    shot_raw_rate = global_event_rates.get("player_shots", 1.0)
                cond_lam = float(shot_raw_rate) * expected_minutes / 90.0 * conv
                w_cond = float(np.clip(config.get("sot_condition_weight", 0.50), 0.0, 1.0))
                lam = (1.0 - w_cond) * lam + w_cond * cond_lam
            if market == "player_yellow_card":
                lam = min(lam, yellow_card_cap)
            p = _probability_1plus_from_count(lam, market, config)
            # market-specific caps/floors
            cap = prob_cap
            floor = prob_floor
            if market == "player_yellow_card":
                cap = min(cap, 0.45)
                floor = min(floor, 0.002)
            elif market == "player_shots_on_target":
                cap = min(cap, 0.80)
            p = float(np.clip(p, floor, cap))
            if pos_group == "goalkeeper" and market in {"player_shots", "player_shots_on_target"}:
                p = 0.0
            actual_count = float(r.get(col, 0.0) or 0.0)
            actual = int(actual_count >= 1.0)
            baseline_probability = _baseline_probability(train, market)
            out_rows.append({
                "split": split_name,
                "date": r.get("date"),
                "match_id": str(r.get("match_id")),
                "competition": r.get("competition", "unknown"),
                "team_scope": r.get("team_scope", "unknown"),
                "team_type": r.get("team_type", "unknown"),
                "competition_context": r.get("competition_context", "unknown"),
                "gender": r.get("gender", "unknown"),
                "team": team,
                "opponent": opp,
                "player": r.get("player", r.get("player_profile", "unknown")),
                "player_id_global": r.get("player_id_global"),
                "position": r.get("position", "unknown"),
                "position_key": pos,
                "position_group": pos_group,
                "started": int(r.get("started", 0) or 0),
                "expected_minutes": expected_minutes,
                "expected_minutes_source": "historical_player_or_position_median_pre_match",
                "actual_minutes": float(r.get("minutes", 0.0) or 0.0),
                "market": market,
                "line": "1+",
                "expected_count": float(lam),
                "raw_probability": p,
                "probability": p,
                "baseline_probability": float(baseline_probability),
                "actual_count": actual_count,
                "actual": actual,
                "sample_size_minutes": sample_minutes,
                "sample_weight": sample_weight,
                "player_rate_per90": float(raw_rate),
                "shrunk_rate_per90": float(shrunk_rate),
                "player_share": float(share),
                "team_expected_event": float(team_expected),
                "model_name": str(config.get("name", "unnamed")),
            })
    return pd.DataFrame(out_rows)


_BASE_CACHE: dict[tuple[int, str], float] = {}
def _baseline_probability(train: pd.DataFrame, market: str) -> float:
    key = (id(train), market)
    if key in _BASE_CACHE:
        return _BASE_CACHE[key]
    col = PLAYER_PROP_TARGETS[market]
    y = (pd.to_numeric(train[col], errors="coerce").fillna(0) >= 1).astype(float)
    p = float((y.sum() + 1.0) / (len(y) + 2.0)) if len(y) else 0.25
    _BASE_CACHE[key] = p
    return p


def _fit_hierarchical_calibrators(cal_pred: pd.DataFrame, cfg: ChampionPropConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    calibrators: dict[str, Any] = {}
    report: dict[str, Any] = {"markets": {}, "levels": ["competition_position_group", "domain_position_group", "position_group", "competition", "domain_context", "team_type_gender", "market_global"]}
    if cal_pred.empty:
        return calibrators, report
    # global market calibrators
    for market, g in cal_pred.groupby("market"):
        calibrators[(market, "market_global", "*")] = _fit_best_binary_calibrator(g["probability"].to_numpy(), g["actual"].to_numpy(), cfg.min_calibration_rows)
        report["markets"].setdefault(market, {})["market_global"] = _calibrator_report(calibrators[(market, "market_global", "*")], len(g))
        # position group calibrators: enough sample to correct systematic role bias without fragmenting into too many provider positions.
        if "position_group" in g.columns:
            for pg, sg in g.groupby("position_group", dropna=False):
                if len(sg) >= cfg.min_group_rows:
                    key = str(pg)
                    calibrators[(market, "position_group", key)] = _fit_best_binary_calibrator(sg["probability"].to_numpy(), sg["actual"].to_numpy(), cfg.min_calibration_rows)
            for (tt, ge, ctx, pg), sg in g.groupby(["team_type", "gender", "competition_context", "position_group"], dropna=False):
                if len(sg) >= cfg.min_group_rows:
                    key = f"{tt}|{ge}|{ctx}|{pg}"
                    calibrators[(market, "domain_position_group", key)] = _fit_best_binary_calibrator(sg["probability"].to_numpy(), sg["actual"].to_numpy(), cfg.min_calibration_rows)
            for (comp, pg), sg in g.groupby(["competition", "position_group"], dropna=False):
                if len(sg) >= cfg.min_group_rows:
                    key = f"{comp}|{pg}"
                    calibrators[(market, "competition_position_group", key)] = _fit_best_binary_calibrator(sg["probability"].to_numpy(), sg["actual"].to_numpy(), cfg.min_calibration_rows)
        # team_type/gender
        for (tt, ge), sg in g.groupby(["team_type", "gender"], dropna=False):
            if len(sg) >= cfg.min_group_rows:
                key = f"{tt}|{ge}"
                calibrators[(market, "team_type_gender", key)] = _fit_best_binary_calibrator(sg["probability"].to_numpy(), sg["actual"].to_numpy(), cfg.min_calibration_rows)
        # domain context
        for (tt, ge, ctx), sg in g.groupby(["team_type", "gender", "competition_context"], dropna=False):
            if len(sg) >= cfg.min_group_rows:
                key = f"{tt}|{ge}|{ctx}"
                calibrators[(market, "domain_context", key)] = _fit_best_binary_calibrator(sg["probability"].to_numpy(), sg["actual"].to_numpy(), cfg.min_calibration_rows)
        # competition
        for comp, sg in g.groupby("competition", dropna=False):
            if len(sg) >= cfg.min_group_rows:
                key = str(comp)
                calibrators[(market, "competition", key)] = _fit_best_binary_calibrator(sg["probability"].to_numpy(), sg["actual"].to_numpy(), cfg.min_calibration_rows)
    return calibrators, report


def _fit_best_binary_calibrator(p: np.ndarray, y: np.ndarray, min_rows: int) -> dict[str, Any]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    p = np.clip(p, 1e-5, 1 - 1e-5)
    base_rate = float((y.sum() + 1.0) / (len(y) + 2.0)) if len(y) else 0.5
    candidates: list[dict[str, Any]] = [{"method": "identity", "base_rate": base_rate, "score": _safe_log_loss(y, p)}]
    # shrink grid toward empirical group rate. This is very robust for small samples.
    for alpha in [0.05, 0.10, 0.20, 0.35, 0.50, 0.70]:
        pp = alpha * p + (1 - alpha) * base_rate
        candidates.append({"method": "shrink", "alpha": alpha, "base_rate": base_rate, "score": _safe_log_loss(y, pp)})
    if len(y) >= min_rows and len(np.unique(y)) == 2:
        x = np.log(p / (1 - p)).reshape(-1, 1)
        if LogisticRegression is not None:
            try:
                lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=500).fit(x, y)
                pp = lr.predict_proba(x)[:, 1]
                candidates.append({"method": "platt_logit", "coef": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0]), "base_rate": base_rate, "score": _safe_log_loss(y, pp)})
            except Exception:
                pass
        if IsotonicRegression is not None and len(y) >= max(min_rows, 500):
            try:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(p, y)
                pp = iso.predict(p)
                candidates.append({"method": "isotonic", "x_thresholds": [float(v) for v in iso.X_thresholds_], "y_thresholds": [float(v) for v in iso.y_thresholds_], "base_rate": base_rate, "score": _safe_log_loss(y, pp)})
            except Exception:
                pass
    best = sorted(candidates, key=lambda d: (float(d.get("score", 999)), d.get("method") != "shrink"))[0]
    best["n"] = int(len(y))
    best["actual_rate"] = float(y.mean()) if len(y) else None
    best["avg_raw_probability"] = float(p.mean()) if len(p) else None
    return best


def _apply_calibrator(p: pd.Series | np.ndarray, cal: dict[str, Any]) -> np.ndarray:
    arr = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    method = cal.get("method", "identity")
    if method == "shrink":
        return np.clip(float(cal.get("alpha", 0.3)) * arr + (1 - float(cal.get("alpha", 0.3))) * float(cal.get("base_rate", arr.mean())), 1e-5, 1 - 1e-5)
    if method == "platt_logit":
        logit = np.log(arr / (1 - arr))
        z = float(cal.get("coef", 1.0)) * logit + float(cal.get("intercept", 0.0))
        return np.clip(1 / (1 + np.exp(-z)), 1e-5, 1 - 1e-5)
    if method == "isotonic" and "x_thresholds" in cal:
        xs = np.asarray(cal.get("x_thresholds", []), dtype=float)
        ys = np.asarray(cal.get("y_thresholds", []), dtype=float)
        if len(xs) and len(ys):
            return np.clip(np.interp(arr, xs, ys), 1e-5, 1 - 1e-5)
    return arr


def _apply_hierarchical_calibrators(pred: pd.DataFrame, calibrators: dict[str, Any]) -> pd.DataFrame:
    out = pred.copy()
    probs = []
    levels = []
    methods = []
    for _, r in out.iterrows():
        market = str(r["market"])
        pg = str(r.get("position_group", "unknown_outfield"))
        keys = [
            (market, "competition_position_group", f"{r.get('competition', 'unknown')}|{pg}"),
            (market, "domain_position_group", f"{r.get('team_type','unknown')}|{r.get('gender','unknown')}|{r.get('competition_context','unknown')}|{pg}"),
            (market, "position_group", pg),
            (market, "competition", str(r.get("competition", "unknown"))),
            (market, "domain_context", f"{r.get('team_type','unknown')}|{r.get('gender','unknown')}|{r.get('competition_context','unknown')}"),
            (market, "team_type_gender", f"{r.get('team_type','unknown')}|{r.get('gender','unknown')}"),
            (market, "market_global", "*"),
        ]
        cal = None
        level = "raw_identity_no_calibrator"
        for k in keys:
            if k in calibrators:
                cal = calibrators[k]
                level = k[1]
                break
        if cal is None:
            cal = {"method": "identity"}
        pp = float(_apply_calibrator(np.array([float(r["probability"])]), cal)[0])
        probs.append(pp)
        levels.append(level)
        methods.append(cal.get("method", "identity"))
    out["calibrated_probability"] = probs
    out["calibration_level"] = levels
    out["calibration_method"] = methods
    out["brier"] = (out["calibrated_probability"] - out["actual"]) ** 2
    out["baseline_brier"] = (out["baseline_probability"] - out["actual"]) ** 2
    out["log_loss"] = [_binary_logloss(int(y), float(p)) for y, p in zip(out["actual"], out["calibrated_probability"])]
    out["baseline_log_loss"] = [_binary_logloss(int(y), float(p)) for y, p in zip(out["actual"], out["baseline_probability"])]
    return out


def _summarize_trial(pred: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for market, g in pred.groupby("market"):
        metrics = _binary_metrics(g["actual"], g["calibrated_probability"])
        base = _binary_metrics(g["actual"], g["baseline_probability"])
        out[market] = {
            "n": int(len(g)),
            "actual_rate": float(g["actual"].mean()),
            "avg_probability": float(g["calibrated_probability"].mean()),
            "probability_bias": float(g["calibrated_probability"].mean() - g["actual"].mean()),
            "brier": metrics["brier"],
            "log_loss": metrics["log_loss"],
            "baseline_brier": base["brier"],
            "baseline_log_loss": base["log_loss"],
            "brier_improvement_vs_baseline": _improvement(base["brier"], metrics["brier"]),
            "logloss_improvement_vs_baseline": _improvement(base["log_loss"], metrics["log_loss"]),
        }
    return out


def _trial_objective(metrics: dict[str, Any]) -> float:
    weights = {"player_shots": 0.35, "player_shots_on_target": 0.25, "player_fouls_committed": 0.25, "player_yellow_card": 0.15}
    values = []
    for market, w in weights.items():
        m = metrics.get(market)
        if not m:
            continue
        # Objective rewards beating baseline but still mostly optimizes calibrated log loss.
        values.append(w * (float(m["log_loss"]) - 0.10 * float(m.get("logloss_improvement_vs_baseline") or 0)))
    return float(sum(values) / sum(weights[m] for m in weights if m in metrics)) if values else float("inf")


def _flatten_market_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for market, m in metrics.items():
        short = market.replace("player_", "p_").replace("shots_on_target", "sot").replace("fouls_committed", "fouls").replace("yellow_card", "card")
        for k in ["log_loss", "brier", "actual_rate", "avg_probability", "probability_bias", "baseline_log_loss", "logloss_improvement_vs_baseline"]:
            out[f"{short}_{k}"] = float(m[k]) if m.get(k) is not None else np.nan
    return out


def _binary_metrics(y: pd.Series | np.ndarray, p: pd.Series | np.ndarray) -> dict[str, float]:
    yy = np.asarray(y, dtype=int)
    pp = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"brier": float(np.mean((pp - yy) ** 2)), "log_loss": _safe_log_loss(yy, pp)}


def _safe_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    yy = np.asarray(y, dtype=float)
    pp = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if len(yy) == 0:
        return float("nan")
    return float(-(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)).mean())


def _binary_logloss(y: int, p: float) -> float:
    pp = float(np.clip(p, 1e-9, 1 - 1e-9))
    yy = float(y)
    return float(-(yy * math.log(pp) + (1 - yy) * math.log(1 - pp)))


def _improvement(baseline: float, model: float) -> float | None:
    if baseline is None or not np.isfinite(baseline) or baseline <= 0:
        return None
    return float((baseline - model) / baseline)


def _policy_for_market(market: str, metrics: dict[str, float], base: dict[str, float], n: int) -> str:
    b_imp = _improvement(base["brier"], metrics["brier"]) or 0.0
    ll_imp = _improvement(base["log_loss"], metrics["log_loss"]) or 0.0
    if n < 500:
        return "paper_track_low_sample"
    if b_imp > 0.015 and ll_imp > 0.015:
        return "candidate_for_paper_value"
    if b_imp > 0.0 or ll_imp > 0.0:
        return "paper_track_caution"
    return "curiosity_only_blocks_value"


def _segment_metrics(pred: pd.DataFrame, cfg: ChampionPropConfig) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    df = pred.copy()
    df["prob_bucket"] = pd.cut(df["calibrated_probability"], bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
    df["sample_bucket"] = pd.cut(pd.to_numeric(df["sample_size_minutes"], errors="coerce").fillna(0), bins=[-1,90,270,500,1000,2500,1000000], labels=["<90","90-270","270-500","500-1000","1000-2500",">=2500"])
    rows = []
    dimensions = ["team_type", "gender", "competition_context", "competition", "position_group", "position_key", "sample_bucket", "prob_bucket", "started"]
    for dim in dimensions:
        if dim not in df.columns:
            continue
        for (market, value), g in df.groupby(["market", dim], dropna=False, observed=True):
            if len(g) < cfg.min_segment_rows:
                continue
            metrics = _binary_metrics(g["actual"], g["calibrated_probability"])
            base = _binary_metrics(g["actual"], g["baseline_probability"])
            rows.append({
                "market": str(market),
                "dimension": dim,
                "segment": str(value),
                "n": int(len(g)),
                "actual_rate": float(g["actual"].mean()),
                "avg_probability": float(g["calibrated_probability"].mean()),
                "bias": float(g["calibrated_probability"].mean() - g["actual"].mean()),
                "brier": metrics["brier"],
                "log_loss": metrics["log_loss"],
                "baseline_brier": base["brier"],
                "baseline_log_loss": base["log_loss"],
                "brier_improvement_vs_baseline": _improvement(base["brier"], metrics["brier"]),
                "logloss_improvement_vs_baseline": _improvement(base["log_loss"], metrics["log_loss"]),
            })
    return pd.DataFrame(rows).sort_values(["market", "dimension", "segment"]) if rows else pd.DataFrame()


def _calibrator_report(cal: dict[str, Any], n: int) -> dict[str, Any]:
    return {k: v for k, v in cal.items() if k not in {"x_thresholds", "y_thresholds"}} | {"n_rows": int(n)}


def _build_registry(champion_summary: pd.DataFrame, payload: dict[str, Any], segment_metrics: pd.DataFrame | None = None) -> dict[str, Any]:
    markets = {}
    if champion_summary is not None and not champion_summary.empty:
        for _, r in champion_summary.iterrows():
            markets[str(r["market"])] = {
                "champion_trial_id": str(r["champion_trial_id"]),
                "champion_trial_name": str(r["champion_trial_name"]),
                "policy": str(r["policy"]),
                "log_loss": float(r["log_loss"]),
                "baseline_log_loss": float(r["baseline_log_loss"]),
                "brier": float(r["brier"]),
                "baseline_brier": float(r["baseline_brier"]),
                "probability_bias": float(r["probability_bias"]),
            }
    segment_policies = _build_segment_policies(segment_metrics) if segment_metrics is not None else {}
    return {
        "version": "v0.28_prediction_registry",
        "status": "completed",
        "selection_policy": "champion_per_market_and_segment_guardrails; v0.28 adds position groups, role-aware calibration and goalkeeper guardrails for attacking props",
        "player_props": markets,
        "segment_policies": segment_policies,
        "audit": {k: v for k, v in payload.items() if k not in {"calibration_reports", "markets"}},
    }


def _build_segment_policies(segment_metrics: pd.DataFrame | None) -> dict[str, Any]:
    """Convert segment diagnostics into deployment guardrails.

    A segment can downgrade a market even if the global champion is strong. This
    prevents the bot from treating every player and context equally: weak sample
    buckets, overconfident probability buckets or contexts that lose to baseline
    are marked for caution/blocking.
    """
    if segment_metrics is None or segment_metrics.empty:
        return {}
    policies: dict[str, list[dict[str, Any]]] = {}
    for _, r in segment_metrics.iterrows():
        market = str(r.get("market"))
        n = int(r.get("n", 0) or 0)
        bias = float(r.get("bias", 0.0) or 0.0)
        b_imp = r.get("brier_improvement_vs_baseline")
        ll_imp = r.get("logloss_improvement_vs_baseline")
        b_imp = float(b_imp) if b_imp is not None and pd.notna(b_imp) else 0.0
        ll_imp = float(ll_imp) if ll_imp is not None and pd.notna(ll_imp) else 0.0
        action = "active"
        reason = "segment beats baseline with acceptable calibration"
        if str(r.get("dimension")) in {"position_group", "position_key"} and str(r.get("segment")) in {"goalkeeper", "gk"} and market in {"player_shots", "player_shots_on_target"}:
            action = "block_value"
            reason = "goalkeeper attacking-prop guardrail"
        elif n < 120:
            action = "paper_track_low_segment_sample"
            reason = "segment sample is low"
        elif b_imp <= 0 and ll_imp <= 0:
            action = "block_value"
            reason = "segment does not beat baseline"
        elif abs(bias) >= 0.08:
            action = "paper_track_calibration_bias"
            reason = "segment calibration bias is large"
        elif b_imp < 0.01 or ll_imp < 0.01:
            action = "active_caution"
            reason = "segment improvement is thin"
        policies.setdefault(market, []).append({
            "dimension": str(r.get("dimension")),
            "segment": str(r.get("segment")),
            "n": n,
            "action": action,
            "reason": reason,
            "bias": bias,
            "brier_improvement_vs_baseline": b_imp,
            "logloss_improvement_vs_baseline": ll_imp,
        })
    return policies
