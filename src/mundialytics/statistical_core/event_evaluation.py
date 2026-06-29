from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mundialytics.statistical_core.distributions import probability_for_count_line, parse_numeric_line
from mundialytics.statistical_core.player_event_model import PLAYER_MARKETS, POSITION_PRIORS, _position_key
from mundialytics.statistical_core.schemas import canonical_name, standardize_fixtures, write_json
from mundialytics.statistical_core.team_stats_model import EVENT_ALIASES, TeamStatsModel, build_team_match_stat_frame


TEAM_EVENT_LINES: dict[str, list[float]] = {
    "shots": [7.5, 9.5, 11.5, 13.5, 15.5],
    "shots_on_target": [2.5, 3.5, 4.5, 5.5],
    "fouls": [8.5, 10.5, 12.5, 14.5],
    "yellow_cards": [0.5, 1.5, 2.5, 3.5],
}

PLAYER_PROP_LINES: dict[str, list[str]] = {
    "player_shots": ["1+", "2+", "3+"],
    "player_shots_on_target": ["1+", "2+"],
    "player_fouls_committed": ["1+", "2+"],
    "player_yellow_card": ["1+"],
}


@dataclass(frozen=True)
class EventEvaluationConfig:
    test_fraction: float = 0.25
    min_train_matches: int = 50
    max_test_matches: int | None = None
    calibration_bins: int = 10
    team_model_config: dict[str, Any] = field(default_factory=dict)
    player_model_config: dict[str, Any] = field(default_factory=dict)
    evaluate_player_props: bool = True


def evaluate_event_models_temporal(
    historical_events: pd.DataFrame,
    cfg: EventEvaluationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Temporal holdout evaluation for team-count and player-prop models.

    This evaluates the event models conditional on known match participants. For
    player props, the historical test lineup is treated as known pre-match, but
    the actual minutes are not passed to the model: expected_minutes is imputed
    from the started flag (75 for starters, 35 for non-starters). Actual minutes
    are only used as the realised target context in the output audit.
    """

    cfg = cfg or EventEvaluationConfig()
    df = _prepare_events(historical_events)
    team_match = build_team_match_stat_frame(df)
    match_frame = _fixtures_from_team_match(team_match)
    cutoff, train_matches, test_matches = _temporal_split(match_frame, cfg.test_fraction, cfg.min_train_matches)
    if cutoff is None:
        summary = {
            "status": "not_enough_matches_for_event_evaluation",
            "matches_available": int(len(match_frame)),
            "min_train_matches": int(cfg.min_train_matches),
        }
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), summary
    if cfg.max_test_matches is not None and int(cfg.max_test_matches) > 0 and len(test_matches) > int(cfg.max_test_matches):
        test_matches = test_matches.tail(int(cfg.max_test_matches)).copy()
    train_ids = set(train_matches["match_id"].astype(str))
    test_ids = set(test_matches["match_id"].astype(str))
    train_events = df[df["match_id"].astype(str).isin(train_ids)].copy()
    test_events = df[df["match_id"].astype(str).isin(test_ids)].copy()
    fixtures = test_matches[["match_id", "date", "home_team", "away_team", "neutral", "competition", "stage", "team_scope", "team_type", "competition_context", "gender"]].copy()

    team_model = TeamStatsModel(**dict(cfg.team_model_config)).fit(train_events)
    team_predictions = team_model.predict_fixtures(fixtures)
    team_scored, team_line_scored = score_team_event_predictions(team_predictions, team_match[team_match["match_id"].astype(str).isin(test_ids)].copy(), train_events)

    if cfg.evaluate_player_props:
        lineups = _lineups_from_test_events(test_events, fixtures)
        player_predictions, player_audit = _fast_player_event_predictions(train_events, lineups, team_predictions, cfg.player_model_config)
        player_warnings = []
        player_scored, player_line_scored = score_player_event_predictions(player_predictions, test_events, train_events)
    else:
        player_predictions = pd.DataFrame()
        player_scored = pd.DataFrame()
        player_line_scored = pd.DataFrame()
        player_warnings = []
        player_audit = {}

    team_summary = summarize_team_event_performance(team_scored, team_line_scored)
    player_summary = summarize_player_event_performance(player_scored, player_line_scored)
    market_policy = build_market_policy(team_summary, player_summary)
    summary = {
        "status": "completed",
        "version": "v0.24_market_specific_event_evaluation",
        "cutoff_date": str(pd.Timestamp(cutoff).date()),
        "train_matches": int(len(train_matches)),
        "test_matches": int(len(test_matches)),
        "train_event_rows": int(len(train_events)),
        "test_event_rows": int(len(test_events)),
        "team_model_audit": team_model.audit,
        "player_model_audit": player_audit,
        "warnings": list(dict.fromkeys([str(w) for w in player_warnings if w])),
        "team_event_performance": team_summary,
        "player_prop_performance": player_summary,
        "market_policy": market_policy,
        "honest_limitations": [
            "Player-prop evaluation is lineup-known: actual participants are known, but actual minutes are not fed into the model.",
            "This is still a single chronological holdout. Rolling-origin by market should be the next validation layer.",
            "A market can look decent on Brier/log loss but still be unprofitable without real odds and closing-line tracking.",
        ],
    }
    return team_scored, team_line_scored, player_scored, player_line_scored, summary


def _prepare_events(historical_events: pd.DataFrame) -> pd.DataFrame:
    if historical_events is None or historical_events.empty:
        return pd.DataFrame()
    df = historical_events.copy()
    df["match_id"] = df["match_id"].astype(str)
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    for col in ["team", "opponent", "player"]:
        if col in df.columns:
            df[col] = df[col].map(canonical_name)
    for col in ["minutes", "shots", "shots_on_target", "fouls_committed", "yellow_cards", "goals"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0)
    return df


def _fixtures_from_team_match(team_match: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if team_match is None or team_match.empty:
        return pd.DataFrame()
    for match_id, g in team_match.groupby("match_id", dropna=False):
        g = g.dropna(subset=["team"]).copy()
        g = g.sort_values(["team", "opponent"]).drop_duplicates(subset=["team"], keep="first")
        if len(g) < 2:
            continue
        teams = sorted(g["team"].astype(str).unique().tolist())[:2]
        a = g[g["team"].astype(str).eq(teams[0])].iloc[0]
        rows.append(
            {
                "match_id": str(match_id),
                "date": pd.to_datetime(a.get("date"), errors="coerce"),
                "home_team": canonical_name(teams[0]),
                "away_team": canonical_name(teams[1]),
                "neutral": 1,
                "competition": "historical_event_eval",
                "stage": "historical_event_eval",
                "team_scope": "unknown",
                "team_type": "unknown",
                "competition_context": "unknown",
                "gender": "unknown",
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.dropna(subset=["date"]).sort_values(["date", "match_id"]).reset_index(drop=True)
    return out


def _temporal_split(matches: pd.DataFrame, test_fraction: float, min_train_matches: int) -> tuple[pd.Timestamp | None, pd.DataFrame, pd.DataFrame]:
    if matches is None or matches.empty:
        return None, pd.DataFrame(), pd.DataFrame()
    m = matches.dropna(subset=["date"]).sort_values(["date", "match_id"]).reset_index(drop=True)
    if len(m) <= max(2, int(min_train_matches)):
        return None, pd.DataFrame(), pd.DataFrame()
    test_n = max(1, int(round(len(m) * float(test_fraction))))
    split_idx = max(int(min_train_matches), len(m) - test_n)
    if split_idx >= len(m):
        split_idx = len(m) - 1
    cutoff = pd.Timestamp(m.iloc[split_idx]["date"])
    return cutoff, m.iloc[:split_idx].copy(), m.iloc[split_idx:].copy()


def score_team_event_predictions(team_predictions: pd.DataFrame, test_team_match: pd.DataFrame, train_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if team_predictions is None or team_predictions.empty or test_team_match is None or test_team_match.empty:
        return pd.DataFrame(), pd.DataFrame()
    actual_rows: list[dict[str, Any]] = []
    for event in EVENT_ALIASES:
        if event not in test_team_match.columns:
            continue
        tmp = test_team_match[["match_id", "team", event]].copy()
        tmp["market"] = event
        tmp = tmp.rename(columns={event: "actual_count"})
        actual_rows.append(tmp)
    actual = pd.concat(actual_rows, ignore_index=True) if actual_rows else pd.DataFrame()
    pred = team_predictions[~team_predictions["market"].astype(str).str.startswith("total_")].copy()
    scored = pred.merge(actual, on=["match_id", "team", "market"], how="inner")
    if scored.empty:
        return scored, pd.DataFrame()
    scored["expected_count"] = pd.to_numeric(scored["expected_count"], errors="coerce").fillna(0.0).clip(lower=1e-9)
    scored["actual_count"] = pd.to_numeric(scored["actual_count"], errors="coerce").fillna(0.0).clip(lower=0)
    train_team = build_team_match_stat_frame(train_events)
    baselines = {event: float(pd.to_numeric(train_team.get(event, pd.Series(dtype=float)), errors="coerce").mean()) for event in EVENT_ALIASES}
    scored["baseline_expected_count"] = scored["market"].map(lambda x: baselines.get(str(x), float(scored["actual_count"].mean())))
    scored["abs_error"] = (scored["expected_count"] - scored["actual_count"]).abs()
    scored["sq_error"] = (scored["expected_count"] - scored["actual_count"]) ** 2
    scored["poisson_nll"] = [_poisson_nll(float(y), float(lam)) for y, lam in zip(scored["actual_count"], scored["expected_count"])]
    scored["baseline_abs_error"] = (scored["baseline_expected_count"] - scored["actual_count"]).abs()
    scored["baseline_sq_error"] = (scored["baseline_expected_count"] - scored["actual_count"]) ** 2
    scored["baseline_poisson_nll"] = [_poisson_nll(float(y), float(lam)) for y, lam in zip(scored["actual_count"], scored["baseline_expected_count"])]

    line_rows: list[dict[str, Any]] = []
    for _, r in scored.iterrows():
        market = str(r["market"])
        for line in TEAM_EVENT_LINES.get(market, []):
            p = probability_for_count_line(float(r["expected_count"]), line, "over")
            base_p = probability_for_count_line(float(r["baseline_expected_count"]), line, "over")
            actual = int(float(r["actual_count"]) > float(line))
            line_rows.append(
                {
                    "scope": "team",
                    "match_id": r["match_id"],
                    "team": r["team"],
                    "market": market,
                    "line": line,
                    "model_probability": p,
                    "baseline_probability": base_p,
                    "actual": actual,
                    "brier": (p - actual) ** 2,
                    "baseline_brier": (base_p - actual) ** 2,
                    "log_loss": _binary_logloss(actual, p),
                    "baseline_log_loss": _binary_logloss(actual, base_p),
                    "expected_count": float(r["expected_count"]),
                    "actual_count": float(r["actual_count"]),
                }
            )
    return scored, pd.DataFrame(line_rows)


def _lineups_from_test_events(test_events: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    if test_events is None or test_events.empty:
        return pd.DataFrame()
    cols = [c for c in ["match_id", "date", "team", "opponent", "player", "position", "started", "minutes"] if c in test_events.columns]
    lineups = test_events[cols].copy()
    lineups = lineups[pd.to_numeric(lineups.get("minutes", 0), errors="coerce").fillna(0) > 0].copy()
    lineups["team"] = lineups["team"].map(canonical_name)
    lineups["player"] = lineups["player"].map(canonical_name)
    if "started" not in lineups.columns:
        lineups["started"] = np.where(pd.to_numeric(lineups.get("minutes", 0), errors="coerce").fillna(0) >= 60, 1, 0)
    lineups["started"] = pd.to_numeric(lineups["started"], errors="coerce").fillna(0).astype(int)
    lineups["expected_minutes"] = np.where(lineups["started"] == 1, 75.0, 35.0)
    if "position" not in lineups.columns:
        lineups["position"] = "UNK"
    keep = ["match_id", "team", "player", "position", "started", "expected_minutes"]
    return lineups[keep].drop_duplicates(["match_id", "team", "player"]).reset_index(drop=True)


def _fast_player_event_predictions(train_events: pd.DataFrame, lineups: pd.DataFrame, team_predictions: pd.DataFrame, config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Vectorized holdout predictor for event evaluation.

    It mirrors the production PlayerEventModel formula but avoids the full
    identity resolver because historical evaluation lineups already use the
    canonical StatsBomb player names from the same dataset.
    """
    config = dict(config or {})
    share_weight = float(config.get("share_weight", 0.65))
    yellow_card_cap = float(config.get("yellow_card_cap", 0.75))
    share_cap = float(config.get("share_cap", 0.45))
    share_floor = float(config.get("share_floor", 0.002))
    if train_events is None or train_events.empty or lineups is None or lineups.empty:
        return pd.DataFrame(), {"player_event_fit": "no_train_or_lineup_rows", "model_config": config}
    train = train_events.copy()
    train["player"] = train["player"].map(canonical_name)
    train["team"] = train["team"].map(canonical_name)
    train["position_key"] = train.get("position", "UNK").map(_position_key) if "position" in train.columns else "unk"
    event_cols = {"shots": "shots", "shots_on_target": "shots_on_target", "fouls": "fouls_committed", "yellow_cards": "yellow_cards"}
    for event, col in event_cols.items():
        if col in train.columns:
            train[event] = pd.to_numeric(train[col], errors="coerce").fillna(0).clip(lower=0)
        else:
            train[event] = 0.0
    train["minutes"] = pd.to_numeric(train.get("minutes", 0), errors="coerce").fillna(0).clip(lower=0)
    agg = {"minutes": "sum", "position_key": "first"}
    for event in event_cols:
        agg[event] = "sum"
    profiles = train.groupby("player", dropna=False).agg(agg).reset_index()
    player_teams = (
        train[["player", "team"]]
        .drop_duplicates()
        .groupby("player", dropna=False)["team"]
        .agg(list)
        .reset_index()
        .rename(columns={"team": "historical_teams"})
    )
    profiles = profiles.merge(player_teams, on="player", how="left")
    team_totals = {event: train.groupby("team", dropna=False)[event].sum().to_dict() for event in event_cols}
    global_rates = {event: 90.0 * float(train[event].sum()) / max(float(train["minutes"].sum()), 1.0) for event in event_cols}
    teams_series = profiles["historical_teams"].apply(lambda v: v if isinstance(v, list) else [])
    for event in event_cols:
        profiles[f"{event}_rate_per90"] = 90.0 * pd.to_numeric(profiles[event], errors="coerce").fillna(0) / profiles["minutes"].clip(lower=1)
        denom = teams_series.apply(lambda teams: sum(float(team_totals[event].get(str(t), 0.0)) for t in teams)).clip(lower=1.0)
        profiles[f"{event}_share"] = (pd.to_numeric(profiles[event], errors="coerce").fillna(0) / denom).clip(lower=share_floor, upper=share_cap)
    prof_cols = ["player", "minutes", "position_key"] + [f"{e}_rate_per90" for e in event_cols] + [f"{e}_share" for e in event_cols]
    candidates = lineups.copy()
    candidates["player"] = candidates["player"].map(canonical_name)
    candidates["team"] = candidates["team"].map(canonical_name)
    candidates["position_key"] = candidates.get("position", "UNK").map(_position_key) if "position" in candidates.columns else "unk"
    candidates = candidates.merge(profiles[prof_cols], on="player", how="left", suffixes=("", "_profile"))
    candidates["sample_size_minutes"] = pd.to_numeric(candidates["minutes"], errors="coerce").fillna(0.0)
    team_lookup: dict[tuple[str, str, str], float] = {}
    if team_predictions is not None and not team_predictions.empty:
        for _, r in team_predictions.iterrows():
            if str(r.get("availability", "available")) != "available" or str(r.get("market", "")).startswith("total_"):
                continue
            val = pd.to_numeric(pd.Series([r.get("expected_count")]), errors="coerce").iloc[0]
            if np.isfinite(val):
                team_lookup[(str(r.get("match_id")), canonical_name(r.get("team")), str(r.get("market")))] = float(val)
    rows: list[dict[str, Any]] = []
    for _, c in candidates.iterrows():
        pos_key = str(c.get("position_key") or "unk")
        expected_minutes = float(c.get("expected_minutes", 75.0) or 75.0)
        sample_minutes = float(c.get("sample_size_minutes", 0.0) or 0.0)
        for market, meta in PLAYER_MARKETS.items():
            event = meta["event"]
            team_exp = team_lookup.get((str(c["match_id"]), canonical_name(c["team"]), event), {"shots": 10.0, "shots_on_target": 3.5, "fouls": 11.0, "yellow_cards": 1.8}.get(event, 1.0))
            rate = c.get(f"{event}_rate_per90")
            if pd.isna(rate):
                rate = global_rates.get(event, 0.2)
            share = c.get(f"{event}_share")
            if pd.isna(share):
                share = POSITION_PRIORS.get(event, {}).get(pos_key, 0.08)
            share = float(np.clip(share, share_floor, share_cap))
            share_component = float(team_exp) * share * expected_minutes / 75.0
            rate_component = float(rate) * expected_minutes / 90.0
            sw = float(np.clip(share_weight, 0.0, 1.0))
            expected_count = max(0.0, sw * share_component + (1.0 - sw) * rate_component)
            if event == "yellow_cards":
                expected_count = min(expected_count, yellow_card_cap)
            prob = probability_for_count_line(expected_count, meta["line"], "over")
            # Evaluation keeps the probability raw-but-safe. Production betting still applies additional caps.
            if market == "player_yellow_card":
                prob = float(np.clip(prob, 0.005, 0.45))
            elif market == "player_shots_on_target":
                prob = float(np.clip(prob, 0.01, 0.75))
            else:
                prob = float(np.clip(prob, 0.02, 0.95))
            confidence = "normal" if sample_minutes >= 270 else ("very_low_sample" if sample_minutes < 90 else "low_sample")
            rows.append({
                "match_id": str(c["match_id"]),
                "date": c.get("date", "unknown"),
                "competition": c.get("competition", "historical_event_eval"),
                "stage": c.get("stage", "historical_event_eval"),
                "team": canonical_name(c["team"]),
                "opponent": canonical_name(c.get("opponent", "")),
                "player": canonical_name(c["player"]),
                "position": c.get("position", "UNK"),
                "candidate_source": "historical_lineup_eval",
                "market": market,
                "line": meta["line"],
                "team_expected_event": float(team_exp),
                "player_share": float(share),
                "historical_rate_per90": float(rate),
                "expected_minutes": expected_minutes,
                "expected_count": float(expected_count),
                "raw_probability": float(prob),
                "safe_probability": float(prob),
                "sample_size_minutes": sample_minutes,
                "confidence_flag": confidence,
                "identity_match_level": "historical_exact_eval" if sample_minutes > 0 else "unresolved",
                "identity_status": "matched" if sample_minutes > 0 else "unresolved",
                "warnings": "" if sample_minutes > 0 else "sample_size_zero_no_player_pick",
                "model_type": "v024_fast_temporal_player_event_eval",
            })
    audit = {"player_event_fit": "fast_historical_player_events_for_evaluation", "players": int(len(profiles)), "events": list(event_cols), "model_config": {"share_weight": share_weight, "yellow_card_cap": yellow_card_cap, "share_cap": share_cap, "share_floor": share_floor}}
    return pd.DataFrame(rows), audit


def score_player_event_predictions(player_predictions: pd.DataFrame, test_events: pd.DataFrame, train_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if player_predictions is None or player_predictions.empty or test_events is None or test_events.empty:
        return pd.DataFrame(), pd.DataFrame()
    actual = test_events.copy()
    actual["team"] = actual["team"].map(canonical_name)
    actual["player"] = actual["player"].map(canonical_name)
    market_event = {market: meta["event"] for market, meta in PLAYER_MARKETS.items()}
    actual_rows: list[dict[str, Any]] = []
    for market, event in market_event.items():
        source_col = "fouls_committed" if event == "fouls" and "fouls_committed" in actual.columns else event
        if source_col not in actual.columns:
            continue
        tmp = actual[["match_id", "team", "player", "minutes", "started", source_col]].copy()
        tmp["market"] = market
        tmp = tmp.rename(columns={source_col: "actual_count", "minutes": "actual_minutes", "started": "actual_started"})
        actual_rows.append(tmp)
    actual_long = pd.concat(actual_rows, ignore_index=True) if actual_rows else pd.DataFrame()
    scored = player_predictions.merge(actual_long, on=["match_id", "team", "player", "market"], how="inner")
    if scored.empty:
        return scored, pd.DataFrame()
    scored["expected_count"] = pd.to_numeric(scored["expected_count"], errors="coerce").fillna(0.0).clip(lower=1e-9)
    scored["actual_count"] = pd.to_numeric(scored["actual_count"], errors="coerce").fillna(0.0).clip(lower=0)
    scored["actual_minutes"] = pd.to_numeric(scored.get("actual_minutes", 0), errors="coerce").fillna(0.0).clip(lower=0)
    scored["actual_over_line"] = (scored["actual_count"] >= 1).astype(int)
    scored["brier_1plus"] = (pd.to_numeric(scored["safe_probability"], errors="coerce").fillna(0.0) - scored["actual_over_line"]) ** 2
    scored["log_loss_1plus"] = [_binary_logloss(int(y), float(p)) for y, p in zip(scored["actual_over_line"], scored["safe_probability"])]
    scored["abs_error_count"] = (scored["expected_count"] - scored["actual_count"]).abs()
    scored["sq_error_count"] = (scored["expected_count"] - scored["actual_count"]) ** 2

    train_base = _player_train_base_rates(train_events)
    scored["baseline_probability"] = scored["market"].map(lambda m: train_base.get(str(m), 0.25))
    scored["baseline_brier_1plus"] = (scored["baseline_probability"] - scored["actual_over_line"]) ** 2
    scored["baseline_log_loss_1plus"] = [_binary_logloss(int(y), float(p)) for y, p in zip(scored["actual_over_line"], scored["baseline_probability"])]

    line_rows: list[dict[str, Any]] = []
    for _, r in scored.iterrows():
        market = str(r["market"])
        for line in PLAYER_PROP_LINES.get(market, []):
            p = probability_for_count_line(float(r["expected_count"]), line, "over")
            safe_p = float(np.clip(p, 0.001, 0.999))
            threshold = int(float(str(line).replace("+", ""))) if str(line).endswith("+") else math.floor(parse_numeric_line(line)) + 1
            actual = int(float(r["actual_count"]) >= threshold)
            line_rows.append(
                {
                    "scope": "player",
                    "match_id": r["match_id"],
                    "team": r["team"],
                    "player": r["player"],
                    "market": market,
                    "line": line,
                    "model_probability": safe_p,
                    "actual": actual,
                    "brier": (safe_p - actual) ** 2,
                    "log_loss": _binary_logloss(actual, safe_p),
                    "expected_count": float(r["expected_count"]),
                    "actual_count": float(r["actual_count"]),
                    "sample_size_minutes": float(r.get("sample_size_minutes", 0.0)),
                    "confidence_flag": r.get("confidence_flag", "unknown"),
                    "identity_match_level": r.get("identity_match_level", "unknown"),
                }
            )
    return scored, pd.DataFrame(line_rows)


def _player_train_base_rates(train_events: pd.DataFrame) -> dict[str, float]:
    base: dict[str, float] = {}
    if train_events is None or train_events.empty:
        return base
    mapping = {"player_shots": "shots", "player_shots_on_target": "shots_on_target", "player_fouls_committed": "fouls_committed", "player_yellow_card": "yellow_cards"}
    for market, col in mapping.items():
        if col in train_events.columns:
            y = (pd.to_numeric(train_events[col], errors="coerce").fillna(0) >= 1).astype(float)
            if len(y):
                # Laplace smoothing to avoid 0/1 baselines in sparse card markets.
                base[market] = float((y.sum() + 1.0) / (len(y) + 2.0))
    return base


def summarize_team_event_performance(scored: pd.DataFrame, line_scored: pd.DataFrame) -> dict[str, Any]:
    if scored is None or scored.empty:
        return {"status": "no_team_event_predictions_scored"}
    rows: dict[str, Any] = {}
    for market, g in scored.groupby("market"):
        n = len(g)
        mae = float(g["abs_error"].mean())
        rmse = float(math.sqrt(g["sq_error"].mean()))
        base_mae = float(g["baseline_abs_error"].mean())
        base_rmse = float(math.sqrt(g["baseline_sq_error"].mean()))
        nll = float(g["poisson_nll"].mean())
        base_nll = float(g["baseline_poisson_nll"].mean())
        rows[str(market)] = {
            "n": int(n),
            "mean_actual": float(g["actual_count"].mean()),
            "mean_predicted": float(g["expected_count"].mean()),
            "bias_pred_minus_actual": float((g["expected_count"] - g["actual_count"]).mean()),
            "mae": mae,
            "rmse": rmse,
            "poisson_nll": nll,
            "baseline_mae": base_mae,
            "baseline_rmse": base_rmse,
            "baseline_poisson_nll": base_nll,
            "mae_improvement_vs_baseline": float((base_mae - mae) / base_mae) if base_mae > 0 else None,
            "nll_improvement_vs_baseline": float((base_nll - nll) / base_nll) if base_nll > 0 else None,
        }
    if line_scored is not None and not line_scored.empty:
        line_summary: dict[str, Any] = {}
        for (market, line), g in line_scored.groupby(["market", "line"]):
            line_summary[f"{market}_over_{line}"] = {
                "n": int(len(g)),
                "actual_rate": float(g["actual"].mean()),
                "mean_probability": float(g["model_probability"].mean()),
                "brier": float(g["brier"].mean()),
                "log_loss": float(g["log_loss"].mean()),
                "baseline_brier": float(g["baseline_brier"].mean()),
                "baseline_log_loss": float(g["baseline_log_loss"].mean()),
            }
        return {"status": "completed", "count_metrics_by_market": rows, "line_probability_metrics": line_summary}
    return {"status": "completed", "count_metrics_by_market": rows, "line_probability_metrics": {}}


def summarize_player_event_performance(scored: pd.DataFrame, line_scored: pd.DataFrame) -> dict[str, Any]:
    if scored is None or scored.empty:
        return {"status": "no_player_event_predictions_scored"}
    rows: dict[str, Any] = {}
    for market, g in scored.groupby("market"):
        rows[str(market)] = {
            "n": int(len(g)),
            "actual_1plus_rate": float(g["actual_over_line"].mean()),
            "mean_probability_1plus": float(pd.to_numeric(g["safe_probability"], errors="coerce").mean()),
            "brier_1plus": float(g["brier_1plus"].mean()),
            "log_loss_1plus": float(g["log_loss_1plus"].mean()),
            "baseline_brier_1plus": float(g["baseline_brier_1plus"].mean()),
            "baseline_log_loss_1plus": float(g["baseline_log_loss_1plus"].mean()),
            "brier_improvement_vs_baseline": float((g["baseline_brier_1plus"].mean() - g["brier_1plus"].mean()) / g["baseline_brier_1plus"].mean()) if g["baseline_brier_1plus"].mean() > 0 else None,
            "logloss_improvement_vs_baseline": float((g["baseline_log_loss_1plus"].mean() - g["log_loss_1plus"].mean()) / g["baseline_log_loss_1plus"].mean()) if g["baseline_log_loss_1plus"].mean() > 0 else None,
            "mean_actual_count": float(g["actual_count"].mean()),
            "mean_expected_count": float(g["expected_count"].mean()),
            "mae_count": float(g["abs_error_count"].mean()),
            "normal_confidence_rows": int(g["confidence_flag"].astype(str).eq("normal").sum()) if "confidence_flag" in g.columns else None,
        }
    segments: dict[str, Any] = {}
    if "sample_size_minutes" in scored.columns:
        s = scored.copy()
        s["sample_bucket"] = pd.cut(pd.to_numeric(s["sample_size_minutes"], errors="coerce").fillna(0), bins=[-1, 90, 270, 500, 1000, 1000000], labels=["<90", "90-270", "270-500", "500-1000", ">=1000"])
        for (market, bucket), g in s.groupby(["market", "sample_bucket"], observed=True):
            segments[f"{market}|sample={bucket}"] = {"n": int(len(g)), "actual_rate": float(g["actual_over_line"].mean()), "mean_probability": float(g["safe_probability"].mean()), "brier": float(g["brier_1plus"].mean()), "log_loss": float(g["log_loss_1plus"].mean())}
    line_summary: dict[str, Any] = {}
    if line_scored is not None and not line_scored.empty:
        for (market, line), g in line_scored.groupby(["market", "line"]):
            line_summary[f"{market}_{line}"] = {"n": int(len(g)), "actual_rate": float(g["actual"].mean()), "mean_probability": float(g["model_probability"].mean()), "brier": float(g["brier"].mean()), "log_loss": float(g["log_loss"].mean())}
    return {"status": "completed", "prop_metrics_by_market": rows, "sample_segment_metrics": segments, "alternative_line_metrics": line_summary}


def build_market_policy(team_summary: dict[str, Any], player_summary: dict[str, Any]) -> dict[str, Any]:
    policy: dict[str, Any] = {"team_events": {}, "player_props": {}}
    for market, m in (team_summary.get("count_metrics_by_market") or {}).items():
        imp = m.get("mae_improvement_vs_baseline")
        nll_imp = m.get("nll_improvement_vs_baseline")
        if imp is not None and imp > 0.03 and (nll_imp is None or nll_imp > -0.02):
            status = "usable_with_caution"
        elif imp is not None and imp >= -0.02:
            status = "curiosity_only_needs_calibration"
        else:
            status = "not_recommended_for_value_yet"
        policy["team_events"][market] = {"status": status, "reason": f"MAE improvement vs baseline={imp}; Poisson NLL improvement={nll_imp}"}
    for market, m in (player_summary.get("prop_metrics_by_market") or {}).items():
        imp = m.get("brier_improvement_vs_baseline")
        ll_imp = m.get("logloss_improvement_vs_baseline")
        if imp is not None and imp > 0.03 and ll_imp is not None and ll_imp > 0.01:
            status = "usable_with_caution"
        elif imp is not None and imp >= -0.02:
            status = "curiosity_only_or_paper_track"
        else:
            status = "not_recommended_for_value_yet"
        policy["player_props"][market] = {"status": status, "reason": f"Brier improvement vs baseline={imp}; log-loss improvement={ll_imp}"}
    return policy


def build_event_evaluation_report(path: str | Path, summary: dict[str, Any], team_scored: pd.DataFrame, player_scored: pd.DataFrame) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    def esc(s: str) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>Mundialytics Event Evaluation v0.24</title>"]
    html.append("<style>body{font-family:Arial,sans-serif;margin:28px;color:#111} table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:22px} th,td{border:1px solid #ddd;padding:6px} th{background:#f4f4f4}.warn{background:#fff4d6;border:1px solid #d7aa28;padding:10px}</style></head><body>")
    html.append("<h1>Mundialytics Event Evaluation v0.24</h1>")
    html.append(f"<p>Status: <strong>{esc(summary.get('status'))}</strong> | Train matches: {summary.get('train_matches')} | Test matches: {summary.get('test_matches')}</p>")
    html.append("<h2>Market policy</h2><pre>" + esc(json.dumps(summary.get("market_policy", {}), indent=2, ensure_ascii=False)) + "</pre>")
    team_metrics = pd.DataFrame.from_dict(summary.get("team_event_performance", {}).get("count_metrics_by_market", {}), orient="index").reset_index().rename(columns={"index": "market"})
    if not team_metrics.empty:
        cols = [c for c in ["market", "n", "mean_actual", "mean_predicted", "bias_pred_minus_actual", "mae", "baseline_mae", "mae_improvement_vs_baseline", "poisson_nll", "baseline_poisson_nll", "nll_improvement_vs_baseline"] if c in team_metrics.columns]
        html.append("<h2>Team event count performance</h2>")
        html.append(team_metrics[cols].to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    player_metrics = pd.DataFrame.from_dict(summary.get("player_prop_performance", {}).get("prop_metrics_by_market", {}), orient="index").reset_index().rename(columns={"index": "market"})
    if not player_metrics.empty:
        cols = [c for c in ["market", "n", "actual_1plus_rate", "mean_probability_1plus", "brier_1plus", "baseline_brier_1plus", "brier_improvement_vs_baseline", "log_loss_1plus", "baseline_log_loss_1plus", "logloss_improvement_vs_baseline", "mae_count"] if c in player_metrics.columns]
        html.append("<h2>Player prop 1+ performance</h2>")
        html.append(player_metrics[cols].to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    if not player_scored.empty:
        worst = player_scored.sort_values("log_loss_1plus", ascending=False).head(30)
        cols = [c for c in ["match_id", "team", "player", "market", "safe_probability", "actual_over_line", "actual_count", "expected_count", "sample_size_minutes", "confidence_flag", "warnings"] if c in worst.columns]
        html.append("<h2>Worst player-prop misses</h2>")
        html.append(worst[cols].to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    html.append("<h2>Audit summary</h2><pre>" + esc(json.dumps({k: v for k, v in summary.items() if k not in {"team_event_performance", "player_prop_performance"}}, indent=2, ensure_ascii=False)) + "</pre>")
    html.append("</body></html>")
    out.write_text("\n".join(html), encoding="utf-8")
    return out


def write_event_evaluation_outputs(out_dir: str | Path, team_scored: pd.DataFrame, team_line_scored: pd.DataFrame, player_scored: pd.DataFrame, player_line_scored: pd.DataFrame, summary: dict[str, Any]) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    files = {
        "team_event_backtest_predictions.csv": team_scored,
        "team_event_line_probabilities.csv": team_line_scored,
        "player_event_backtest_predictions.csv": player_scored,
        "player_event_line_probabilities.csv": player_line_scored,
    }
    for name, frame in files.items():
        path = out / name
        frame.to_csv(path, index=False)
        paths[name] = str(path)
    summary_path = out / "event_evaluation_summary.json"
    write_json(summary_path, summary)
    paths["event_evaluation_summary.json"] = str(summary_path)
    report_path = build_event_evaluation_report(out / "event_evaluation_report.html", summary, team_scored, player_scored)
    paths["event_evaluation_report.html"] = str(report_path)
    return paths


def _poisson_nll(y: float, lam: float) -> float:
    lam = max(float(lam), 1e-9)
    y = max(float(y), 0.0)
    return float(lam - y * math.log(lam) + math.lgamma(y + 1.0))


def _binary_logloss(y: int | float, p: float) -> float:
    yy = float(y)
    pp = float(np.clip(p, 1e-9, 1 - 1e-9))
    return float(-(yy * math.log(pp) + (1.0 - yy) * math.log(1.0 - pp)))
