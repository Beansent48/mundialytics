from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from mundialytics.models.player_event_model import PlayerEventModel


@dataclass
class PlayerPropBacktestConfig:
    min_train_matches: int = 20
    test_matches: int | None = 200
    markets: tuple[str, ...] = (
        "player_shots",
        "player_shots_on_target",
        "player_fouls_committed",
        "player_yellow_card",
    )
    line: str = "1+"
    min_minutes_in_test: float = 5.0
    use_observed_test_minutes: bool = False  # False prevents leakage in pre-match backtests.


def _date_sorted_events(player_events: pd.DataFrame) -> pd.DataFrame:
    df = player_events.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["date", "match_id", "team", "player"])
    else:
        df = df.sort_values(["match_id", "team", "player"])
    return df.reset_index(drop=True)


def _event_col_for_market(market: str) -> str:
    event = PlayerEventModel.MARKET_TO_EVENT.get(market)
    if event is None:
        raise ValueError(f"Unsupported market: {market}")
    return event


def backtest_player_props(player_events: pd.DataFrame, config: PlayerPropBacktestConfig | None = None, feature_events: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Simple historical validation for player-event props.

    The split is match-based and temporal: fit rates on earlier matches and
    predict player rows in later matches. This is not a final trading model, but
    it verifies that events/minutes/markets are wired correctly and gives an
    initial calibration signal for markets such as 1+ shot, 1+ SOT, 1+ foul.
    """
    cfg = config or PlayerPropBacktestConfig()
    df = _date_sorted_events(player_events)
    required = {"match_id", "team", "player", "minutes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"player_events missing required columns: {sorted(missing)}")

    match_order = df[["match_id", "date"] if "date" in df.columns else ["match_id"]].drop_duplicates("match_id")
    if "date" in match_order.columns:
        match_order = match_order.sort_values(["date", "match_id"])
    else:
        match_order = match_order.sort_values("match_id")
    match_ids = match_order["match_id"].astype(str).tolist()
    if len(match_ids) <= cfg.min_train_matches:
        raise ValueError(f"Need more than {cfg.min_train_matches} matches for player-props backtest; got {len(match_ids)}.")

    test_ids = match_ids[cfg.min_train_matches:]
    if cfg.test_matches is not None:
        test_ids = test_ids[-cfg.test_matches:]
        first_test_index = match_ids.index(test_ids[0])
    else:
        first_test_index = cfg.min_train_matches
    train_ids = set(match_ids[:first_test_index])
    test_ids_set = set(test_ids)
    train = df[df["match_id"].astype(str).isin(train_ids)].copy()
    test = df[df["match_id"].astype(str).isin(test_ids_set)].copy()
    test = test[pd.to_numeric(test["minutes"], errors="coerce").fillna(0) >= cfg.min_minutes_in_test]

    model_train = train.copy()
    feature_training_report = {"used_feature_events": False}
    if feature_events is not None and not feature_events.empty:
        feat = _date_sorted_events(feature_events)
        if "date" in feat.columns and "date" in match_order.columns:
            first_test_date = pd.to_datetime(match_order.loc[match_order["match_id"].astype(str) == str(test_ids[0]), "date"], errors="coerce")
            cutoff = first_test_date.iloc[0] if len(first_test_date) else pd.NaT
            if pd.notna(cutoff):
                feat = feat[pd.to_datetime(feat["date"], errors="coerce") < cutoff].copy()
        else:
            feat = feat[feat["match_id"].astype(str).isin(train_ids)].copy()
        # De-duplicate rows already present in the target-domain training set.
        dedupe_cols = [c for c in ["match_id", "team", "player", "player_id_global"] if c in feat.columns and c in model_train.columns]
        if dedupe_cols:
            feat["__k"] = feat[dedupe_cols].astype(str).agg("|".join, axis=1)
            model_train["__k"] = model_train[dedupe_cols].astype(str).agg("|".join, axis=1)
            feat = feat[~feat["__k"].isin(set(model_train["__k"]))].drop(columns=["__k"])
            model_train = model_train.drop(columns=["__k"])
        model_train = pd.concat([model_train, feat], ignore_index=True, sort=False)
        feature_training_report = {
            "used_feature_events": True,
            "feature_rows_after_cutoff": int(len(feat)),
            "model_training_rows_total": int(len(model_train)),
            "model_training_matches_total": int(model_train["match_id"].astype(str).nunique()) if "match_id" in model_train.columns else None,
            "feature_max_date": str(pd.to_datetime(feat["date"], errors="coerce").max().date()) if "date" in feat.columns and pd.to_datetime(feat["date"], errors="coerce").notna().any() else None,
        }

    model = PlayerEventModel().fit(model_train)
    rows: list[dict] = []
    for _, r in test.iterrows():
        actual_minutes = float(pd.to_numeric(pd.Series([r.get("minutes", 0)]), errors="coerce").fillna(0).iloc[0])
        position = r.get("position") if pd.notna(r.get("position")) else None
        competition_context = r.get("competition_context") if pd.notna(r.get("competition_context")) else None
        team_type = r.get("team_type") if pd.notna(r.get("team_type")) else None
        started = r.get("started") if "started" in test.columns else None
        player_id = r.get("player_id_global") if "player_id_global" in test.columns else None
        if cfg.use_observed_test_minutes:
            expected_minutes = float(pd.to_numeric(pd.Series([r.get("minutes", 90)]), errors="coerce").fillna(90).iloc[0])
            expected_minutes_source = "observed_test_minutes_LEAKY_DIAGNOSTIC_ONLY"
        else:
            expected_minutes, expected_minutes_source = model.expected_minutes_for_player(
                str(r["player"]), position=position, started=started, player_id_global=player_id, competition_context=competition_context, team_type=team_type
            )
        for market in cfg.markets:
            event_col = _event_col_for_market(market)
            if event_col not in test.columns:
                continue
            pred = model.predict_market(
                str(r["player"]),
                market,
                cfg.line,
                expected_minutes=expected_minutes,
                team_context=None,
                position=position,
                player_id_global=player_id,
                competition_context=competition_context,
                team_type=team_type,
            )
            actual_count = float(pd.to_numeric(pd.Series([r.get(event_col, 0)]), errors="coerce").fillna(0).iloc[0])
            actual = int(actual_count >= PlayerEventModel.parse_line(cfg.line))
            sample_profile = model.player_sample_profile(str(r["player"]), player_id_global=player_id)
            cross_context_flag = bool(str(team_type) == "national_team" and sample_profile.get("club_minutes_sample", 0.0) > 0)
            row = {
                "match_id": r["match_id"],
                "date": r.get("date"),
                "competition": r.get("competition"),
                "season": r.get("season"),
                "stage": r.get("stage"),
                "team_scope": r.get("team_scope"),
                "team_type": r.get("team_type"),
                "competition_context": r.get("competition_context"),
                "gender": r.get("gender"),
                "source": r.get("source"),
                "team": r.get("team"),
                "opponent": r.get("opponent"),
                "team_id": r.get("team_id"),
                "opponent_id": r.get("opponent_id"),
                "player": r["player"],
                "player_id_global": r.get("player_id_global"),
                "player_context_id": r.get("player_context_id"),
                "position": position,
                "started": started,
                "market_type": market,
                "line": cfg.line,
                "probability": pred.probability,
                "raw_probability": pred.probability,
                "expected_count": pred.expected_count,
                "expected_minutes": pred.expected_minutes,
                "expected_minutes_source": expected_minutes_source,
                "actual_minutes": actual_minutes,
                "sample_size": pred.sample_size,
                "club_minutes_sample": sample_profile.get("club_minutes_sample", 0.0),
                "national_minutes_sample": sample_profile.get("national_minutes_sample", 0.0),
                "cross_context_feature_used": cross_context_flag,
                "actual_count": actual_count,
                "actual": actual,
            }
            rows.append(row)

    pred_df = pd.DataFrame(rows)
    summary: dict = {
        "n_train_matches": int(len(train_ids)),
        "n_test_matches": int(len(test_ids_set)),
        "n_predictions": int(len(pred_df)),
        "markets": {},
        "feature_training": feature_training_report,
    }
    if pred_df.empty:
        return pred_df, summary
    eps = 1e-6
    pred_df["probability"] = pred_df["probability"].clip(eps, 1 - eps)
    for market, g in pred_df.groupby("market_type"):
        y = g["actual"].astype(int).to_numpy()
        p = g["probability"].astype(float).to_numpy()
        if len(np.unique(y)) < 2:
            ll = None
        else:
            ll = float(log_loss(y, np.vstack([1 - p, p]).T, labels=[0, 1]))
        summary["markets"][market] = {
            "n": int(len(g)),
            "actual_rate": float(y.mean()) if len(y) else 0.0,
            "avg_probability": float(p.mean()) if len(p) else 0.0,
            "brier": float(brier_score_loss(y, p)) if len(y) else None,
            "log_loss": ll,
            "avg_sample_minutes": float(g["sample_size"].mean()) if "sample_size" in g else 0.0,
        }
    return pred_df, summary
