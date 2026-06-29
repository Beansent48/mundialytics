from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mundialytics.statistical_core.distributions import ScoreDistribution, probability_for_count_line, scoreline_distribution
from mundialytics.statistical_core.match_model import _team_match_goal_frame
from mundialytics.statistical_core.schemas import canonical_name, standardize_fixtures
from mundialytics.statistical_core.team_stats_model import EVENT_ALIASES, build_team_match_stat_frame


@dataclass(frozen=True)
class DynamicLineConfig:
    """Configuration for v0.33 dynamic line/evidence generation.

    The dynamic-line layer is a market board, not an automatic betting trigger.
    It converts current model outputs into one row per line/scope and attaches
    auditable, structured evidence columns for the final frontend.
    """

    recent_n: int = 10
    h2h_years: int = 5
    h2h_max_matches: int = 8
    similar_elo_years: int = 4
    similar_elo_range: float = 100.0
    similar_elo_max_matches: int = 12
    min_context_sample: int = 3
    min_strong_context_sample: int = 5
    max_player_rows_per_market: int = 60
    demo_odds_policy: str = "label_only"  # label_only | allow_value


GOAL_TOTAL_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]
TEAM_GOAL_LINES = [0.5, 1.5, 2.5]
TEAM_LINES = {
    "shots": [6.5, 8.5, 10.5, 12.5, 14.5],
    "shots_on_target": [1.5, 2.5, 3.5, 4.5],
    "fouls": [8.5, 10.5, 12.5, 14.5],
    "yellow_cards": [0.5, 1.5, 2.5, 3.5],
    "corners": [2.5, 3.5, 4.5, 5.5, 6.5],
}
MATCH_TOTAL_LINES = {
    "shots": [16.5, 20.5, 24.5, 28.5, 32.5],
    "shots_on_target": [4.5, 6.5, 8.5, 10.5],
    "fouls": [18.5, 22.5, 26.5, 30.5],
    "yellow_cards": [2.5, 3.5, 4.5, 5.5, 6.5],
    "corners": [7.5, 8.5, 9.5, 10.5, 11.5],
}
PLAYER_LINES = {
    "player_shots": [0.5, 1.5, 2.5],
    "player_shots_on_target": [0.5, 1.5],
    "player_fouls_committed": [0.5, 1.5, 2.5],
    "player_yellow_card": [0.5],
}
PLAYER_TO_EVENT = {
    "player_shots": "shots",
    "player_shots_on_target": "shots_on_target",
    "player_fouls_committed": "fouls_committed",
    "player_yellow_card": "yellow_cards",
}
DISPLAY_MARKET = {
    "goals": "goals",
    "shots": "shots",
    "shots_on_target": "shots_on_target",
    "fouls": "fouls",
    "yellow_cards": "yellow_cards",
    "corners": "corners",
    "player_shots": "player_shots",
    "player_shots_on_target": "player_shots_on_target",
    "player_fouls_committed": "player_fouls_committed",
    "player_yellow_card": "player_yellow_card",
}
PLAYER_EVENT_COLUMNS = ["shots", "shots_on_target", "fouls_committed", "yellow_cards"]
MARKET_ORDER = {
    "goals": 0,
    "shots": 1,
    "shots_on_target": 2,
    "fouls": 3,
    "yellow_cards": 4,
    "player_shots": 5,
    "player_shots_on_target": 6,
    "player_fouls_committed": 7,
    "player_yellow_card": 8,
    "corners": 9,
}
SCOPE_ORDER = {"match": 0, "team": 1, "player": 2}
SIGNAL_ORDER = {"high_model_signal": 0, "medium_model_signal": 1, "fair_or_thin_signal": 2, "low_model_signal": 3, "not_available": 4}




def _to_naive_utc_timestamp(value: Any) -> pd.Timestamp:
    """Return a timezone-naive UTC timestamp for safe pandas comparisons.

    Fixtures generated from live/today providers may be timezone-aware, while
    historical StatsBomb-derived rows are usually timezone-naive. Pandas raises
    on comparisons between both. Normalising both sides to tz-naive UTC keeps
    temporal filtering deterministic without changing the calendar ordering.
    """
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    # pd.Timestamp handles both scalar Timestamp and Python datetime inputs.
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _to_naive_utc_series(values: Any) -> pd.Series:
    """Convert a date-like Series to timezone-naive UTC datetimes."""
    ser = pd.to_datetime(values, errors="coerce", utc=True)
    return ser.dt.tz_convert("UTC").dt.tz_localize(None)

def build_dynamic_market_lines(
    fixtures: pd.DataFrame,
    match_predictions: pd.DataFrame,
    scoreline_distribution_df: pd.DataFrame,
    team_stats_predictions: pd.DataFrame,
    player_event_predictions: pd.DataFrame,
    historical_events: pd.DataFrame | None,
    odds: pd.DataFrame | None = None,
    config: DynamicLineConfig | None = None,
) -> pd.DataFrame:
    """Build a dynamic market board with one row per scope/line/side.

    Scopes:
    - match: both teams combined.
    - team: one side only.
    - player: one current eligible player.

    v0.30+ separates statistical signal (`signal_label`) from price value
    (`value_label`) so a high-probability line is not confused with a good bet
    unless a real bookmaker price is attached.
    """

    cfg = config or DynamicLineConfig()
    f = standardize_fixtures(fixtures)
    mp = match_predictions.copy() if match_predictions is not None else pd.DataFrame()
    scores = scoreline_distribution_df.copy() if scoreline_distribution_df is not None else pd.DataFrame()
    team_stats = team_stats_predictions.copy() if team_stats_predictions is not None else pd.DataFrame()
    player_props = player_event_predictions.copy() if player_event_predictions is not None else pd.DataFrame()
    historical = historical_events.copy() if historical_events is not None else pd.DataFrame()
    ctx = _HistoricalContext.from_events(historical, cfg)
    rows: list[dict[str, Any]] = []

    for _, fixture in f.iterrows():
        match_id = str(fixture.get("match_id"))
        pred_row = _first(mp[mp["match_id"].astype(str).eq(match_id)]) if not mp.empty and "match_id" in mp.columns else {}
        rows.extend(_goal_line_rows(fixture, pred_row, scores, ctx, cfg))
        rows.extend(_team_stat_line_rows(fixture, team_stats, ctx, cfg))
        rows.extend(_player_prop_line_rows(fixture, player_props, ctx, cfg))

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _attach_odds(out, odds, cfg)
    return _sort_market_board(out)


class _HistoricalContext:
    def __init__(self, team_events: pd.DataFrame, player_events: pd.DataFrame, rating_lookup: dict[str, float], cfg: DynamicLineConfig):
        self.team_events = team_events
        self.player_events = player_events
        self.rating_lookup = rating_lookup
        self.cfg = cfg

    @classmethod
    def from_events(cls, historical_events: pd.DataFrame, cfg: DynamicLineConfig) -> "_HistoricalContext":
        if historical_events is None or historical_events.empty:
            return cls(pd.DataFrame(), pd.DataFrame(), {}, cfg)
        goals = _team_match_goal_frame(historical_events)
        stats = build_team_match_stat_frame(historical_events)
        frames = []
        if not goals.empty:
            g = goals.copy()
            g["team"] = g["team"].map(canonical_name)
            g["opponent"] = g["opponent"].map(canonical_name)
            g["goals"] = pd.to_numeric(g["goals_for"], errors="coerce").fillna(0.0)
            g["goals_against"] = pd.to_numeric(g["goals_against"], errors="coerce").fillna(0.0)
            g["total_goals"] = g["goals"] + g["goals_against"]
            keep = ["match_id", "date", "team", "opponent", "goals", "goals_against", "total_goals"]
            frames.append(g[keep])
        if not stats.empty:
            s = stats.copy()
            s["team"] = s["team"].map(canonical_name)
            s["opponent"] = s["opponent"].map(canonical_name)
            keep = ["match_id", "date", "team", "opponent"]
            for event in EVENT_ALIASES:
                if event in s.columns:
                    s[event] = pd.to_numeric(s[event], errors="coerce")
                    keep.append(event)
                against = f"{event}_against"
                if against in s.columns:
                    s[against] = pd.to_numeric(s[against], errors="coerce")
                    keep.append(against)
            frames.append(s[keep])
        if frames:
            base = frames[0]
            for other in frames[1:]:
                base = base.merge(other, on=["match_id", "date", "team", "opponent"], how="outer", suffixes=("", "_dup"))
                for col in list(base.columns):
                    if col.endswith("_dup"):
                        main = col[:-4]
                        if main in base.columns:
                            base[main] = base[main].combine_first(base[col])
                            base = base.drop(columns=[col])
                        else:
                            base = base.rename(columns={col: main})
            base["date"] = _to_naive_utc_series(base["date"])
        else:
            base = pd.DataFrame()
        rating_lookup = _simple_rating_lookup(base)
        if not base.empty:
            base["opponent_rating"] = base["opponent"].map(rating_lookup).astype(float)
        player_events = _build_player_context_frame(historical_events, rating_lookup)
        return cls(base, player_events, rating_lookup, cfg)

    def team_recent_values(self, team: str, market: str, fixture_date: Any, n: int | None = None) -> pd.Series:
        if self.team_events.empty or market not in self.team_events.columns:
            return pd.Series(dtype=float)
        team_key = canonical_name(team)
        date = _to_naive_utc_timestamp(fixture_date)
        work = self.team_events[self.team_events["team"].eq(team_key)].copy()
        if pd.notna(date):
            work = work[work["date"].isna() | (work["date"] < date)]
        work = work.sort_values("date").tail(n or self.cfg.recent_n)
        return pd.to_numeric(work[market], errors="coerce").dropna()

    def h2h_values(self, team: str, opponent: str, market: str, fixture_date: Any) -> pd.Series:
        if self.team_events.empty or market not in self.team_events.columns:
            return pd.Series(dtype=float)
        team_key = canonical_name(team)
        opp_key = canonical_name(opponent)
        date = _to_naive_utc_timestamp(fixture_date)
        work = self.team_events[(self.team_events["team"].eq(team_key)) & (self.team_events["opponent"].eq(opp_key))].copy()
        if pd.notna(date):
            min_date = date - pd.DateOffset(years=self.cfg.h2h_years)
            work = work[(work["date"].isna() | ((work["date"] < date) & (work["date"] >= min_date)))]
        work = work.sort_values("date").tail(self.cfg.h2h_max_matches)
        return pd.to_numeric(work[market], errors="coerce").dropna()

    def similar_elo_values(self, team: str, opponent: str, market: str, fixture_date: Any) -> pd.Series:
        if self.team_events.empty or market not in self.team_events.columns:
            return pd.Series(dtype=float)
        team_key = canonical_name(team)
        opp_key = canonical_name(opponent)
        target_rating = self.rating_lookup.get(opp_key)
        if target_rating is None or not math.isfinite(float(target_rating)):
            return pd.Series(dtype=float)
        date = _to_naive_utc_timestamp(fixture_date)
        work = self.team_events[self.team_events["team"].eq(team_key)].copy()
        work = work[pd.to_numeric(work.get("opponent_rating"), errors="coerce").sub(float(target_rating)).abs() <= self.cfg.similar_elo_range]
        if pd.notna(date):
            min_date = date - pd.DateOffset(years=self.cfg.similar_elo_years)
            work = work[(work["date"].isna() | ((work["date"] < date) & (work["date"] >= min_date)))]
        work = work.sort_values("date").tail(self.cfg.similar_elo_max_matches)
        return pd.to_numeric(work[market], errors="coerce").dropna()

    def player_recent_values(self, player: str, team: str, market: str, fixture_date: Any) -> pd.Series:
        values, _source = self.player_recent_values_with_source(player, team, market, fixture_date)
        return values

    def player_recent_values_with_source(self, player: str, team: str, market: str, fixture_date: Any) -> tuple[pd.Series, str]:
        work = self._player_base(player, market, fixture_date)
        if work.empty:
            return pd.Series(dtype=float), "not_available"
        team_key = canonical_name(team)
        source = "canonical_player_recent"
        if team_key:
            same_team = work[work["team"].eq(team_key)]
            # Prefer current-team/national-context history if available.
            if len(same_team) >= self.cfg.min_context_sample:
                work = same_team
                source = "current_team_recent"
        work_recent = work.sort_values("date").tail(self.cfg.recent_n)
        values = pd.to_numeric(work_recent[market], errors="coerce").dropna()
        if len(values) > 0:
            return values, source
        # Final fallback: any canonical player history with the market present.
        values = pd.to_numeric(work[market], errors="coerce").dropna()
        if len(values) > 0:
            return values.tail(self.cfg.recent_n), "canonical_player_historical"
        return pd.Series(dtype=float), "not_available"

    def player_h2h_values(self, player: str, team: str, opponent: str, market: str, fixture_date: Any) -> pd.Series:
        work = self._player_base(player, market, fixture_date)
        if work.empty:
            return pd.Series(dtype=float)
        opp_key = canonical_name(opponent)
        date = _to_naive_utc_timestamp(fixture_date)
        work = work[work["opponent"].eq(opp_key)]
        if pd.notna(date):
            min_date = date - pd.DateOffset(years=self.cfg.h2h_years)
            work = work[(work["date"].isna() | ((work["date"] < date) & (work["date"] >= min_date)))]
        work = work.sort_values("date").tail(self.cfg.h2h_max_matches)
        return pd.to_numeric(work[market], errors="coerce").dropna()

    def player_similar_elo_values(self, player: str, team: str, opponent: str, market: str, fixture_date: Any) -> pd.Series:
        values, _source = self.player_similar_elo_values_with_source(player, team, opponent, market, fixture_date)
        return values

    def player_h2h_values_with_source(self, player: str, team: str, opponent: str, market: str, fixture_date: Any) -> tuple[pd.Series, str]:
        values = self.player_h2h_values(player, team, opponent, market, fixture_date)
        return values, "player_h2h_recent" if len(values) else "not_available"

    def player_similar_elo_values_with_source(self, player: str, team: str, opponent: str, market: str, fixture_date: Any) -> tuple[pd.Series, str]:
        work = self._player_base(player, market, fixture_date)
        if work.empty:
            return pd.Series(dtype=float), "not_available"
        opp_key = canonical_name(opponent)
        target_rating = self.rating_lookup.get(opp_key)
        if target_rating is None or not math.isfinite(float(target_rating)):
            return pd.Series(dtype=float), "not_available"
        date = _to_naive_utc_timestamp(fixture_date)
        work = work[pd.to_numeric(work.get("opponent_rating"), errors="coerce").sub(float(target_rating)).abs() <= self.cfg.similar_elo_range]
        if pd.notna(date):
            min_date = date - pd.DateOffset(years=self.cfg.similar_elo_years)
            work = work[(work["date"].isna() | ((work["date"] < date) & (work["date"] >= min_date)))]
        work = work.sort_values("date").tail(self.cfg.similar_elo_max_matches)
        values = pd.to_numeric(work[market], errors="coerce").dropna()
        return values, "player_similar_elo_recent" if len(values) else "not_available"

    def _player_base(self, player: str, market: str, fixture_date: Any) -> pd.DataFrame:
        if self.player_events.empty or market not in self.player_events.columns:
            return pd.DataFrame()
        player_key = canonical_name(player)
        date = _to_naive_utc_timestamp(fixture_date)
        work = self.player_events[self.player_events["player"].eq(player_key)].copy()
        if pd.notna(date):
            work = work[work["date"].isna() | (work["date"] < date)]
        return work


def _build_player_context_frame(historical_events: pd.DataFrame, rating_lookup: dict[str, float]) -> pd.DataFrame:
    if historical_events is None or historical_events.empty:
        return pd.DataFrame()
    required = {"match_id", "date", "team", "opponent", "player"}
    if not required.issubset(historical_events.columns):
        return pd.DataFrame()
    keep = ["match_id", "date", "team", "opponent", "player"]
    for optional in ["position", "team_type", "competition_context", "gender"]:
        if optional in historical_events.columns:
            keep.append(optional)
    for col in PLAYER_EVENT_COLUMNS:
        if col in historical_events.columns:
            keep.append(col)
    out = historical_events[keep].copy()
    out["team"] = out["team"].map(canonical_name)
    out["opponent"] = out["opponent"].map(canonical_name)
    out["player"] = out["player"].map(canonical_name)
    out["date"] = _to_naive_utc_series(out["date"])
    for col in PLAYER_EVENT_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["opponent_rating"] = out["opponent"].map(rating_lookup).astype(float)
    return out


def _goal_line_rows(fixture: pd.Series, pred_row: dict[str, Any], scores: pd.DataFrame, ctx: _HistoricalContext, cfg: DynamicLineConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    match_id = str(fixture.get("match_id"))
    home = canonical_name(fixture.get("home_team"))
    away = canonical_name(fixture.get("away_team"))
    date = fixture.get("date")
    lh = _num(pred_row.get("lambda_home"), np.nan)
    la = _num(pred_row.get("lambda_away"), np.nan)
    dist = _score_dist_for_match(match_id, scores, lh, la)
    expected_total = float(lh + la) if np.isfinite(lh) and np.isfinite(la) else np.nan
    for line in GOAL_TOTAL_LINES:
        for side in ("over", "under"):
            prob = dist.total_goals_probability(line, side) if dist is not None else np.nan
            rows.append(_line_row_base(fixture, "goals", "match", "both", "match_total", "", line, side, prob, expected_total, "available", ctx, "total_goals", home, away, date, cfg))
    for team, opponent, is_home, lam in [(home, away, True, lh), (away, home, False, la)]:
        for line in TEAM_GOAL_LINES:
            for side in ("over", "under"):
                prob = probability_for_count_line(lam, line, side) if np.isfinite(lam) else np.nan
                rows.append(_line_row_base(fixture, "goals", "team", "home" if is_home else "away", team, "", line, side, prob, lam, "available", ctx, "goals", team, opponent, date, cfg))
    return rows


def _team_stat_line_rows(fixture: pd.Series, team_stats: pd.DataFrame, ctx: _HistoricalContext, cfg: DynamicLineConfig) -> list[dict[str, Any]]:
    if team_stats is None or team_stats.empty:
        return []
    rows: list[dict[str, Any]] = []
    match_id = str(fixture.get("match_id"))
    home = canonical_name(fixture.get("home_team"))
    away = canonical_name(fixture.get("away_team"))
    date = fixture.get("date")
    stats = team_stats[team_stats["match_id"].astype(str).eq(match_id)].copy()
    for event, lines in TEAM_LINES.items():
        for team, opponent, side_name in [(home, away, "home"), (away, home, "away")]:
            row = _first(stats[(stats["team"].map(canonical_name).eq(team)) & (stats["market"].astype(str).eq(event))])
            availability = str(row.get("availability", "not_available")) if row else "not_available"
            expected = _num(row.get("expected_count"), np.nan) if row else np.nan
            for line in lines:
                for side in ("over", "under"):
                    prob = probability_for_count_line(expected, line, side) if availability == "available" and np.isfinite(expected) else np.nan
                    rows.append(_line_row_base(fixture, event, "team", side_name, team, "", line, side, prob, expected, availability, ctx, event, team, opponent, date, cfg))
        total_row = _first(stats[(stats["team"].astype(str).eq("match_total")) & (stats["market"].astype(str).eq(f"total_{event}"))])
        total_availability = str(total_row.get("availability", "available")) if total_row else "not_available"
        expected_total = _num(total_row.get("expected_count"), np.nan) if total_row else np.nan
        if event == "corners" and not np.isfinite(expected_total):
            total_availability = "not_available"
        for line in MATCH_TOTAL_LINES[event]:
            for side in ("over", "under"):
                prob = probability_for_count_line(expected_total, line, side) if total_availability == "available" and np.isfinite(expected_total) else np.nan
                rows.append(_line_row_base(fixture, event, "match", "both", "match_total", "", line, side, prob, expected_total, total_availability, ctx, f"total_{event}", home, away, date, cfg))
    return rows


def _player_prop_line_rows(fixture: pd.Series, player_props: pd.DataFrame, ctx: _HistoricalContext, cfg: DynamicLineConfig) -> list[dict[str, Any]]:
    if player_props is None or player_props.empty:
        return []
    rows: list[dict[str, Any]] = []
    match_id = str(fixture.get("match_id"))
    props = player_props[player_props["match_id"].astype(str).eq(match_id)].copy()
    if props.empty:
        return rows
    for market, lines in PLAYER_LINES.items():
        g = props[props["market"].astype(str).eq(market)].copy()
        if g.empty:
            continue
        g["_expected"] = pd.to_numeric(g.get("expected_count"), errors="coerce").fillna(0.0)
        g = g.sort_values("_expected", ascending=False).head(cfg.max_player_rows_per_market)
        event = PLAYER_TO_EVENT.get(market, market)
        for _, r in g.iterrows():
            team = canonical_name(r.get("team"))
            opponent = canonical_name(r.get("opponent"))
            player = canonical_name(r.get("player", ""))
            expected = _num(r.get("expected_count"), np.nan)
            availability = "available"
            warnings = str(r.get("warnings", ""))
            identity_status = str(r.get("identity_status", "matched"))
            match_level = str(r.get("identity_match_level", ""))
            sample_minutes = _num(r.get("sample_size_minutes"), 0.0)
            candidate_source = str(r.get("candidate_source", ""))
            candidate_policy = str(r.get("candidate_policy", ""))
            player_selection_confidence = str(r.get("player_selection_confidence", ""))
            availability_reason = ""
            if "role_guardrail_goalkeeper_attacking_prop_blocked" in warnings:
                availability = "not_available"
                availability_reason = "role_guardrail_blocked"
            elif identity_status != "matched" or match_level in {"unresolved", "ambiguous"}:
                availability = "not_available"
                availability_reason = "identity_unresolved"
            elif sample_minutes <= 0:
                availability = "not_available"
                availability_reason = "sample_size_zero"
            elif candidate_policy in {"excluded_identity_or_sample", "squad_excluded_low_confidence", "squad_excluded_rank_limit"}:
                availability = "not_available"
                availability_reason = candidate_policy
            for line in lines:
                for side in ("over", "under"):
                    row_availability = availability
                    row_reason = availability_reason
                    if row_availability == "available" and candidate_policy == "squad_low_confidence_basic_only":
                        if market == "player_shots_on_target":
                            row_availability = "not_available"
                            row_reason = "low_confidence_squad_sot_blocked"
                        elif float(line) > 0.5:
                            row_availability = "not_available"
                            row_reason = "low_confidence_squad_line_blocked"
                    prob = probability_for_count_line(expected, line, side) if row_availability == "available" and np.isfinite(expected) else np.nan
                    row = _line_row_base(
                        fixture,
                        market,
                        "player",
                        "player",
                        team,
                        player,
                        line,
                        side,
                        prob,
                        expected,
                        row_availability,
                        ctx,
                        event,
                        team,
                        opponent,
                        fixture.get("date"),
                        cfg,
                        player_for_context=player,
                    )
                    row.update({
                        "position_group": r.get("position_group", ""),
                        "position_key": r.get("position_key", ""),
                        "input_position": r.get("input_position", r.get("position", "")),
                        "resolved_position": r.get("resolved_position", r.get("position", "")),
                        "position_source": r.get("position_source", ""),
                        "player_input_source": r.get("player_input_source", r.get("candidate_source", "")),
                        "player_selection_confidence": r.get("player_selection_confidence", ""),
                        "candidate_rank_team": r.get("candidate_rank_team", ""),
                        "candidate_policy": r.get("candidate_policy", ""),
                        "candidate_reason": r.get("candidate_reason", ""),
                        "candidate_score": _num(r.get("candidate_score"), np.nan),
                        "expected_minutes": _num(r.get("expected_minutes"), np.nan),
                        "sample_size_minutes": _num(r.get("sample_size_minutes"), np.nan),
                        "calibration_policy": r.get("model_type", "player_prop_model"),
                        "segment_policy": r.get("confidence_flag", ""),
                    })
                    if row_availability != "available":
                        tag = row_reason or "player_prop_not_available"
                        row["evidence_tags"] = _append_tag(row.get("evidence_tags", ""), tag)
                        row["data_quality_flag"] = "not_available"
                        row["reason_code"] = tag
                        row["signal_reason_code"] = tag
                    elif candidate_source == "squads":
                        row["evidence_tags"] = _append_tag(row.get("evidence_tags", ""), "squad_fallback_unconfirmed")
                        if candidate_policy:
                            row["evidence_tags"] = _append_tag(row.get("evidence_tags", ""), candidate_policy)
                        if player_selection_confidence in {"low", "very_low"}:
                            row["evidence_tags"] = _append_tag(row.get("evidence_tags", ""), f"selection_confidence_{player_selection_confidence}")
                        if row.get("data_quality_flag") == "full_context_sample":
                            row["data_quality_flag"] = "squad_fallback_context"
                    rows.append(row)
    return rows


def _line_row_base(
    fixture: pd.Series,
    market: str,
    scope: str,
    side: str,
    team: str,
    player: str,
    line: float,
    over_under: str,
    probability: float,
    expected_stat: float,
    availability: str,
    ctx: _HistoricalContext,
    evidence_market: str,
    team_for_context: str,
    opponent_for_context: str,
    date: Any,
    cfg: DynamicLineConfig,
    player_for_context: str = "",
) -> dict[str, Any]:
    p = _clip_prob(probability)
    fair_odds = 1.0 / p if p and np.isfinite(p) and p > 0 else np.nan
    if availability != "available":
        recent_n, recent_d = 0, 0
        similar_n, similar_d = 0, 0
        h2h_n, h2h_d = 0, 0
        tags = ["market_not_available"]
        signal_label = "not_available"
        value_label = "not_available"
        data_quality = "not_available"
        reason_code = "market_data_unavailable"
        recent_source = "not_available"
        similar_source = "not_available"
        h2h_source = "not_available"
    else:
        if scope == "player" and player_for_context:
            recent_values, recent_source = _player_context_values_with_source(ctx, player_for_context, team_for_context, opponent_for_context, evidence_market, date, "recent")
            similar_values, similar_source = _player_context_values_with_source(ctx, player_for_context, team_for_context, opponent_for_context, evidence_market, date, "similar_elo")
            h2h_values, h2h_source = _player_context_values_with_source(ctx, player_for_context, team_for_context, opponent_for_context, evidence_market, date, "h2h")
        else:
            recent_values, recent_source = _context_values_with_source(ctx, scope, team_for_context, opponent_for_context, evidence_market, date, "recent")
            similar_values, similar_source = _context_values_with_source(ctx, scope, team_for_context, opponent_for_context, evidence_market, date, "similar_elo")
            h2h_values, h2h_source = _context_values_with_source(ctx, scope, team_for_context, opponent_for_context, evidence_market, date, "h2h")
        recent_n, recent_d = _hit_rate(recent_values, line, over_under)
        similar_n, similar_d = _hit_rate(similar_values, line, over_under)
        h2h_n, h2h_d = _hit_rate(h2h_values, line, over_under)
        tags = _evidence_tags(p, (recent_n, recent_d), (similar_n, similar_d), (h2h_n, h2h_d), cfg)
        signal_label = _signal_label(p, tags)
        value_label = "odds_not_available"
        data_quality = _data_quality((recent_d, similar_d, h2h_d), tags)
        reason_code = _reason_code(signal_label, tags)
    return {
        "match_id": str(fixture.get("match_id")),
        "date": fixture.get("date", "unknown"),
        "competition": fixture.get("competition", "unknown"),
        "competition_context": fixture.get("competition_context", fixture.get("stage", "unknown")),
        "team_type": fixture.get("team_type", "unknown"),
        "gender": fixture.get("gender", "unknown"),
        "home_team": canonical_name(fixture.get("home_team")),
        "away_team": canonical_name(fixture.get("away_team")),
        "market": DISPLAY_MARKET.get(market, market),
        "scope": scope,
        "side": side,
        "team": canonical_name(team) if team else "",
        "player": canonical_name(player) if player else "",
        "line": float(line),
        "over_under": over_under,
        "model_probability": p,
        "fair_odds": fair_odds,
        "book_odds": np.nan,
        "implied_probability": np.nan,
        "edge": np.nan,
        "ev": np.nan,
        "expected_stat": expected_stat,
        "recent_hit_rate_n": int(recent_n),
        "recent_hit_rate_d": int(recent_d),
        "recent_hit_rate": _rate_text(recent_n, recent_d),
        "recent_evidence_source": recent_source,
        "similar_elo_hit_rate_n": int(similar_n),
        "similar_elo_hit_rate_d": int(similar_d),
        "similar_elo_hit_rate": _rate_text(similar_n, similar_d),
        "similar_elo_evidence_source": similar_source,
        "h2h_recent_hit_rate_n": int(h2h_n),
        "h2h_recent_hit_rate_d": int(h2h_d),
        "h2h_recent_hit_rate": _rate_text(h2h_n, h2h_d),
        "h2h_recent_evidence_source": h2h_source,
        "time_window": f"recent_last_{cfg.recent_n};h2h_{cfg.h2h_years}y_max{cfg.h2h_max_matches};similar_elo_{cfg.similar_elo_years}y_±{int(cfg.similar_elo_range)}",
        "availability": availability,
        "data_quality_flag": data_quality,
        "calibration_policy": "model_probability_from_current_champion_or_poisson_count_line",
        "segment_policy": "pending_frontend_filter",
        "signal_label": signal_label,
        "value_label": value_label,
        "evidence_tags": ";".join(tags),
        "reason_code": reason_code,
        "signal_reason_code": reason_code,
        "value_reason_code": "odds_not_attached" if value_label == "odds_not_available" else reason_code,
    }


def _context_values_with_source(ctx: _HistoricalContext, scope: str, team: str, opponent: str, evidence_market: str, date: Any, mode: str) -> tuple[pd.Series, str]:
    values = _context_values(ctx, scope, team, opponent, evidence_market, date, mode)
    if len(values) == 0:
        return values, "not_available"
    if mode == "recent":
        return values, "match_recent" if scope == "match" else "team_recent"
    if mode == "similar_elo":
        return values, "match_similar_elo_recent" if scope == "match" else "team_similar_elo_recent"
    if mode == "h2h":
        return values, "match_h2h_recent" if scope == "match" else "team_h2h_recent"
    return values, "unknown"


def _player_context_values_with_source(ctx: _HistoricalContext, player: str, team: str, opponent: str, market: str, date: Any, mode: str) -> tuple[pd.Series, str]:
    if mode == "recent":
        return ctx.player_recent_values_with_source(player, team, market, date)
    if mode == "h2h":
        return ctx.player_h2h_values_with_source(player, team, opponent, market, date)
    if mode == "similar_elo":
        return ctx.player_similar_elo_values_with_source(player, team, opponent, market, date)
    return pd.Series(dtype=float), "not_available"


def _context_values(ctx: _HistoricalContext, scope: str, team: str, opponent: str, evidence_market: str, date: Any, mode: str) -> pd.Series:
    market = evidence_market
    if market.startswith("total_"):
        event = market.replace("total_", "")
        if event == "goals":
            market = "total_goals"
        elif event in EVENT_ALIASES:
            base_values = _team_context_values(ctx, team, opponent, event, date, mode)
            against_col = f"{event}_against"
            against_values = _team_context_values(ctx, team, opponent, against_col, date, mode) if against_col in ctx.team_events.columns else pd.Series(dtype=float)
            if len(base_values) and len(against_values) and len(base_values) == len(against_values):
                return (base_values.reset_index(drop=True) + against_values.reset_index(drop=True)).dropna()
            return base_values
    return _team_context_values(ctx, team, opponent, market, date, mode)


def _team_context_values(ctx: _HistoricalContext, team: str, opponent: str, market: str, date: Any, mode: str) -> pd.Series:
    if mode == "recent":
        return ctx.team_recent_values(team, market, date)
    if mode == "h2h":
        return ctx.h2h_values(team, opponent, market, date)
    if mode == "similar_elo":
        return ctx.similar_elo_values(team, opponent, market, date)
    return pd.Series(dtype=float)


def _player_context_values(ctx: _HistoricalContext, player: str, team: str, opponent: str, market: str, date: Any, mode: str) -> pd.Series:
    if mode == "recent":
        return ctx.player_recent_values(player, team, market, date)
    if mode == "h2h":
        return ctx.player_h2h_values(player, team, opponent, market, date)
    if mode == "similar_elo":
        return ctx.player_similar_elo_values(player, team, opponent, market, date)
    return pd.Series(dtype=float)


def _score_dist_for_match(match_id: str, scores: pd.DataFrame, lh: float, la: float):
    if scores is not None and not scores.empty and {"match_id", "home_goals", "away_goals", "probability"}.issubset(scores.columns):
        s = scores[scores["match_id"].astype(str).eq(match_id)].copy()
        if not s.empty:
            matrix = s.pivot_table(index="home_goals", columns="away_goals", values="probability", aggfunc="sum").fillna(0.0)
            matrix = matrix.sort_index().sort_index(axis=1)
            return ScoreDistribution(lambda_home=lh if np.isfinite(lh) else 1.2, lambda_away=la if np.isfinite(la) else 1.2, matrix=matrix)
    if np.isfinite(lh) and np.isfinite(la):
        return scoreline_distribution(lh, la, max_goals=10)
    return None


def _simple_rating_lookup(team_events: pd.DataFrame) -> dict[str, float]:
    if team_events.empty or "team" not in team_events.columns:
        return {}
    work = team_events.copy()
    if "goals" not in work.columns:
        return {canonical_name(t): 1500.0 for t in work["team"].dropna().unique()}
    work["goals"] = pd.to_numeric(work.get("goals"), errors="coerce").fillna(0.0)
    work["goals_against"] = pd.to_numeric(work.get("goals_against"), errors="coerce").fillna(work["goals"].mean())
    ratings: dict[str, float] = {}
    for team, g in work.groupby("team"):
        gf = float(g["goals"].mean())
        ga = float(g["goals_against"].mean())
        rating = float(np.clip(1500.0 + 85.0 * (gf - ga), 1200.0, 1850.0))
        ratings[canonical_name(team)] = rating
    return ratings


def _hit_rate(values: pd.Series, line: float, side: str) -> tuple[int, int]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    d = int(len(vals))
    if d == 0:
        return 0, 0
    if str(side).lower().startswith("u"):
        n = int((vals < float(line)).sum())
    else:
        n = int((vals > float(line)).sum())
    return n, d


def _evidence_tags(prob: float, recent: tuple[int, int], similar: tuple[int, int], h2h: tuple[int, int], cfg: DynamicLineConfig) -> list[str]:
    tags: list[str] = []
    if not np.isfinite(prob):
        return ["model_not_available"]
    if prob >= 0.68:
        tags.append("model_high")
    elif prob >= 0.58:
        tags.append("model_medium")
    elif prob <= 0.42:
        tags.append("model_low")
    else:
        tags.append("model_neutral")
    for prefix, pair in [("recent", recent), ("similar_elo", similar), ("h2h_recent", h2h)]:
        n, d = pair
        if d < cfg.min_context_sample:
            tags.append(f"{prefix}_not_enough_data")
            continue
        rate = n / max(d, 1)
        thin = d < cfg.min_strong_context_sample
        if thin and rate >= 0.65:
            tags.append(f"{prefix}_positive_thin_sample")
        elif thin and rate <= 0.35:
            tags.append(f"{prefix}_negative_thin_sample")
        elif thin:
            tags.append(f"{prefix}_mixed_thin_sample")
        elif rate >= 0.65:
            tags.append(f"{prefix}_strong")
        elif rate >= 0.45:
            tags.append(f"{prefix}_mixed")
        else:
            tags.append(f"{prefix}_weak")
    return tags


def _signal_label(prob: float, tags: list[str]) -> str:
    if not np.isfinite(prob):
        return "not_available"
    positives = sum(t.endswith("_strong") for t in tags) + (1 if "model_high" in tags else 0)
    negatives = sum(t.endswith("_weak") for t in tags) + (1 if "model_low" in tags else 0)
    thin_positive = sum(t.endswith("positive_thin_sample") for t in tags)
    if prob >= 0.68 and positives >= 2:
        return "high_model_signal"
    if prob >= 0.58 and (positives >= 1 or thin_positive >= 1):
        return "medium_model_signal"
    if negatives >= 2:
        return "low_model_signal"
    return "fair_or_thin_signal"


def _data_quality(samples: tuple[int, int, int], tags: list[str]) -> str:
    if "market_not_available" in tags or "model_not_available" in tags:
        return "not_available"
    if any("thin_sample" in t for t in tags):
        return "thin_context_sample"
    if any("not_enough_data" in t for t in tags):
        return "partial_context_sample"
    if max(samples) < 3:
        return "thin_context_sample"
    return "ok"


def _reason_code(signal_label: str, tags: list[str]) -> str:
    if signal_label == "not_available":
        return "market_data_unavailable"
    if signal_label == "high_model_signal":
        return "model_and_context_support"
    if signal_label == "medium_model_signal":
        return "model_support_with_some_context"
    if signal_label == "low_model_signal":
        return "model_or_context_weak"
    if any("not_enough_data" in t or "thin_sample" in t for t in tags):
        return "thin_context_use_model_only"
    return "mixed_evidence"


def _attach_odds(lines: pd.DataFrame, odds: pd.DataFrame | None, cfg: DynamicLineConfig | None = None) -> pd.DataFrame:
    out = lines.copy()
    cfg = cfg or DynamicLineConfig()
    if odds is None or odds.empty:
        return out
    req = {"match_id", "market", "line", "odds_decimal"}
    if not req.issubset(odds.columns):
        return out
    od = odds.copy()
    od["match_id"] = od["match_id"].astype(str)
    od["market_norm"] = od["market"].map(_normalize_odds_market)
    od["raw_market_norm"] = od["market"].astype(str).str.lower().str.strip()
    od["line_num"] = pd.to_numeric(od["line"].astype(str).str.replace("+", "", regex=False), errors="coerce")
    od["selection_norm"] = od["selection"].astype(str).str.lower() if "selection" in od.columns else ""
    out["line_num"] = pd.to_numeric(out["line"], errors="coerce")
    for idx, r in out.iterrows():
        if str(r.get("availability", "")) != "available":
            continue
        line_val = _num(r.get("line"), np.nan)
        if not np.isfinite(line_val):
            continue
        candidates = od[(od["match_id"].eq(str(r["match_id"]))) & (od["line_num"].sub(line_val).abs() < 1e-9)]
        if candidates.empty:
            continue
        m = str(r["market"]).lower()
        candidates = candidates[candidates["market_norm"].eq(m)]
        if candidates.empty:
            continue
        # Avoid attaching total/match odds to team or player scoped rows.
        scope = str(r.get("scope", "")).lower()
        raw = candidates["raw_market_norm"].astype(str)
        total_mask = raw.str.contains("total|match|both", regex=True, na=False)
        if total_mask.any():
            candidates = candidates[total_mask]
            if scope != "match":
                continue
        side = str(r["over_under"]).lower()
        side_candidates = candidates[candidates["selection_norm"].str.contains(side, na=False)]
        if side_candidates.empty:
            # Never attach an over price to an under row or vice versa.
            # If the odds feed does not expose selection text, leave this row unpriced.
            continue
        best = side_candidates.iloc[0]
        odds_decimal = _num(best.get("odds_decimal"), np.nan)
        if np.isfinite(odds_decimal) and odds_decimal > 1.0:
            p = _num(r.get("model_probability"), np.nan)
            implied = 1.0 / odds_decimal
            edge = p - implied if np.isfinite(p) else np.nan
            ev = p * odds_decimal - 1.0 if np.isfinite(p) else np.nan
            out.at[idx, "book_odds"] = odds_decimal
            out.at[idx, "implied_probability"] = implied
            out.at[idx, "edge"] = edge
            out.at[idx, "ev"] = ev
            is_demo = _is_demo_odds_row(best)
            if is_demo and cfg.demo_odds_policy != "allow_value":
                out.at[idx, "value_label"] = "demo_odds_only"
                out.at[idx, "value_reason_code"] = "demo_odds_detected_not_for_real_value"
                out.at[idx, "evidence_tags"] = _append_tag(out.at[idx, "evidence_tags"], "demo_odds_only")
            else:
                value_label, reason = _price_value_label(edge, ev)
                out.at[idx, "value_label"] = value_label
                out.at[idx, "value_reason_code"] = reason
    return out.drop(columns=["line_num"], errors="ignore")


def _is_demo_odds_row(row: pd.Series | dict[str, Any]) -> bool:
    try:
        bookmaker = str(row.get("bookmaker", "")).strip().lower()
        source = str(row.get("source", "")).strip().lower()
    except Exception:
        return False
    return bookmaker in {"demo_book", "demo", "sample", "example"} or "demo" in source


def _normalize_odds_market(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    aliases = {
        "total_goals": "goals",
        "match_goals": "goals",
        "goals_total": "goals",
        "over_under_goals": "goals",
        "total_shots": "shots",
        "match_shots": "shots",
        "total_shots_on_target": "shots_on_target",
        "match_shots_on_target": "shots_on_target",
        "sot": "shots_on_target",
        "total_fouls": "fouls",
        "match_fouls": "fouls",
        "total_yellow_cards": "yellow_cards",
        "yellow_card": "yellow_cards",
        "cards": "yellow_cards",
        "total_corners": "corners",
        "match_corners": "corners",
    }
    return aliases.get(text, text)


def _price_value_label(edge: float, ev: float) -> tuple[str, str]:
    if not np.isfinite(edge) or not np.isfinite(ev):
        return "odds_not_available", "odds_not_attached"
    if ev >= 0.08 and edge >= 0.04:
        return "high_value", "positive_ev_clear_edge"
    if ev >= 0.03 and edge >= 0.015:
        return "medium_value", "positive_ev_thin_edge"
    if ev >= -0.02:
        return "fair_price", "price_close_to_model"
    return "no_value", "negative_ev_or_overpriced"


def _sort_market_board(out: pd.DataFrame) -> pd.DataFrame:
    work = out.copy()
    work["_availability_rank"] = work["availability"].astype(str).ne("available").astype(int)
    work["_market_rank"] = work["market"].map(MARKET_ORDER).fillna(99).astype(int)
    work["_scope_rank"] = work["scope"].map(SCOPE_ORDER).fillna(99).astype(int)
    work["_signal_rank"] = work.get("signal_label", pd.Series(index=work.index, dtype=str)).map(SIGNAL_ORDER).fillna(99).astype(int)
    sort_cols = [c for c in ["match_id", "_availability_rank", "_market_rank", "_scope_rank", "team", "player", "line", "over_under", "_signal_rank"] if c in work.columns]
    work = work.sort_values(sort_cols).drop(columns=["_availability_rank", "_market_rank", "_scope_rank", "_signal_rank"], errors="ignore")
    return work.reset_index(drop=True)


def _clip_prob(value: float) -> float:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not math.isfinite(p):
        return np.nan
    return float(np.clip(p, 0.0, 1.0))


def _rate_text(n: int, d: int) -> str:
    return "not_available" if d <= 0 else f"{int(n)}/{int(d)}"


def _first(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    return df.iloc[0].to_dict()


def _num(value: Any, default: float = np.nan) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _append_tag(existing: str, tag: str) -> str:
    parts = [p for p in str(existing).split(";") if p]
    if tag not in parts:
        parts.append(tag)
    return ";".join(parts)
