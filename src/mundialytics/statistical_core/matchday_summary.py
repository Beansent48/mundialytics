from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import math
import pandas as pd


MATCHDAY_SUMMARY_VERSION = "v0.48.2_matchday_summary_rankings"

MATCHDAY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "ranking_category",
    "rank",
    "match_id",
    "match",
    "home_team",
    "away_team",
    "metric_name",
    "metric_value",
    "secondary_metric_name",
    "secondary_metric_value",
    "market",
    "scope",
    "side",
    "line",
    "over_under",
    "statistical_label",
    "data_quality_flag",
    "evidence_tags",
    "short_structured_reason",
)


def build_matchday_summary(
    *,
    match_predictions: pd.DataFrame,
    scoreline_distribution: pd.DataFrame | None = None,
    dynamic_market_lines: pd.DataFrame | None = None,
    team_stats_predictions: pd.DataFrame | None = None,
    player_event_predictions: pd.DataFrame | None = None,
    audit: dict[str, Any] | None = None,
    top_n: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build simulator-first matchday rankings.

    The summary deliberately ranks statistical situations rather than betting
    picks. It uses only already-generated simulator outputs and does not alter
    prediction/model logic.
    """
    match_predictions = _ensure_frame(match_predictions)
    scoreline_distribution = _ensure_frame(scoreline_distribution)
    dynamic_market_lines = _ensure_frame(dynamic_market_lines)
    team_stats_predictions = _ensure_frame(team_stats_predictions)
    player_event_predictions = _ensure_frame(player_event_predictions)

    base = _build_match_base(
        match_predictions=match_predictions,
        scoreline_distribution=scoreline_distribution,
        dynamic_market_lines=dynamic_market_lines,
        team_stats_predictions=team_stats_predictions,
        player_event_predictions=player_event_predictions,
    )

    rows: list[dict[str, Any]] = []
    if not base.empty:
        rows.extend(
            _rank_match_metric(
                base,
                category="high_goal_expectation",
                metric_column="expected_total_goals",
                metric_name="expected_total_goals",
                ascending=False,
                top_n=top_n,
                reason_code="HIGH_EXPECTED_TOTAL_GOALS",
                secondary_metric_column="top_scorelines",
                secondary_metric_name="top_scorelines",
            )
        )
        rows.extend(
            _rank_match_metric(
                base,
                category="low_goal_environment",
                metric_column="under_2_5_probability",
                metric_name="under_2_5_probability",
                ascending=False,
                top_n=top_n,
                reason_code="LOW_GOAL_ENVIRONMENT",
                secondary_metric_column="expected_total_goals",
                secondary_metric_name="expected_total_goals",
            )
        )
        rows.extend(
            _rank_match_metric(
                base,
                category="most_balanced_matches",
                metric_column="balance_score",
                metric_name="balance_score",
                ascending=False,
                top_n=top_n,
                reason_code="OUTCOME_PROBABILITIES_BALANCED",
                secondary_metric_column="favorite_probability",
                secondary_metric_name="favorite_probability",
            )
        )
        non_draw_favorites = base[base["favorite_selection"].ne("Draw")].copy()
        favorite_frame = non_draw_favorites if not non_draw_favorites.empty else base
        rows.extend(
            _rank_match_metric(
                favorite_frame,
                category="strongest_favorites",
                metric_column="favorite_probability",
                metric_name="favorite_probability",
                ascending=False,
                top_n=top_n,
                reason_code="CLEAR_FAVORITE_OR_MOST_LIKELY_OUTCOME_BY_MODEL",
                secondary_metric_column="favorite_selection",
                secondary_metric_name="favorite_selection",
            )
        )
        rows.extend(
            _rank_match_metric(
                base,
                category="highest_uncertainty",
                metric_column="uncertainty_score",
                metric_name="uncertainty_score",
                ascending=False,
                top_n=top_n,
                reason_code="SCORELINE_DISTRIBUTION_DISPERSED",
                secondary_metric_column="top_scoreline_probability",
                secondary_metric_name="top_scoreline_probability",
            )
        )
        rows.extend(
            _rank_match_metric(
                base,
                category="btts_lean",
                metric_column="btts_probability",
                metric_name="btts_probability",
                ascending=False,
                top_n=top_n,
                reason_code="BOTH_TEAMS_TO_SCORE_MODEL_LEAN",
                secondary_metric_column="expected_total_goals",
                secondary_metric_name="expected_total_goals",
            )
        )

    rows.extend(_rank_dynamic_line_signals(dynamic_market_lines, match_predictions, top_n=top_n * 2))
    rows.extend(_rank_data_quality_watchlist(base, top_n=top_n))

    summary = pd.DataFrame(rows, columns=MATCHDAY_SUMMARY_COLUMNS)
    if not summary.empty:
        summary["rank"] = pd.to_numeric(summary["rank"], errors="coerce").astype("Int64")

    payload = _build_summary_payload(summary, base, audit)
    return summary, payload


def _ensure_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _build_match_base(
    *,
    match_predictions: pd.DataFrame,
    scoreline_distribution: pd.DataFrame,
    dynamic_market_lines: pd.DataFrame,
    team_stats_predictions: pd.DataFrame,
    player_event_predictions: pd.DataFrame,
) -> pd.DataFrame:
    if match_predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in match_predictions.iterrows():
        match_id = str(row.get("match_id", ""))
        home = str(row.get("home_team", "home"))
        away = str(row.get("away_team", "away"))
        p_home = _to_float(row.get("p_home_win"))
        p_draw = _to_float(row.get("p_draw"))
        p_away = _to_float(row.get("p_away_win"))
        expected_home = _to_float(row.get("expected_home_goals", row.get("lambda_home")))
        expected_away = _to_float(row.get("expected_away_goals", row.get("lambda_away")))
        outcomes = [(home, p_home), ("Draw", p_draw), (away, p_away)]
        favorite_selection, favorite_probability = max(outcomes, key=lambda item: item[1])
        top_scorelines, top_scoreline_probability, scoreline_entropy = _scoreline_summary(scoreline_distribution, match_id)
        outcome_entropy = _normalized_entropy([p_home, p_draw, p_away])
        uncertainty_score = scoreline_entropy if scoreline_entropy is not None else outcome_entropy
        warning_count = _warning_count(row.get("warnings", ""))
        unavailable_dynamic_rows = _unavailable_rows(dynamic_market_lines, match_id, "availability")
        low_quality_dynamic_rows = _quality_rows(dynamic_market_lines, match_id, "data_quality_flag")
        unavailable_team_rows = _unavailable_rows(team_stats_predictions, match_id, "availability")
        player_warning_rows = _quality_rows(player_event_predictions, match_id, "confidence_flag")

        data_quality_score = warning_count + unavailable_dynamic_rows + unavailable_team_rows + player_warning_rows
        rows.append(
            {
                "match_id": match_id,
                "match": f"{home} vs {away}",
                "home_team": home,
                "away_team": away,
                "expected_total_goals": expected_home + expected_away,
                "home_win_probability": p_home,
                "draw_probability": p_draw,
                "away_win_probability": p_away,
                "favorite_selection": favorite_selection,
                "favorite_probability": favorite_probability,
                "balance_score": max(0.0, 1.0 - favorite_probability),
                "outcome_entropy": outcome_entropy,
                "uncertainty_score": uncertainty_score,
                "btts_probability": _to_float(row.get("p_btts")),
                "over_2_5_probability": _to_float(row.get("p_over_25")),
                "under_2_5_probability": max(0.0, 1.0 - _to_float(row.get("p_over_25"))),
                "most_likely_score": row.get("most_likely_score", ""),
                "top_scoreline_probability": top_scoreline_probability,
                "top_scorelines": top_scorelines,
                "warning_count": warning_count,
                "unavailable_dynamic_rows": unavailable_dynamic_rows,
                "low_quality_dynamic_rows": low_quality_dynamic_rows,
                "unavailable_team_rows": unavailable_team_rows,
                "player_warning_rows": player_warning_rows,
                "data_quality_score": data_quality_score,
                "data_quality_flag": _data_quality_flag(data_quality_score, low_quality_dynamic_rows),
            }
        )
    return pd.DataFrame(rows)


def _rank_match_metric(
    frame: pd.DataFrame,
    *,
    category: str,
    metric_column: str,
    metric_name: str,
    ascending: bool,
    top_n: int,
    reason_code: str,
    secondary_metric_column: str,
    secondary_metric_name: str,
) -> list[dict[str, Any]]:
    if frame.empty or metric_column not in frame.columns:
        return []
    ranked = frame.copy()
    ranked["_metric_sort"] = pd.to_numeric(ranked[metric_column], errors="coerce")
    ranked = ranked.dropna(subset=["_metric_sort"])
    if ranked.empty:
        return []
    ranked = ranked.sort_values("_metric_sort", ascending=ascending).head(top_n)
    rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        rows.append(
            _summary_row(
                category=category,
                rank=rank,
                row=row,
                metric_name=metric_name,
                metric_value=row.get(metric_column),
                secondary_metric_name=secondary_metric_name,
                secondary_metric_value=row.get(secondary_metric_column),
                statistical_label=_label_for_category(category, _to_float(row.get(metric_column))),
                data_quality_flag=row.get("data_quality_flag", "standard"),
                evidence_tags=_evidence_tags_for_category(category, row),
                short_structured_reason=reason_code,
            )
        )
    return rows


def _rank_dynamic_line_signals(
    dynamic_market_lines: pd.DataFrame,
    match_predictions: pd.DataFrame,
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    if dynamic_market_lines.empty or "model_probability" not in dynamic_market_lines.columns:
        return []
    df = dynamic_market_lines.copy()
    if "availability" in df.columns:
        df = df[df["availability"].astype(str).eq("available")].copy()
    if df.empty:
        return []

    df["_model_probability"] = pd.to_numeric(df["model_probability"], errors="coerce")
    df = df.dropna(subset=["_model_probability"])
    if df.empty:
        return []
    df["_signal_strength"] = (df["_model_probability"] - 0.5).abs()
    # Keep match-level and team-level rows first; player rows are still
    # included but should not dominate the day summary.
    if "scope" in df.columns:
        df["_scope_sort"] = df["scope"].astype(str).map({"match": 3, "team": 2, "player": 1}).fillna(0)
    else:
        df["_scope_sort"] = 0
    df = df.sort_values(["_scope_sort", "_signal_strength"], ascending=[False, False]).head(top_n)

    match_lookup = _match_lookup(match_predictions)
    rows: list[dict[str, Any]] = []
    for rank, (_, line) in enumerate(df.iterrows(), start=1):
        match_id = str(line.get("match_id", ""))
        match = match_lookup.get(match_id, {})
        rows.append(
            {
                "ranking_category": "top_dynamic_statistical_signals",
                "rank": rank,
                "match_id": match_id,
                "match": match.get("match", ""),
                "home_team": match.get("home_team", ""),
                "away_team": match.get("away_team", ""),
                "metric_name": "model_probability",
                "metric_value": line.get("model_probability"),
                "secondary_metric_name": "fair_odds",
                "secondary_metric_value": line.get("fair_odds"),
                "market": line.get("market", ""),
                "scope": line.get("scope", ""),
                "side": line.get("side", line.get("team", line.get("player", ""))),
                "line": line.get("line", ""),
                "over_under": line.get("over_under", ""),
                "statistical_label": line.get("signal_label", "statistical_signal"),
                "data_quality_flag": line.get("data_quality_flag", ""),
                "evidence_tags": line.get("evidence_tags", ""),
                "short_structured_reason": line.get("reason_code", "DYNAMIC_LINE_MODEL_SIGNAL"),
            }
        )
    return rows


def _rank_data_quality_watchlist(base: pd.DataFrame, *, top_n: int) -> list[dict[str, Any]]:
    if base.empty or "data_quality_score" not in base.columns:
        return []
    ranked = base.copy()
    ranked["_quality_sort"] = pd.to_numeric(ranked["data_quality_score"], errors="coerce").fillna(0)
    ranked = ranked[ranked["_quality_sort"] > 0].sort_values("_quality_sort", ascending=False).head(top_n)
    rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        rows.append(
            _summary_row(
                category="data_quality_watchlist",
                rank=rank,
                row=row,
                metric_name="data_quality_score",
                metric_value=row.get("data_quality_score"),
                secondary_metric_name="warning_count",
                secondary_metric_value=row.get("warning_count"),
                statistical_label="review_recommended",
                data_quality_flag=row.get("data_quality_flag", "review"),
                evidence_tags=(
                    f"unavailable_dynamic_rows={int(row.get('unavailable_dynamic_rows', 0) or 0)};"
                    f"low_quality_dynamic_rows={int(row.get('low_quality_dynamic_rows', 0) or 0)};"
                    f"unavailable_team_rows={int(row.get('unavailable_team_rows', 0) or 0)};"
                    f"player_warning_rows={int(row.get('player_warning_rows', 0) or 0)}"
                ),
                short_structured_reason="DATA_QUALITY_REVIEW_RECOMMENDED",
            )
        )
    return rows


def _summary_row(
    *,
    category: str,
    rank: int,
    row: pd.Series,
    metric_name: str,
    metric_value: Any,
    secondary_metric_name: str,
    secondary_metric_value: Any,
    statistical_label: str,
    data_quality_flag: str,
    evidence_tags: str,
    short_structured_reason: str,
) -> dict[str, Any]:
    return {
        "ranking_category": category,
        "rank": rank,
        "match_id": row.get("match_id", ""),
        "match": row.get("match", ""),
        "home_team": row.get("home_team", ""),
        "away_team": row.get("away_team", ""),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "secondary_metric_name": secondary_metric_name,
        "secondary_metric_value": secondary_metric_value,
        "market": "",
        "scope": "match",
        "side": "",
        "line": "",
        "over_under": "",
        "statistical_label": statistical_label,
        "data_quality_flag": data_quality_flag,
        "evidence_tags": evidence_tags,
        "short_structured_reason": short_structured_reason,
    }


def _build_summary_payload(summary: pd.DataFrame, base: pd.DataFrame, audit: dict[str, Any] | None) -> dict[str, Any]:
    categories = {}
    if not summary.empty:
        for category, frame in summary.groupby("ranking_category", dropna=False):
            categories[str(category)] = frame.head(10).to_dict(orient="records")

    return {
        "version": MATCHDAY_SUMMARY_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_version": audit.get("version") if audit else None,
        "paper_mode": bool(audit.get("paper_mode", True)) if audit else True,
        "description": "Simulator-first daily rankings. These are statistical summaries, not betting recommendations.",
        "match_count": int(len(base)),
        "summary_rows": int(len(summary)),
        "categories": categories,
        "category_counts": {str(k): int(v) for k, v in summary["ranking_category"].value_counts().to_dict().items()} if not summary.empty else {},
        "principles": {
            "odds_required": False,
            "live_betting": False,
            "changes_model_logic": False,
            "betting_recommendations": False,
        },
    }


def _scoreline_summary(scoreline_distribution: pd.DataFrame, match_id: str) -> tuple[str, float, float | None]:
    if scoreline_distribution.empty or "match_id" not in scoreline_distribution.columns:
        return "", 0.0, None
    df = scoreline_distribution[scoreline_distribution["match_id"].astype(str).eq(match_id)].copy()
    if df.empty or "probability" not in df.columns:
        return "", 0.0, None
    df["_probability"] = pd.to_numeric(df["probability"], errors="coerce").fillna(0.0)
    df = df.sort_values("_probability", ascending=False)
    if "score" not in df.columns and {"home_goals", "away_goals"}.issubset(df.columns):
        df["score"] = df["home_goals"].astype(str) + "-" + df["away_goals"].astype(str)
    top = df.head(3)
    top_scorelines = "; ".join(f"{row.get('score', '')}:{float(row.get('_probability', 0.0)):.3f}" for _, row in top.iterrows())
    top_probability = float(df["_probability"].max()) if not df.empty else 0.0
    entropy = _normalized_entropy(df["_probability"].tolist())
    return top_scorelines, top_probability, entropy


def _normalized_entropy(probabilities: list[float]) -> float:
    clean = [float(p) for p in probabilities if p is not None and not pd.isna(p) and float(p) > 0]
    if not clean:
        return 0.0
    total = sum(clean)
    if total <= 0:
        return 0.0
    normalized = [p / total for p in clean]
    entropy = -sum(p * math.log(p) for p in normalized if p > 0)
    max_entropy = math.log(len(normalized)) if len(normalized) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy else 0.0


def _match_lookup(match_predictions: pd.DataFrame) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    if match_predictions.empty or "match_id" not in match_predictions.columns:
        return lookup
    for _, row in match_predictions.iterrows():
        match_id = str(row.get("match_id", ""))
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        lookup[match_id] = {"match": f"{home} vs {away}", "home_team": home, "away_team": away}
    return lookup


def _warning_count(value: Any) -> int:
    text = "" if value is None or pd.isna(value) else str(value)
    if not text or text.lower() in {"nan", "none"}:
        return 0
    return len([part for part in re_split_warnings(text) if part])


def re_split_warnings(text: str) -> list[str]:
    return [part.strip() for part in text.replace("|", ";").replace(",", ";").split(";")]


def _unavailable_rows(frame: pd.DataFrame, match_id: str, column: str) -> int:
    if frame.empty or "match_id" not in frame.columns or column not in frame.columns:
        return 0
    scoped = frame[frame["match_id"].astype(str).eq(match_id)]
    if scoped.empty:
        return 0
    return int(scoped[column].astype(str).ne("available").sum())


def _quality_rows(frame: pd.DataFrame, match_id: str, column: str) -> int:
    if frame.empty or "match_id" not in frame.columns or column not in frame.columns:
        return 0
    scoped = frame[frame["match_id"].astype(str).eq(match_id)]
    if scoped.empty:
        return 0
    pattern = "low|thin|partial|unavailable|blocked|zero|unresolved|review"
    return int(scoped[column].astype(str).str.contains(pattern, case=False, regex=True, na=False).sum())


def _data_quality_flag(score: int | float, low_quality_dynamic_rows: int | float) -> str:
    score_value = float(score or 0)
    low_quality_value = float(low_quality_dynamic_rows or 0)
    if score_value >= 10:
        return "review_required"
    if score_value >= 4 or low_quality_value >= 4:
        return "partial_context_review"
    return "standard"


def _label_for_category(category: str, value: float) -> str:
    if category == "high_goal_expectation":
        if value >= 3.0:
            return "high_goal_environment"
        if value >= 2.4:
            return "medium_goal_environment"
        return "low_model_goal_environment"
    if category == "low_goal_environment":
        if value >= 0.70:
            return "strong_low_goal_environment"
        if value >= 0.58:
            return "medium_low_goal_environment"
        return "thin_low_goal_environment"
    if category == "most_balanced_matches":
        if value >= 0.62:
            return "very_balanced"
        if value >= 0.52:
            return "balanced"
        return "moderately_balanced"
    if category == "strongest_favorites":
        if value >= 0.65:
            return "strong_favorite"
        if value >= 0.52:
            return "clear_favorite"
        return "mild_favorite"
    if category == "highest_uncertainty":
        if value >= 0.85:
            return "high_uncertainty"
        if value >= 0.70:
            return "medium_uncertainty"
        return "lower_uncertainty"
    if category == "btts_lean":
        if value >= 0.62:
            return "strong_btts_lean"
        if value >= 0.52:
            return "medium_btts_lean"
        return "thin_btts_lean"
    return "statistical_signal"


def _evidence_tags_for_category(category: str, row: pd.Series) -> str:
    base = [
        f"expected_total_goals={_to_float(row.get('expected_total_goals')):.3f}",
        f"favorite_probability={_to_float(row.get('favorite_probability')):.3f}",
        f"uncertainty_score={_to_float(row.get('uncertainty_score')):.3f}",
        f"data_quality_flag={row.get('data_quality_flag', '')}",
    ]
    if category in {"highest_uncertainty", "high_goal_expectation"}:
        base.append(f"top_scorelines={row.get('top_scorelines', '')}")
    if category == "data_quality_watchlist":
        base.extend(
            [
                f"warning_count={int(row.get('warning_count', 0) or 0)}",
                f"unavailable_dynamic_rows={int(row.get('unavailable_dynamic_rows', 0) or 0)}",
            ]
        )
    return ";".join(str(item) for item in base if str(item))


def _to_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
