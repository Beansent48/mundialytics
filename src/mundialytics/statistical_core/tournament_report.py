from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import math
import pandas as pd


TOURNAMENT_REPORT_VERSION = "v0.48.4_simulation_evaluation_foundation"

TOURNAMENT_REPORT_COLUMNS: tuple[str, ...] = (
    "report_section",
    "rank",
    "team",
    "group",
    "stage",
    "metric_name",
    "metric_value",
    "secondary_metric_name",
    "secondary_metric_value",
    "uncertainty_low",
    "uncertainty_high",
    "uncertainty_band",
    "statistical_label",
    "data_quality_flag",
    "short_structured_reason",
)


def build_tournament_report(
    *,
    tournament_simulation: pd.DataFrame | None,
    tournament_details: pd.DataFrame | None = None,
    match_predictions: pd.DataFrame | None = None,
    fixtures: pd.DataFrame | None = None,
    competition_summary: pd.DataFrame | None = None,
    top_scorer_predictions: pd.DataFrame | None = None,
    audit: dict[str, Any] | None = None,
    top_n: int = 12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build visual tournament report rows from existing simulator outputs.

    The report is deliberately a presentation/summary layer. It does not alter
    Monte Carlo simulation logic, model probabilities, odds handling or picks.
    """
    tournament_simulation = _ensure_frame(tournament_simulation)
    tournament_details = _ensure_frame(tournament_details)
    match_predictions = _ensure_frame(match_predictions)
    fixtures = _ensure_frame(fixtures)
    competition_summary = _ensure_frame(competition_summary)
    top_scorer_predictions = _ensure_frame(top_scorer_predictions)

    rows: list[dict[str, Any]] = []
    team_context = _build_team_context(fixtures)
    enriched = _enrich_tournament(tournament_simulation, team_context)

    if not enriched.empty:
        rows.extend(_rank_probability(enriched, "championship_race", "champion_probability", "CHAMPION_PROBABILITY_FROM_MONTE_CARLO", top_n))
        rows.extend(_rank_probability(enriched, "qualification_race", "qualify_group_probability", "QUALIFICATION_PROBABILITY_FROM_MONTE_CARLO", top_n))
        rows.extend(_rank_probability(enriched, "group_winner_race", "group_winner_probability", "GROUP_WINNER_PROBABILITY_FROM_MONTE_CARLO", top_n))
        rows.extend(_expected_group_table(enriched, top_n=top_n))
        rows.extend(_knockout_path(enriched, top_n=top_n))
        rows.extend(_attacking_projection(enriched, top_n=top_n))
        rows.extend(_uncertainty_watchlist(enriched, top_n=top_n))

    rows.extend(_top_scorer_section(top_scorer_predictions, top_n=min(top_n, 10)))
    rows.extend(_competition_headlines(competition_summary, top_n=8))

    report = pd.DataFrame(rows, columns=TOURNAMENT_REPORT_COLUMNS)
    if not report.empty:
        report["rank"] = pd.to_numeric(report["rank"], errors="coerce").astype("Int64")

    payload = _build_payload(report, enriched, tournament_details, audit)
    return report, payload


def _ensure_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _build_team_context(fixtures: pd.DataFrame) -> dict[str, dict[str, str]]:
    context: dict[str, dict[str, str]] = {}
    if fixtures.empty:
        return context
    for _, row in fixtures.iterrows():
        group = str(row.get("group", row.get("Group", "unknown")) or "unknown")
        stage = str(row.get("stage", row.get("Stage", "unknown")) or "unknown")
        for team_col in ("home_team", "away_team"):
            team = str(row.get(team_col, "") or "")
            if not team:
                continue
            previous = context.get(team, {})
            context[team] = {
                "group": previous.get("group") if previous.get("group") and previous.get("group") != "unknown" else group,
                "stage": previous.get("stage") if previous.get("stage") and previous.get("stage") != "unknown" else stage,
            }
    return context


def _enrich_tournament(tournament_simulation: pd.DataFrame, team_context: dict[str, dict[str, str]]) -> pd.DataFrame:
    if tournament_simulation.empty:
        return pd.DataFrame()
    df = tournament_simulation.copy()
    df["team"] = df["team"].astype(str)
    df["group"] = df["team"].map(lambda t: team_context.get(t, {}).get("group", "unknown"))
    df["stage"] = df["team"].map(lambda t: team_context.get(t, {}).get("stage", "tournament"))
    for col in [
        "group_winner_probability",
        "qualify_group_probability",
        "r16_probability",
        "qf_probability",
        "sf_probability",
        "final_probability",
        "champion_probability",
        "expected_points",
        "expected_goals_for",
        "simulations",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _rank_probability(df: pd.DataFrame, section: str, metric: str, reason: str, top_n: int) -> list[dict[str, Any]]:
    if df.empty or metric not in df.columns:
        return []
    work = df.sort_values(metric, ascending=False).head(top_n).copy()
    rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(work.iterrows(), start=1):
        p = float(row.get(metric, 0.0))
        low, high = _probability_interval(p, row.get("simulations"))
        rows.append(
            _row(
                report_section=section,
                rank=rank,
                team=row.get("team"),
                group=row.get("group", "unknown"),
                stage=row.get("stage", "tournament"),
                metric_name=metric,
                metric_value=p,
                secondary_metric_name="expected_points",
                secondary_metric_value=row.get("expected_points", ""),
                uncertainty_low=low,
                uncertainty_high=high,
                statistical_label=_probability_label(p),
                data_quality_flag=_simulation_quality(row.get("simulations")),
                short_structured_reason=reason,
            )
        )
    return rows


def _expected_group_table(df: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    if df.empty or "expected_points" not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    if "group" in df.columns and df["group"].astype(str).ne("unknown").any():
        iterator = df.groupby("group", dropna=False)
    else:
        iterator = [("all", df)]
    for group, frame in iterator:
        work = frame.sort_values(["expected_points", "qualify_group_probability"], ascending=False).head(top_n)
        for rank, (_, row) in enumerate(work.iterrows(), start=1):
            rows.append(
                _row(
                    report_section="expected_group_table",
                    rank=rank,
                    team=row.get("team"),
                    group=group,
                    stage="group",
                    metric_name="expected_points",
                    metric_value=row.get("expected_points", 0.0),
                    secondary_metric_name="qualify_group_probability",
                    secondary_metric_value=row.get("qualify_group_probability", ""),
                    uncertainty_low="",
                    uncertainty_high="",
                    statistical_label="EXPECTED_POINTS_ORDERING",
                    data_quality_flag=_simulation_quality(row.get("simulations")),
                    short_structured_reason="GROUP_TABLE_EXPECTED_POINTS_FROM_MONTE_CARLO",
                )
            )
    return rows


def _knockout_path(df: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    stage_metrics = [
        ("r16_probability", "round_of_16_path"),
        ("qf_probability", "quarter_final_path"),
        ("sf_probability", "semi_final_path"),
        ("final_probability", "final_path"),
        ("champion_probability", "champion_path"),
    ]
    rows: list[dict[str, Any]] = []
    for metric, section in stage_metrics:
        if metric not in df.columns:
            continue
        work = df.sort_values(metric, ascending=False).head(max(1, min(top_n, 8)))
        for rank, (_, row) in enumerate(work.iterrows(), start=1):
            p = float(row.get(metric, 0.0))
            low, high = _probability_interval(p, row.get("simulations"))
            rows.append(
                _row(
                    report_section=section,
                    rank=rank,
                    team=row.get("team"),
                    group=row.get("group", "unknown"),
                    stage=section.replace("_path", ""),
                    metric_name=metric,
                    metric_value=p,
                    secondary_metric_name="champion_probability",
                    secondary_metric_value=row.get("champion_probability", ""),
                    uncertainty_low=low,
                    uncertainty_high=high,
                    statistical_label=_probability_label(p),
                    data_quality_flag=_simulation_quality(row.get("simulations")),
                    short_structured_reason="ROUND_PROGRESSION_PROBABILITY_FROM_MONTE_CARLO",
                )
            )
    return rows


def _attacking_projection(df: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    if df.empty or "expected_goals_for" not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    work = df.sort_values("expected_goals_for", ascending=False).head(top_n)
    for rank, (_, row) in enumerate(work.iterrows(), start=1):
        rows.append(
            _row(
                report_section="attacking_projection",
                rank=rank,
                team=row.get("team"),
                group=row.get("group", "unknown"),
                stage="tournament",
                metric_name="expected_goals_for",
                metric_value=row.get("expected_goals_for", 0.0),
                secondary_metric_name="expected_points",
                secondary_metric_value=row.get("expected_points", ""),
                uncertainty_low="",
                uncertainty_high="",
                statistical_label="EXPECTED_ATTACKING_OUTPUT",
                data_quality_flag=_simulation_quality(row.get("simulations")),
                short_structured_reason="EXPECTED_GOALS_FOR_ACCUMULATED_FROM_SIMULATED_GROUP_MATCHES",
            )
        )
    return rows


def _uncertainty_watchlist(df: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    prob_cols = [c for c in ["qualify_group_probability", "qf_probability", "sf_probability", "final_probability", "champion_probability"] if c in df.columns]
    if not prob_cols:
        return rows
    work = df.copy()
    # Highest uncertainty is near 50% for key probabilities; this is more useful
    # than ranking only by Monte Carlo standard error, which rewards low n.
    work["_decision_uncertainty"] = work[prob_cols].apply(lambda s: max(1.0 - abs(float(v) - 0.5) * 2.0 for v in s), axis=1)
    work = work.sort_values("_decision_uncertainty", ascending=False).head(top_n)
    for rank, (_, row) in enumerate(work.iterrows(), start=1):
        metric = max(prob_cols, key=lambda col: 1.0 - abs(float(row.get(col, 0.0)) - 0.5) * 2.0)
        p = float(row.get(metric, 0.0))
        low, high = _probability_interval(p, row.get("simulations"))
        rows.append(
            _row(
                report_section="uncertainty_watchlist",
                rank=rank,
                team=row.get("team"),
                group=row.get("group", "unknown"),
                stage="tournament",
                metric_name=metric,
                metric_value=p,
                secondary_metric_name="decision_uncertainty",
                secondary_metric_value=row.get("_decision_uncertainty", ""),
                uncertainty_low=low,
                uncertainty_high=high,
                statistical_label="HIGH_DECISION_UNCERTAINTY",
                data_quality_flag=_simulation_quality(row.get("simulations")),
                short_structured_reason="PROGRESSION_PROBABILITY_NEAR_DECISION_BOUNDARY",
            )
        )
    return rows


def _top_scorer_section(top_scorer_predictions: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    if top_scorer_predictions.empty:
        return []
    probability_col = "top_scorer_probability"
    if probability_col not in top_scorer_predictions.columns:
        probability_col = "top_scorer_probability_approx" if "top_scorer_probability_approx" in top_scorer_predictions.columns else ""
    if not probability_col:
        return []
    work = top_scorer_predictions.copy()
    work[probability_col] = pd.to_numeric(work[probability_col], errors="coerce").fillna(0.0)
    work = work.sort_values(probability_col, ascending=False).head(top_n)
    rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(work.iterrows(), start=1):
        p = float(row.get(probability_col, 0.0))
        rows.append(
            _row(
                report_section="top_scorer_projection",
                rank=rank,
                team=row.get("team", ""),
                group="",
                stage="player_awards_experimental",
                metric_name=probability_col,
                metric_value=p,
                secondary_metric_name="player",
                secondary_metric_value=row.get("player", ""),
                uncertainty_low="",
                uncertainty_high="",
                statistical_label="EXPERIMENTAL_PLAYER_AWARD_PROJECTION",
                data_quality_flag=str(row.get("confidence", "experimental")),
                short_structured_reason="TOP_SCORER_IS_APPROXIMATE_AND_REQUIRES_RELIABLE_PLAYER_EVENT_INPUTS",
            )
        )
    return rows


def _competition_headlines(competition_summary: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    if competition_summary.empty:
        return []
    rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(competition_summary.head(top_n).iterrows(), start=1):
        rows.append(
            _row(
                report_section="competition_headlines",
                rank=rank,
                team=row.get("team", ""),
                group="",
                stage=str(row.get("record_type", "competition")),
                metric_name="headline",
                metric_value="",
                secondary_metric_name="headline",
                secondary_metric_value=row.get("headline", ""),
                uncertainty_low="",
                uncertainty_high="",
                statistical_label="COMPETITION_LEVEL_SUMMARY",
                data_quality_flag=str(row.get("confidence", "summary")),
                short_structured_reason="COMPETITION_FORECAST_HEADLINE_FROM_EXISTING_OUTPUTS",
            )
        )
    return rows


def _row(**kwargs: Any) -> dict[str, Any]:
    low = kwargs.get("uncertainty_low", "")
    high = kwargs.get("uncertainty_high", "")
    uncertainty_band = ""
    if low != "" and high != "":
        uncertainty_band = f"{float(low):.3f}-{float(high):.3f}"
    return {
        "report_section": kwargs.get("report_section", ""),
        "rank": kwargs.get("rank", ""),
        "team": kwargs.get("team", ""),
        "group": kwargs.get("group", ""),
        "stage": kwargs.get("stage", ""),
        "metric_name": kwargs.get("metric_name", ""),
        "metric_value": kwargs.get("metric_value", ""),
        "secondary_metric_name": kwargs.get("secondary_metric_name", ""),
        "secondary_metric_value": kwargs.get("secondary_metric_value", ""),
        "uncertainty_low": low,
        "uncertainty_high": high,
        "uncertainty_band": uncertainty_band,
        "statistical_label": kwargs.get("statistical_label", ""),
        "data_quality_flag": kwargs.get("data_quality_flag", ""),
        "short_structured_reason": kwargs.get("short_structured_reason", ""),
    }


def _probability_interval(probability: float, simulations: Any) -> tuple[float, float]:
    try:
        n = int(float(simulations))
    except (TypeError, ValueError):
        n = 0
    p = min(max(float(probability), 0.0), 1.0)
    if n <= 0:
        return p, p
    margin = 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return max(0.0, p - margin), min(1.0, p + margin)


def _probability_label(probability: float) -> str:
    p = float(probability)
    if p >= 0.75:
        return "VERY_HIGH_PROBABILITY"
    if p >= 0.55:
        return "HIGH_PROBABILITY"
    if p >= 0.35:
        return "MEDIUM_PROBABILITY"
    if p >= 0.15:
        return "LOW_MEDIUM_PROBABILITY"
    return "LOW_PROBABILITY"


def _simulation_quality(simulations: Any) -> str:
    try:
        n = int(float(simulations))
    except (TypeError, ValueError):
        return "simulation_count_unknown"
    if n >= 50000:
        return "large_run_ready"
    if n >= 5000:
        return "medium_run"
    return "smoke_run_not_for_final_probability_claims"


def _build_payload(
    report: pd.DataFrame,
    tournament_simulation: pd.DataFrame,
    tournament_details: pd.DataFrame,
    audit: dict[str, Any] | None,
) -> dict[str, Any]:
    categories = []
    category_counts: dict[str, int] = {}
    if not report.empty and "report_section" in report.columns:
        categories = sorted(report["report_section"].dropna().astype(str).unique().tolist())
        category_counts = {str(k): int(v) for k, v in report["report_section"].astype(str).value_counts().to_dict().items()}
    n_simulations = None
    if not tournament_simulation.empty and "simulations" in tournament_simulation.columns:
        values = pd.to_numeric(tournament_simulation["simulations"], errors="coerce").dropna()
        if not values.empty:
            n_simulations = int(values.max())
    return {
        "version": TOURNAMENT_REPORT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_mode": bool(audit.get("paper_mode", True)) if audit else True,
        "summary_rows": int(len(report)),
        "tournament_rows": int(len(tournament_simulation)),
        "detail_rows": int(len(tournament_details)),
        "n_simulations": n_simulations,
        "recommended_large_run_n_simulations": 50000,
        "large_run_ready": bool(n_simulations is not None and n_simulations >= 50000),
        "categories": categories,
        "category_counts": category_counts,
        "principles": {
            "odds_required": False,
            "betting_recommendations": False,
            "model_logic_changed": False,
            "tournament_report_is_summary_layer": True,
        },
    }
