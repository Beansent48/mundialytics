from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd


REPORT_VERSION_LABEL = "Mundialytics Statistical Simulator v0.48.4"


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "<p><em>No data available.</em></p>"
    safe = df.head(max_rows).copy()
    safe = _format_table_values(safe)
    return safe.to_html(index=False, escape=True, classes="data-table")


def build_daily_html_report(
    out_path: str | Path,
    match_predictions: pd.DataFrame,
    team_stats_predictions: pd.DataFrame,
    player_event_predictions: pd.DataFrame,
    betting_edges: pd.DataFrame,
    tournament_simulation: pd.DataFrame | None = None,
    top_scorer_predictions: pd.DataFrame | None = None,
    award_predictions: pd.DataFrame | None = None,
    competition_summary: pd.DataFrame | None = None,
    dynamic_market_lines: pd.DataFrame | None = None,
    audit: dict[str, Any] | None = None,
    scoreline_distribution: pd.DataFrame | None = None,
    matchday_summary: pd.DataFrame | None = None,
    tournament_report: pd.DataFrame | None = None,
) -> Path:
    """Build the v0.48.4 advanced statistical matchday/tournament HTML report.

    The report intentionally surfaces existing simulator outputs instead of
    changing model logic. Betting information is shown only as optional paper
    context; the main report remains statistical and simulator-first.
    """
    # Backwards compatibility with v0.20/v0.21 calls where audit was the
    # seventh positional argument.
    if isinstance(top_scorer_predictions, dict) and audit is None:
        audit = top_scorer_predictions
        top_scorer_predictions = None

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    match_predictions = _ensure_frame(match_predictions)
    team_stats_predictions = _ensure_frame(team_stats_predictions)
    player_event_predictions = _ensure_frame(player_event_predictions)
    betting_edges = _ensure_frame(betting_edges)
    tournament_simulation = _ensure_frame(tournament_simulation)
    top_scorer_predictions = _ensure_frame(top_scorer_predictions)
    award_predictions = _ensure_frame(award_predictions)
    competition_summary = _ensure_frame(competition_summary)
    dynamic_market_lines = _ensure_frame(dynamic_market_lines)
    scoreline_distribution = _ensure_frame(scoreline_distribution)
    matchday_summary = _ensure_frame(matchday_summary)
    tournament_report = _ensure_frame(tournament_report)

    warnings = audit.get("warnings", []) if audit else []
    picks = _recommended_picks(betting_edges)
    executive_summary = _build_executive_summary(match_predictions)
    not_available = _build_not_available_markets(dynamic_market_lines, team_stats_predictions, player_event_predictions)
    match_cards = "\n".join(
        _build_match_card(
            row=row,
            scoreline_distribution=scoreline_distribution,
            dynamic_market_lines=dynamic_market_lines,
            team_stats_predictions=team_stats_predictions,
            player_event_predictions=player_event_predictions,
        )
        for _, row in match_predictions.iterrows()
    )
    if not match_cards:
        match_cards = "<p><em>No match predictions available.</em></p>"

    css = """
    :root { --border: #d9dee7; --muted: #626c7a; --bg: #f7f9fc; --card: #ffffff; }
    body { font-family: Arial, sans-serif; margin: 28px; color: #111827; background: #ffffff; }
    h1 { margin-bottom: 8px; }
    h2 { margin-top: 30px; padding-bottom: 6px; border-bottom: 2px solid var(--border); }
    h3 { margin-top: 22px; }
    h4 { margin-bottom: 8px; }
    .subtitle { color: var(--muted); margin-top: 0; }
    .badge { display: inline-block; padding: 4px 8px; border: 1px solid #9aa4b2; border-radius: 999px; margin: 4px 6px 4px 0; font-size: 12px; background: var(--bg); }
    .warn { background: #fff7df; padding: 10px; border: 1px solid #d7aa28; border-radius: 8px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 14px 0; }
    .metric { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 10px; }
    .metric .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }
    .metric .value { font-size: 20px; font-weight: 700; margin-top: 4px; }
    .match-card { border: 1px solid var(--border); border-radius: 14px; padding: 16px; margin: 18px 0; background: var(--card); box-shadow: 0 1px 2px rgba(0,0,0,.04); }
    .match-title { display: flex; justify-content: space-between; gap: 16px; align-items: baseline; flex-wrap: wrap; }
    .muted { color: var(--muted); }
    table.data-table { border-collapse: collapse; width: 100%; margin: 10px 0 20px; font-size: 13px; }
    table.data-table th, table.data-table td { border: 1px solid #e2e8f0; padding: 6px; vertical-align: top; }
    table.data-table th { background: #f1f5f9; text-align: left; }
    .small { font-size: 12px; }
    .summary-block { margin: 14px 0 22px; }
    .prob-row { display: grid; grid-template-columns: minmax(160px, 1.4fr) minmax(220px, 3fr) minmax(90px, .8fr); gap: 10px; align-items: center; padding: 7px 0; border-bottom: 1px solid #eef2f7; }
    .bar-track { background: #e5e7eb; border-radius: 999px; overflow: hidden; height: 14px; }
    .bar-fill { background: #1f2937; height: 14px; border-radius: 999px; }
    pre { white-space: pre-wrap; background: #0f172a; color: #e5e7eb; padding: 12px; border-radius: 8px; overflow-x: auto; }
    """

    html_doc = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Mundialytics Advanced Match Report</title><style>{css}</style></head>
<body>
<h1>{REPORT_VERSION_LABEL}</h1>
<p class='subtitle'>Advanced Match Report — simulator-first, auditable, paper mode only.</p>
<p>
  <span class='badge'>Statistical simulator first</span>
  <span class='badge'>No live betting</span>
  <span class='badge'>Odds optional</span>
  <span class='badge'>Current player gate</span>
  <span class='badge'>Dynamic lines</span>
</p>

<div class='warn'><strong>Warnings:</strong> {html.escape('; '.join(map(str, warnings)) if warnings else 'none')}</div>

<h2>Executive Summary</h2>
{dataframe_to_html_table(executive_summary, 50)}

<h2>Matchday Summary Rankings</h2>
<p class='muted'>Simulator-first daily rankings. These sections order the day statistically; they are not betting picks.</p>
{_render_matchday_summary_sections(matchday_summary)}

<h2>Match Probabilities</h2>
<p class='muted'>1X2 probabilities, expected goals and top outcomes from the statistical core.</p>
{dataframe_to_html_table(_select_columns(match_predictions, ['match_id','home_team','away_team','lambda_home','lambda_away','expected_home_goals','expected_away_goals','p_home_win','p_draw','p_away_win','p_over_25','p_btts','most_likely_score','most_likely_score_probability','warnings']), 80)}

<h2>Advanced Match Cards</h2>
{match_cards}

<h2>Dynamic Goal Lines</h2>
<p class='muted'>Line-specific probabilities and fair odds. These are statistical outputs, not betting recommendations.</p>
{dataframe_to_html_table(_dynamic_line_overview(dynamic_market_lines), 120)}

<h2>Not Available Markets</h2>
<p class='muted'>Markets are kept explicit when the input data is unavailable or unreliable. The engine must not invent missing data.</p>
{dataframe_to_html_table(not_available, 100)}

<h2>Team Statistics</h2>
{dataframe_to_html_table(_select_columns(team_stats_predictions, ['match_id','team','opponent','market','expected_count','availability','confidence','warnings','model_type']), 120)}

<h2>Player Statistics</h2>
{dataframe_to_html_table(_select_columns(player_event_predictions, ['match_id','team','player','market','line','expected_count','safe_probability','expected_minutes','sample_size_minutes','player_input_source','player_selection_confidence','candidate_rank_team','candidate_policy','candidate_reason','input_position','resolved_position','position_group','confidence_flag','warnings']), 120)}

<h2>Data Quality</h2>
{dataframe_to_html_table(_build_data_quality_table(match_predictions, team_stats_predictions, player_event_predictions, dynamic_market_lines, audit), 120)}

<h2>Tournament Visual Report</h2>
<p class='muted'>Monte Carlo tournament summary: champion race, qualification, expected group tables, paths and uncertainty. This is not a betting screen.</p>
{_render_tournament_report_sections(tournament_report)}

<h2>Tournament Simulation</h2>
{dataframe_to_html_table(tournament_simulation, 80)}

<h2>Competition Forecast</h2>
{dataframe_to_html_table(competition_summary, 50)}

<h2>Top Scorer Forecast</h2>
{dataframe_to_html_table(_select_columns(top_scorer_predictions, ['player','team','expected_goals_current_fixtures','expected_tournament_goals_approx','top_scorer_probability','confidence','warnings']), 50)}

<h2>Award Forecast</h2>
{dataframe_to_html_table(award_predictions, 60)}

<h2>Optional Paper Value Context</h2>
<p class='muted'>Shown only when odds are supplied. This section remains paper mode and does not execute bets.</p>
<h3>Recommended picks</h3>
{dataframe_to_html_table(_select_columns(picks, ['match_id','market','selection','line','odds_decimal','model_probability','implied_probability','edge','ev','stake_virtual','confidence','risk','reason']), 40)}
<h3>Betting edges</h3>
{dataframe_to_html_table(betting_edges, 80)}

<h2>Simulation Metadata</h2>
<p><strong>Run version:</strong> {html.escape(str(audit.get('version', 'unknown') if audit else 'unknown'))}</p>
{dataframe_to_html_table(pd.DataFrame([audit.get('simulation_policy', {})]) if audit and audit.get('simulation_policy') else pd.DataFrame(), 5)}
{dataframe_to_html_table(pd.DataFrame([audit.get('tournament_simulator', {})]) if audit and audit.get('tournament_simulator') else pd.DataFrame(), 5)}

<h2>Audit Summary</h2>
<pre>{html.escape(str(audit or {}))}</pre>
</body></html>"""
    out.write_text(html_doc, encoding="utf-8")
    return out




def _render_tournament_report_sections(tournament_report: pd.DataFrame) -> str:
    if tournament_report is None or tournament_report.empty or "report_section" not in tournament_report.columns:
        return "<p><em>No tournament visual report available.</em></p>"

    section_order = [
        ("championship_race", "Championship Race", True),
        ("qualification_race", "Qualification Race", True),
        ("group_winner_race", "Group Winner Race", True),
        ("expected_group_table", "Expected Group Tables", False),
        ("quarter_final_path", "Quarter-Final Path", True),
        ("semi_final_path", "Semi-Final Path", True),
        ("final_path", "Final Path", True),
        ("champion_path", "Champion Path", True),
        ("attacking_projection", "Attacking Projection", False),
        ("uncertainty_watchlist", "Uncertainty Watchlist", True),
        ("top_scorer_projection", "Top Scorer Projection", True),
        ("competition_headlines", "Competition Headlines", False),
    ]
    blocks: list[str] = []
    for section, title, include_bars in section_order:
        frame = tournament_report[tournament_report["report_section"].astype(str).eq(section)].copy()
        if frame.empty:
            continue
        columns = [
            "rank",
            "team",
            "group",
            "stage",
            "metric_name",
            "metric_value",
            "secondary_metric_name",
            "secondary_metric_value",
            "uncertainty_band",
            "statistical_label",
            "data_quality_flag",
            "short_structured_reason",
        ]
        body = _render_probability_bars(frame) if include_bars else ""
        blocks.append(
            "<div class='summary-block'>"
            f"<h3>{html.escape(title)}</h3>"
            f"{body}"
            f"{dataframe_to_html_table(_select_columns(frame, columns), 12)}"
            "</div>"
        )
    if not blocks:
        return dataframe_to_html_table(tournament_report, 80)
    return "\n".join(blocks)


def _render_probability_bars(frame: pd.DataFrame, max_rows: int = 8) -> str:
    if frame is None or frame.empty or "metric_value" not in frame.columns:
        return ""
    rows: list[str] = []
    for _, row in frame.head(max_rows).iterrows():
        value = _to_float(row.get("metric_value"))
        value = max(0.0, min(1.0, value))
        pct = value * 100.0
        label = str(row.get("team", ""))
        secondary = str(row.get("secondary_metric_value", ""))
        if secondary and str(row.get("secondary_metric_name", "")) in {"player", "headline"}:
            label = secondary if not label else f"{secondary} · {label}"
        rows.append(
            "<div class='prob-row'>"
            f"<div>{html.escape(label)}</div>"
            "<div class='bar-track'>"
            f"<div class='bar-fill' style='width:{pct:.1f}%'></div>"
            "</div>"
            f"<div>{pct:.1f}%</div>"
            "</div>"
        )
    if not rows:
        return ""
    return "<div class='prob-bars'>" + "\n".join(rows) + "</div>"

def _render_matchday_summary_sections(matchday_summary: pd.DataFrame) -> str:
    if matchday_summary is None or matchday_summary.empty or "ranking_category" not in matchday_summary.columns:
        return "<p><em>No matchday summary rankings available.</em></p>"

    category_order = [
        ("high_goal_expectation", "High Goal Expectation"),
        ("low_goal_environment", "Low Goal Environment"),
        ("most_balanced_matches", "Most Balanced Matches"),
        ("strongest_favorites", "Strongest Favorites"),
        ("highest_uncertainty", "Highest Uncertainty"),
        ("btts_lean", "BTTS Lean"),
        ("top_dynamic_statistical_signals", "Top Dynamic Statistical Signals"),
        ("data_quality_watchlist", "Data Quality Watchlist"),
    ]
    blocks: list[str] = []
    for category, title in category_order:
        frame = matchday_summary[matchday_summary["ranking_category"].astype(str).eq(category)].copy()
        if frame.empty:
            continue
        columns = [
            "rank",
            "match",
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
            "short_structured_reason",
        ]
        blocks.append(
            "<div class='summary-block'>"
            f"<h3>{html.escape(title)}</h3>"
            f"{dataframe_to_html_table(_select_columns(frame, columns), 10)}"
            "</div>"
        )
    if not blocks:
        return dataframe_to_html_table(matchday_summary, 50)
    return "\n".join(blocks)

def _ensure_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df[[c for c in columns if c in df.columns]].copy()


def _format_table_values(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_numeric_dtype(safe[col]):
            safe[col] = safe[col].map(_format_numeric)
    return safe.astype(object).where(pd.notna(safe), "")


def _format_numeric(value: Any) -> Any:
    try:
        if pd.isna(value):
            return ""
        number = float(value)
    except (TypeError, ValueError):
        return value
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if 0 <= number <= 1:
        return f"{number:.3f}"
    return f"{number:.2f}"


def _format_probability(value: Any) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_count(value: Any) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _recommended_picks(betting_edges: pd.DataFrame) -> pd.DataFrame:
    if betting_edges is None or betting_edges.empty or "recommended" not in betting_edges.columns:
        return pd.DataFrame()
    return betting_edges[betting_edges["recommended"] == True].copy()


def _build_executive_summary(match_predictions: pd.DataFrame) -> pd.DataFrame:
    if match_predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in match_predictions.iterrows():
        home = str(row.get("home_team", "home"))
        away = str(row.get("away_team", "away"))
        p_home = _to_float(row.get("p_home_win"))
        p_draw = _to_float(row.get("p_draw"))
        p_away = _to_float(row.get("p_away_win"))
        outcomes = [(home, p_home), ("Draw", p_draw), (away, p_away)]
        favorite, favorite_probability = max(outcomes, key=lambda item: item[1])
        expected_home = _to_float(row.get("expected_home_goals", row.get("lambda_home")))
        expected_away = _to_float(row.get("expected_away_goals", row.get("lambda_away")))
        rows.append(
            {
                "match_id": row.get("match_id"),
                "match": f"{home} vs {away}",
                "favorite_or_most_likely_outcome": favorite,
                "favorite_probability": favorite_probability,
                "home_win": p_home,
                "draw": p_draw,
                "away_win": p_away,
                "expected_total_goals": expected_home + expected_away,
                "btts_probability": _to_float(row.get("p_btts")),
                "over_2_5_probability": _to_float(row.get("p_over_25")),
                "most_likely_score": row.get("most_likely_score"),
                "warning_flags": row.get("warnings", ""),
            }
        )
    return pd.DataFrame(rows)


def _build_match_card(
    *,
    row: pd.Series,
    scoreline_distribution: pd.DataFrame,
    dynamic_market_lines: pd.DataFrame,
    team_stats_predictions: pd.DataFrame,
    player_event_predictions: pd.DataFrame,
) -> str:
    match_id = str(row.get("match_id", ""))
    home = str(row.get("home_team", "home"))
    away = str(row.get("away_team", "away"))

    scorelines = _top_scorelines(scoreline_distribution, match_id, max_rows=8)
    dynamic_lines = _match_dynamic_lines(dynamic_market_lines, match_id, max_rows=16)
    team_stats = _match_team_stats(team_stats_predictions, match_id, max_rows=20)
    player_stats = _match_player_stats(player_event_predictions, match_id, max_rows=16)
    quality = _match_quality_summary(row, dynamic_market_lines, team_stats_predictions, player_event_predictions, match_id)

    metrics = f"""
    <div class='grid'>
      <div class='metric'><div class='label'>Home win</div><div class='value'>{_format_probability(row.get('p_home_win'))}</div></div>
      <div class='metric'><div class='label'>Draw</div><div class='value'>{_format_probability(row.get('p_draw'))}</div></div>
      <div class='metric'><div class='label'>Away win</div><div class='value'>{_format_probability(row.get('p_away_win'))}</div></div>
      <div class='metric'><div class='label'>Expected goals</div><div class='value'>{_format_count(row.get('lambda_home'))} - {_format_count(row.get('lambda_away'))}</div></div>
      <div class='metric'><div class='label'>BTTS</div><div class='value'>{_format_probability(row.get('p_btts'))}</div></div>
      <div class='metric'><div class='label'>Over 2.5</div><div class='value'>{_format_probability(row.get('p_over_25'))}</div></div>
      <div class='metric'><div class='label'>Most likely score</div><div class='value'>{html.escape(str(row.get('most_likely_score', 'n/a')))}</div></div>
    </div>
    """

    return f"""
    <section class='match-card'>
      <div class='match-title'>
        <h3>{html.escape(home)} vs {html.escape(away)}</h3>
        <span class='muted'>{html.escape(match_id)}</span>
      </div>
      <p class='muted'>{html.escape(str(row.get('competition', '')))} · {html.escape(str(row.get('stage', '')))} · {html.escape(str(row.get('date', '')))}</p>
      <h4>Match Probabilities</h4>
      {metrics}
      <h4>Top Scorelines</h4>
      {dataframe_to_html_table(scorelines, 8)}
      <h4>Dynamic Goal Lines</h4>
      {dataframe_to_html_table(dynamic_lines, 16)}
      <h4>Team Statistics</h4>
      {dataframe_to_html_table(team_stats, 20)}
      <h4>Player Statistics</h4>
      {dataframe_to_html_table(player_stats, 16)}
      <h4>Data Quality</h4>
      {dataframe_to_html_table(quality, 20)}
    </section>
    """


def _top_scorelines(scoreline_distribution: pd.DataFrame, match_id: str, max_rows: int) -> pd.DataFrame:
    if scoreline_distribution.empty or "match_id" not in scoreline_distribution.columns:
        return pd.DataFrame()
    df = scoreline_distribution[scoreline_distribution["match_id"].astype(str).eq(match_id)].copy()
    if df.empty:
        return pd.DataFrame()
    if "probability" in df.columns:
        df["_probability_sort"] = pd.to_numeric(df["probability"], errors="coerce").fillna(0.0)
        df = df.sort_values("_probability_sort", ascending=False)
    if "score" not in df.columns and {"home_goals", "away_goals"}.issubset(df.columns):
        df["score"] = df["home_goals"].astype(str) + "-" + df["away_goals"].astype(str)
    return _select_columns(df.head(max_rows), ["score", "home_goals", "away_goals", "probability"])


def _match_dynamic_lines(dynamic_market_lines: pd.DataFrame, match_id: str, max_rows: int) -> pd.DataFrame:
    if dynamic_market_lines.empty or "match_id" not in dynamic_market_lines.columns:
        return pd.DataFrame()
    df = dynamic_market_lines[dynamic_market_lines["match_id"].astype(str).eq(match_id)].copy()
    if df.empty:
        return pd.DataFrame()
    if "market" in df.columns:
        preferred = df["market"].astype(str).str.contains("goals|btts|shots|cards|fouls|saves|corners", case=False, na=False)
        df = df[preferred].copy() if preferred.any() else df
    if "availability" in df.columns:
        df["_availability_sort"] = df["availability"].astype(str).eq("available").astype(int)
    else:
        df["_availability_sort"] = 0
    if "model_probability" in df.columns:
        df["_probability_sort"] = (pd.to_numeric(df["model_probability"], errors="coerce") - 0.5).abs()
    else:
        df["_probability_sort"] = 0
    df = df.sort_values(["_availability_sort", "_probability_sort"], ascending=[False, False])
    return _select_columns(
        df.head(max_rows),
        [
            "market",
            "scope",
            "side",
            "team",
            "player",
            "line",
            "over_under",
            "model_probability",
            "fair_odds",
            "expected_stat",
            "signal_label",
            "value_label",
            "availability",
            "data_quality_flag",
            "evidence_tags",
            "reason_code",
        ],
    )


def _match_team_stats(team_stats_predictions: pd.DataFrame, match_id: str, max_rows: int) -> pd.DataFrame:
    if team_stats_predictions.empty or "match_id" not in team_stats_predictions.columns:
        return pd.DataFrame()
    df = team_stats_predictions[team_stats_predictions["match_id"].astype(str).eq(match_id)].copy()
    if "expected_count" in df.columns:
        df["_expected_sort"] = pd.to_numeric(df["expected_count"], errors="coerce").fillna(0.0)
        df = df.sort_values(["team", "_expected_sort"], ascending=[True, False])
    return _select_columns(df.head(max_rows), ["team", "opponent", "market", "expected_count", "availability", "confidence", "warnings"])


def _match_player_stats(player_event_predictions: pd.DataFrame, match_id: str, max_rows: int) -> pd.DataFrame:
    if player_event_predictions.empty or "match_id" not in player_event_predictions.columns:
        return pd.DataFrame()
    df = player_event_predictions[player_event_predictions["match_id"].astype(str).eq(match_id)].copy()
    if "safe_probability" in df.columns:
        df["_probability_sort"] = pd.to_numeric(df["safe_probability"], errors="coerce").fillna(0.0)
        df = df.sort_values("_probability_sort", ascending=False)
    return _select_columns(
        df.head(max_rows),
        [
            "team",
            "player",
            "market",
            "line",
            "expected_count",
            "safe_probability",
            "expected_minutes",
            "sample_size_minutes",
            "player_input_source",
            "player_selection_confidence",
            "candidate_rank_team",
            "position_group",
            "confidence_flag",
            "warnings",
        ],
    )


def _build_not_available_markets(
    dynamic_market_lines: pd.DataFrame,
    team_stats_predictions: pd.DataFrame,
    player_event_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not dynamic_market_lines.empty and "availability" in dynamic_market_lines.columns:
        unavailable = dynamic_market_lines[dynamic_market_lines["availability"].astype(str).ne("available")]
        for _, row in unavailable.head(80).iterrows():
            rows.append(
                {
                    "source": "dynamic_market_lines",
                    "match_id": row.get("match_id"),
                    "market": row.get("market"),
                    "scope": row.get("scope"),
                    "side": row.get("side"),
                    "line": row.get("line"),
                    "availability": row.get("availability"),
                    "data_quality_flag": row.get("data_quality_flag"),
                    "reason": row.get("reason_code", row.get("evidence_tags", "")),
                }
            )
    if not team_stats_predictions.empty and "availability" in team_stats_predictions.columns:
        unavailable = team_stats_predictions[team_stats_predictions["availability"].astype(str).ne("available")]
        for _, row in unavailable.head(40).iterrows():
            rows.append(
                {
                    "source": "team_stats_predictions",
                    "match_id": row.get("match_id"),
                    "market": row.get("market"),
                    "scope": "team",
                    "side": row.get("team"),
                    "line": "",
                    "availability": row.get("availability"),
                    "data_quality_flag": row.get("confidence"),
                    "reason": row.get("warnings", ""),
                }
            )
    if not player_event_predictions.empty and "confidence_flag" in player_event_predictions.columns:
        unavailable = player_event_predictions[player_event_predictions["confidence_flag"].astype(str).str.contains("unavailable|blocked|zero|unresolved", case=False, na=False)]
        for _, row in unavailable.head(40).iterrows():
            rows.append(
                {
                    "source": "player_event_predictions",
                    "match_id": row.get("match_id"),
                    "market": row.get("market"),
                    "scope": "player",
                    "side": row.get("player"),
                    "line": row.get("line"),
                    "availability": row.get("confidence_flag"),
                    "data_quality_flag": row.get("player_selection_confidence"),
                    "reason": row.get("warnings", row.get("candidate_reason", "")),
                }
            )
    if not rows:
        rows.append(
            {
                "source": "audit",
                "match_id": "",
                "market": "",
                "scope": "",
                "side": "",
                "line": "",
                "availability": "none_detected_in_current_sample",
                "data_quality_flag": "not_applicable",
                "reason": "All generated sample rows are available; audit warnings still list unsupported markets such as corners/saves when source data is absent.",
            }
        )
    return pd.DataFrame(rows)


def _build_data_quality_table(
    match_predictions: pd.DataFrame,
    team_stats_predictions: pd.DataFrame,
    player_event_predictions: pd.DataFrame,
    dynamic_market_lines: pd.DataFrame,
    audit: dict[str, Any] | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"area": "audit", "metric": "run_status", "value": audit.get("status", "unknown") if audit else "unknown", "note": ""})
    rows.append({"area": "audit", "metric": "warnings_count", "value": len(audit.get("warnings", [])) if audit else 0, "note": "; ".join(map(str, audit.get("warnings", []))) if audit else ""})
    rows.append({"area": "matches", "metric": "rows", "value": len(match_predictions), "note": ""})
    rows.append({"area": "team_stats", "metric": "rows", "value": len(team_stats_predictions), "note": _value_counts_note(team_stats_predictions, "availability")})
    rows.append({"area": "player_stats", "metric": "rows", "value": len(player_event_predictions), "note": _value_counts_note(player_event_predictions, "confidence_flag")})
    rows.append({"area": "dynamic_lines", "metric": "rows", "value": len(dynamic_market_lines), "note": _value_counts_note(dynamic_market_lines, "availability")})
    rows.append({"area": "dynamic_lines", "metric": "data_quality_flags", "value": "", "note": _value_counts_note(dynamic_market_lines, "data_quality_flag")})
    return pd.DataFrame(rows)


def _match_quality_summary(
    row: pd.Series,
    dynamic_market_lines: pd.DataFrame,
    team_stats_predictions: pd.DataFrame,
    player_event_predictions: pd.DataFrame,
    match_id: str,
) -> pd.DataFrame:
    rows = [
        {"area": "match_model", "metric": "warnings", "value": row.get("warnings", "")},
        {"area": "match_model", "metric": "model_type", "value": row.get("model_type", "")},
    ]
    for frame_name, frame, column in [
        ("team_stats", team_stats_predictions, "availability"),
        ("player_stats", player_event_predictions, "confidence_flag"),
        ("dynamic_lines", dynamic_market_lines, "data_quality_flag"),
    ]:
        if frame.empty or "match_id" not in frame.columns:
            rows.append({"area": frame_name, "metric": "rows", "value": 0})
            continue
        scoped = frame[frame["match_id"].astype(str).eq(match_id)]
        rows.append({"area": frame_name, "metric": "rows", "value": len(scoped)})
        if column in scoped.columns:
            rows.append({"area": frame_name, "metric": column, "value": _value_counts_note(scoped, column)})
    return pd.DataFrame(rows)


def _dynamic_line_overview(dynamic_market_lines: pd.DataFrame) -> pd.DataFrame:
    if dynamic_market_lines.empty:
        return pd.DataFrame()
    df = dynamic_market_lines.copy()
    if "availability" in df.columns:
        df = df[df["availability"].astype(str).eq("available")].copy()
    if "model_probability" in df.columns:
        df["_probability_sort"] = (pd.to_numeric(df["model_probability"], errors="coerce") - 0.5).abs()
        df = df.sort_values("_probability_sort", ascending=False)
    return _select_columns(
        df,
        [
            "match_id",
            "market",
            "scope",
            "side",
            "team",
            "player",
            "line",
            "over_under",
            "model_probability",
            "fair_odds",
            "expected_stat",
            "signal_label",
            "value_label",
            "availability",
            "data_quality_flag",
            "evidence_tags",
            "reason_code",
        ],
    )


def _value_counts_note(df: pd.DataFrame, column: str) -> str:
    if df is None or df.empty or column not in df.columns:
        return ""
    counts = df[column].fillna("missing").astype(str).value_counts().head(8)
    return "; ".join(f"{key}={int(value)}" for key, value in counts.items())


def _to_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
