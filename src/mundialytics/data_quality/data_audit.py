
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import html
import json
import math
import unicodedata

import pandas as pd

from .entity_guardrails import build_entity_squad_guardrails


DATA_AUDIT_VERSION = "v0.49.0_data_audit_report"

DATASET_PROFILES: dict[str, dict[str, Any]] = {
    "fixtures": {
        "required_columns": ["match_id", "date", "home_team", "away_team"],
        "key_columns": ["match_id"],
        "date_columns": ["date"],
        "team_columns": ["home_team", "away_team"],
        "player_columns": [],
    },
    "actual_results": {
        "required_columns": ["match_id", "home_goals", "away_goals"],
        "key_columns": ["match_id"],
        "date_columns": ["date"],
        "team_columns": ["home_team", "away_team"],
        "player_columns": [],
    },
    "lineups": {
        "required_columns": ["match_id", "team", "player"],
        "key_columns": ["match_id", "team", "player"],
        "date_columns": ["date"],
        "team_columns": ["team"],
        "player_columns": ["player"],
    },
    "squads": {
        "required_columns": ["team", "player"],
        "key_columns": ["team", "player"],
        "date_columns": [],
        "team_columns": ["team"],
        "player_columns": ["player"],
    },
    "player_events": {
        "required_columns": ["match_id", "team", "player"],
        "key_columns": ["match_id", "team", "player"],
        "date_columns": ["date"],
        "team_columns": ["team", "opponent"],
        "player_columns": ["player"],
    },
    "odds": {
        "required_columns": ["match_id"],
        "key_columns": [],
        "date_columns": ["date", "snapshot_time", "changedAt"],
        "team_columns": ["team", "side", "selection"],
        "player_columns": ["player"],
    },
    "predictions": {
        "required_columns": ["match_id"],
        "key_columns": ["match_id"],
        "date_columns": ["date"],
        "team_columns": ["home_team", "away_team"],
        "player_columns": [],
    },
    "scorelines": {
        "required_columns": ["match_id"],
        "key_columns": [],
        "date_columns": [],
        "team_columns": [],
        "player_columns": [],
    },
    "dynamic_lines": {
        "required_columns": ["match_id", "market", "line"],
        "key_columns": [],
        "date_columns": [],
        "team_columns": ["side", "team"],
        "player_columns": ["player"],
    },
    "matchday_summary": {
        "required_columns": ["ranking_category", "match_id"],
        "key_columns": [],
        "date_columns": [],
        "team_columns": ["home_team", "away_team"],
        "player_columns": [],
    },
    "tournament_simulation": {
        "required_columns": ["team"],
        "key_columns": ["team"],
        "date_columns": [],
        "team_columns": ["team"],
        "player_columns": [],
    },
    "tournament_report": {
        "required_columns": ["report_section", "team"],
        "key_columns": [],
        "date_columns": [],
        "team_columns": ["team"],
        "player_columns": ["player"],
    },
}

DATA_AUDIT_REPORT_COLUMNS: tuple[str, ...] = (
    "dataset",
    "status",
    "input_path",
    "row_count",
    "column_count",
    "required_columns_present",
    "missing_required_columns",
    "duplicate_key_rows",
    "missing_value_cells",
    "missing_value_rate",
    "high_missing_columns",
    "date_min",
    "date_max",
    "unique_matches",
    "unique_teams",
    "unique_players",
    "data_quality_flag",
    "notes",
)

COVERAGE_REPORT_COLUMNS: tuple[str, ...] = (
    "coverage_area",
    "numerator",
    "denominator",
    "coverage_rate",
    "status",
    "notes",
)

DATA_GAPS_COLUMNS: tuple[str, ...] = (
    "dataset",
    "gap_type",
    "severity",
    "status",
    "message",
    "recommendation",
)

ENTITY_QUALITY_COLUMNS: tuple[str, ...] = (
    "entity_type",
    "dataset",
    "entity_value",
    "status",
    "issue",
    "evidence",
    "recommendation",
)

FEATURE_AVAILABILITY_COLUMNS: tuple[str, ...] = (
    "feature",
    "required_datasets",
    "status",
    "data_quality_flag",
    "coverage_rate",
    "notes",
)

NEXT_REQUIREMENTS_COLUMNS: tuple[str, ...] = (
    "area",
    "required_data",
    "minimum_columns",
    "why_it_matters",
    "future_use",
    "priority",
)


@dataclass(frozen=True)
class DataAuditOutputs:
    report: pd.DataFrame
    coverage: pd.DataFrame
    gaps: pd.DataFrame
    entity_quality: pd.DataFrame
    feature_availability: pd.DataFrame
    next_requirements: pd.DataFrame
    entity_guardrails: pd.DataFrame
    squad_guardrails: pd.DataFrame
    guardrail_summary: dict[str, Any]
    summary: dict[str, Any]


def audit_data_sources(
    *,
    sources: dict[str, pd.DataFrame],
    source_paths: dict[str, str] | None = None,
    run_label: str = "data_audit",
) -> DataAuditOutputs:
    """Audit offline data files without changing model behavior.

    The audit checks schema, coverage, entities and high-level feature
    availability. It intentionally avoids external API calls and does not infer
    or fabricate missing data.
    """
    source_paths = source_paths or {}
    normalized_sources = {name: _ensure_frame(frame) for name, frame in sources.items()}

    report_rows = [
        _audit_single_dataset(dataset=name, df=frame, input_path=source_paths.get(name, ""))
        for name, frame in normalized_sources.items()
    ]

    for known_dataset in DATASET_PROFILES:
        if known_dataset not in normalized_sources:
            report_rows.append(_missing_dataset_row(known_dataset, source_paths.get(known_dataset, "")))

    report = pd.DataFrame(report_rows, columns=DATA_AUDIT_REPORT_COLUMNS)
    coverage = _build_coverage_report(normalized_sources)
    gaps = _build_data_gaps(report=report, coverage=coverage, sources=normalized_sources)
    entity_quality = _build_entity_quality_report(normalized_sources)
    feature_availability = _build_feature_availability_matrix(report=report, coverage=coverage, sources=normalized_sources)
    next_requirements = _build_next_requirements()
    guardrails = build_entity_squad_guardrails(normalized_sources)

    status = _overall_status(report, gaps, guardrails.summary)
    gap_warnings = set(gaps.loc[gaps["severity"].isin(["warning", "critical"]), "gap_type"].astype(str).tolist()) if not gaps.empty else set()
    guardrail_warnings = set(guardrails.summary.get("reason_codes", []))
    warnings = sorted(gap_warnings | guardrail_warnings)

    summary: dict[str, Any] = {
        "version": DATA_AUDIT_VERSION,
        "run_label": run_label,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "principles": {
            "offline_only": True,
            "model_logic_changed": False,
            "external_api_calls": False,
            "missing_data_policy": "not_available",
            "player_props_conservative": True,
        },
        "datasets_supplied": sorted([name for name, frame in normalized_sources.items() if not frame.empty]),
        "dataset_count": int(sum(1 for frame in normalized_sources.values() if not frame.empty)),
        "warnings": warnings,
        "summary_counts": {
            "dataset_rows": int(len(report)),
            "coverage_rows": int(len(coverage)),
            "gap_rows": int(len(gaps)),
            "entity_quality_rows": int(len(entity_quality)),
            "feature_availability_rows": int(len(feature_availability)),
            "entity_guardrail_rows": int(len(guardrails.entity_guardrails)),
            "squad_guardrail_rows": int(len(guardrails.squad_guardrails)),
        },
        "guardrails": guardrails.summary,
        "next_phase": {
            "recommended_version": "v0.49.2_dataset_foundation",
            "goal": "collect and normalize real historical/current data before model changes or visual dashboards",
            "required_docs": ["docs/NEXT_DATA_FOUNDATION_REQUIREMENTS.md"],
        },
    }

    return DataAuditOutputs(
        report=report,
        coverage=coverage,
        gaps=gaps,
        entity_quality=entity_quality,
        feature_availability=feature_availability,
        next_requirements=next_requirements,
        entity_guardrails=guardrails.entity_guardrails,
        squad_guardrails=guardrails.squad_guardrails,
        guardrail_summary=guardrails.summary,
        summary=summary,
    )


def write_data_audit_outputs(outputs: DataAuditOutputs, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "data_audit_report.csv": out / "data_audit_report.csv",
        "coverage_report.csv": out / "coverage_report.csv",
        "data_gaps_report.csv": out / "data_gaps_report.csv",
        "entity_quality_report.csv": out / "entity_quality_report.csv",
        "feature_availability_matrix.csv": out / "feature_availability_matrix.csv",
        "next_data_requirements.csv": out / "next_data_requirements.csv",
        "entity_guardrails_report.csv": out / "entity_guardrails_report.csv",
        "squad_guardrails_report.csv": out / "squad_guardrails_report.csv",
        "guardrail_summary.json": out / "guardrail_summary.json",
        "data_audit_summary.json": out / "data_audit_summary.json",
        "data_audit_report.html": out / "data_audit_report.html",
    }

    outputs.report.to_csv(paths["data_audit_report.csv"], index=False)
    outputs.coverage.to_csv(paths["coverage_report.csv"], index=False)
    outputs.gaps.to_csv(paths["data_gaps_report.csv"], index=False)
    outputs.entity_quality.to_csv(paths["entity_quality_report.csv"], index=False)
    outputs.feature_availability.to_csv(paths["feature_availability_matrix.csv"], index=False)
    outputs.next_requirements.to_csv(paths["next_data_requirements.csv"], index=False)
    outputs.entity_guardrails.to_csv(paths["entity_guardrails_report.csv"], index=False)
    outputs.squad_guardrails.to_csv(paths["squad_guardrails_report.csv"], index=False)
    paths["guardrail_summary.json"].write_text(json.dumps(outputs.guardrail_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["data_audit_summary.json"].write_text(json.dumps(outputs.summary, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["data_audit_report.html"].write_text(render_data_audit_html(outputs), encoding="utf-8")

    return {name: str(path) for name, path in paths.items()}


def render_data_audit_html(outputs: DataAuditOutputs) -> str:
    summary = outputs.summary
    status = _escape(summary.get("status", "unknown"))
    warnings = summary.get("warnings", [])
    warnings_text = ", ".join(_escape(str(w)) for w in warnings) if warnings else "none"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Mundialytics Data Audit Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; }}
h1, h2 {{ color: #102a43; }}
.card {{ border: 1px solid #d9e2ec; border-radius: 10px; padding: 16px; margin: 16px 0; background: #f8fafc; }}
.status-ok {{ color: #0b6b3a; font-weight: 700; }}
.status-warning {{ color: #9a5b00; font-weight: 700; }}
.status-critical {{ color: #9b1c1c; font-weight: 700; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
th, td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #edf2f7; }}
.badge {{ display: inline-block; padding: 3px 7px; border-radius: 999px; background: #e6f0ff; margin: 2px; font-size: 12px; }}
.small {{ color: #52606d; font-size: 12px; }}
</style>
</head>
<body>
<h1>Mundialytics Data Audit Report</h1>
<div class="card">
  <p><strong>Version:</strong> {_escape(str(summary.get("version", DATA_AUDIT_VERSION)))}</p>
  <p><strong>Status:</strong> <span class="{_status_class(status)}">{status}</span></p>
  <p><strong>Run label:</strong> {_escape(str(summary.get("run_label", "")))}</p>
  <p><strong>Run timestamp UTC:</strong> {_escape(str(summary.get("run_timestamp_utc", "")))}</p>
  <p><strong>Warnings:</strong> {warnings_text}</p>
  <p class="small">Offline-only audit. No model logic, API calls, odds collection, player-prop modeling or betting recommendations are performed here.</p>
</div>

<h2>Dataset Health</h2>
{_table_html(outputs.report.head(30))}

<h2>Coverage Report</h2>
{_table_html(outputs.coverage)}

<h2>Data Gaps</h2>
{_table_html(outputs.gaps.head(50))}

<h2>Entity Quality</h2>
{_table_html(outputs.entity_quality.head(50))}

<h2>Entity Guardrails</h2>
{_table_html(outputs.entity_guardrails.head(80))}

<h2>Squad Guardrails</h2>
{_table_html(outputs.squad_guardrails.head(80))}

<h2>Feature Availability Matrix</h2>
{_table_html(outputs.feature_availability)}

<h2>Next Data Requirements</h2>
{_table_html(outputs.next_requirements)}

<div class="card">
  <h2>Policy Notes</h2>
  <p>Missing or unreliable datasets must be surfaced as <code>not_available</code>, not replaced with invented data.</p>
  <p>Player props and Golden Boot development require current squads, expected minutes and reliable goal/event histories before live/current inference.</p>
</div>
</body>
</html>
"""


def _audit_single_dataset(*, dataset: str, df: pd.DataFrame, input_path: str) -> dict[str, Any]:
    profile = DATASET_PROFILES.get(dataset, {})
    required = list(profile.get("required_columns", []))
    key_columns = [c for c in profile.get("key_columns", []) if c in df.columns]
    date_columns = [c for c in profile.get("date_columns", []) if c in df.columns]
    team_columns = [c for c in profile.get("team_columns", []) if c in df.columns]
    player_columns = [c for c in profile.get("player_columns", []) if c in df.columns]

    if df.empty:
        return {
            "dataset": dataset,
            "status": "not_available",
            "input_path": input_path,
            "row_count": 0,
            "column_count": 0,
            "required_columns_present": False,
            "missing_required_columns": ",".join(required),
            "duplicate_key_rows": 0,
            "missing_value_cells": 0,
            "missing_value_rate": 0.0,
            "high_missing_columns": "",
            "date_min": "",
            "date_max": "",
            "unique_matches": 0,
            "unique_teams": 0,
            "unique_players": 0,
            "data_quality_flag": "not_available",
            "notes": "dataset was not supplied or is empty",
        }

    missing_required = [c for c in required if c not in df.columns]
    duplicate_key_rows = 0
    if key_columns:
        duplicate_key_rows = int(df.duplicated(subset=key_columns, keep=False).sum())

    missing_cells = int(df.isna().sum().sum())
    total_cells = int(max(len(df) * max(len(df.columns), 1), 1))
    missing_rate = round(float(missing_cells / total_cells), 6)

    high_missing_columns = []
    for col in df.columns:
        rate = float(df[col].isna().mean()) if len(df) else 0.0
        if rate >= 0.5:
            high_missing_columns.append(f"{col}:{rate:.2f}")

    parsed_dates = []
    for col in date_columns:
        series = pd.to_datetime(df[col], errors="coerce", utc=True)
        parsed_dates.append(series)

    if parsed_dates:
        combined = pd.concat(parsed_dates, ignore_index=True).dropna()
        date_min = combined.min().date().isoformat() if not combined.empty else ""
        date_max = combined.max().date().isoformat() if not combined.empty else ""
    else:
        date_min = ""
        date_max = ""

    unique_matches = _unique_count_from_columns(df, ["match_id", "fixture_id"])
    unique_teams = _unique_entity_count(df, team_columns)
    unique_players = _unique_entity_count(df, player_columns)

    quality_flag = _dataset_quality_flag(
        row_count=len(df),
        missing_required=missing_required,
        duplicate_key_rows=duplicate_key_rows,
        missing_value_rate=missing_rate,
    )
    status = "ok" if quality_flag in {"high", "medium"} else "warning"

    notes = []
    if missing_required:
        notes.append("missing_required_columns")
    if duplicate_key_rows:
        notes.append("duplicate_key_rows")
    if high_missing_columns:
        notes.append("high_missing_columns_detected")
    if not notes:
        notes.append("basic_schema_ok")

    return {
        "dataset": dataset,
        "status": status,
        "input_path": input_path,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "required_columns_present": not missing_required,
        "missing_required_columns": ",".join(missing_required),
        "duplicate_key_rows": duplicate_key_rows,
        "missing_value_cells": missing_cells,
        "missing_value_rate": missing_rate,
        "high_missing_columns": ";".join(high_missing_columns),
        "date_min": date_min,
        "date_max": date_max,
        "unique_matches": unique_matches,
        "unique_teams": unique_teams,
        "unique_players": unique_players,
        "data_quality_flag": quality_flag,
        "notes": "|".join(notes),
    }


def _missing_dataset_row(dataset: str, input_path: str) -> dict[str, Any]:
    required = DATASET_PROFILES.get(dataset, {}).get("required_columns", [])
    return {
        "dataset": dataset,
        "status": "not_available",
        "input_path": input_path,
        "row_count": 0,
        "column_count": 0,
        "required_columns_present": False,
        "missing_required_columns": ",".join(required),
        "duplicate_key_rows": 0,
        "missing_value_cells": 0,
        "missing_value_rate": 0.0,
        "high_missing_columns": "",
        "date_min": "",
        "date_max": "",
        "unique_matches": 0,
        "unique_teams": 0,
        "unique_players": 0,
        "data_quality_flag": "not_available",
        "notes": "dataset was not supplied",
    }


def _build_coverage_report(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fixtures = sources.get("fixtures", pd.DataFrame())
    fixture_match_ids = _normalized_set(fixtures.get("match_id")) if "match_id" in fixtures.columns else set()
    fixture_teams = _team_set_from_fixtures(fixtures)

    rows: list[dict[str, Any]] = []
    if fixture_match_ids:
        for dataset in ["actual_results", "lineups", "odds", "predictions", "scorelines", "dynamic_lines", "matchday_summary"]:
            df = sources.get(dataset, pd.DataFrame())
            ids = _normalized_set(df.get("match_id")) if "match_id" in df.columns else set()
            rows.append(_coverage_row(
                coverage_area=f"{dataset}_match_coverage_vs_fixtures",
                numerator=len(ids & fixture_match_ids),
                denominator=len(fixture_match_ids),
                notes=f"matches in {dataset} with match_id also present in fixtures",
            ))

        squads = sources.get("squads", pd.DataFrame())
        squad_teams = _normalized_set(squads.get("team")) if "team" in squads.columns else set()
        rows.append(_coverage_row(
            coverage_area="squads_team_coverage_vs_fixture_teams",
            numerator=len(squad_teams & fixture_teams),
            denominator=len(fixture_teams),
            notes="fixture teams with at least one squad row",
        ))

        lineups = sources.get("lineups", pd.DataFrame())
        lineup_teams = _normalized_set(lineups.get("team")) if "team" in lineups.columns else set()
        rows.append(_coverage_row(
            coverage_area="lineups_team_coverage_vs_fixture_teams",
            numerator=len(lineup_teams & fixture_teams),
            denominator=len(fixture_teams),
            notes="fixture teams with at least one lineup row",
        ))

    lineups = sources.get("lineups", pd.DataFrame())
    squads = sources.get("squads", pd.DataFrame())
    if not lineups.empty and not squads.empty and {"team", "player"}.issubset(lineups.columns) and {"team", "player"}.issubset(squads.columns):
        lineup_pairs = _entity_pairs(lineups, "team", "player")
        current_squads = squads.copy()
        if "status" in current_squads.columns:
            current_squads = current_squads[current_squads["status"].astype(str).str.lower().str.strip().eq("current")]
        squad_pairs = _entity_pairs(current_squads, "team", "player")
        rows.append(_coverage_row(
            coverage_area="current_squad_player_coverage_for_lineups",
            numerator=len(lineup_pairs & squad_pairs),
            denominator=len(lineup_pairs),
            notes="lineup players found in current squad rows",
        ))

    if not rows:
        rows.append(_coverage_row("fixture_coverage_not_available", 0, 0, "fixtures not supplied or missing match_id"))

    return pd.DataFrame(rows, columns=COVERAGE_REPORT_COLUMNS)


def _build_data_gaps(*, report: pd.DataFrame, coverage: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in report.iterrows():
        dataset = str(r["dataset"])
        if str(r["status"]) == "not_available":
            severity = _missing_dataset_severity(dataset)
            rows.append(_gap(
                dataset=dataset,
                gap_type=f"{dataset}_not_available",
                severity=severity,
                status="not_available",
                message=f"{dataset} was not supplied.",
                recommendation=_dataset_recommendation(dataset),
            ))
        missing_required = str(r.get("missing_required_columns", "") or "")
        if missing_required and str(r["status"]) != "not_available":
            rows.append(_gap(
                dataset=dataset,
                gap_type="missing_required_columns",
                severity="critical",
                status="schema_gap",
                message=f"{dataset} is missing required columns: {missing_required}",
                recommendation="Add the missing columns or map provider columns before using this dataset.",
            ))
        if int(r.get("duplicate_key_rows", 0) or 0) > 0:
            rows.append(_gap(
                dataset=dataset,
                gap_type="duplicate_key_rows",
                severity="warning",
                status="quality_gap",
                message=f"{dataset} has {int(r['duplicate_key_rows'])} duplicate rows on configured key columns.",
                recommendation="Deduplicate or add a deterministic provider/source key before model training/evaluation.",
            ))

    for _, r in coverage.iterrows():
        rate = _safe_float(r.get("coverage_rate"))
        if not math.isnan(rate) and rate < 0.8 and int(r.get("denominator", 0) or 0) > 0:
            severity = "critical" if rate < 0.5 else "warning"
            rows.append(_gap(
                dataset=str(r["coverage_area"]),
                gap_type="low_coverage",
                severity=severity,
                status="coverage_gap",
                message=f"{r['coverage_area']} is {rate:.1%}.",
                recommendation="Collect, normalize or join the missing rows before relying on this segment.",
            ))

    # Golden Boot / player-prop future guardrail.
    player_events = sources.get("player_events", pd.DataFrame())
    squads = sources.get("squads", pd.DataFrame())
    if player_events.empty:
        rows.append(_gap(
            dataset="player_events",
            gap_type="golden_boot_inputs_not_available",
            severity="warning",
            status="future_requirement",
            message="Golden Boot and scorer projections need reliable player scoring/event history.",
            recommendation="Collect player_id, team, match_id, minutes, goals, shots and expected minutes/progression inputs.",
        ))
    if squads.empty:
        rows.append(_gap(
            dataset="squads",
            gap_type="current_squad_guardrail_not_available",
            severity="warning",
            status="future_requirement",
            message="Current squad status is required before current player-prop or award inference.",
            recommendation="Maintain current squads with player_id, status, team and expected_minutes.",
        ))

    if not rows:
        rows.append(_gap(
            dataset="all",
            gap_type="no_major_gaps_detected",
            severity="info",
            status="ok",
            message="No major schema or coverage gaps detected by basic audit.",
            recommendation="Proceed to deeper provider/entity validation before model changes.",
        ))

    return pd.DataFrame(rows, columns=DATA_GAPS_COLUMNS)


def _build_entity_quality_report(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fixtures = sources.get("fixtures", pd.DataFrame())
    fixture_teams = _team_set_from_fixtures(fixtures)

    for dataset, team_col in [("lineups", "team"), ("squads", "team"), ("player_events", "team"), ("actual_results", "home_team")]:
        df = sources.get(dataset, pd.DataFrame())
        if df.empty:
            continue
        team_values: set[str] = set()
        for col in ["home_team", "away_team", "team", "opponent"]:
            if col in df.columns:
                team_values |= _normalized_set(df[col])
        if fixture_teams:
            unknown = sorted(team_values - fixture_teams)
            for team in unknown[:50]:
                rows.append(_entity_issue(
                    entity_type="team",
                    dataset=dataset,
                    entity_value=team,
                    status="review_required",
                    issue="team_not_in_current_fixtures",
                    evidence="team appears in dataset but not in supplied fixtures",
                    recommendation="If this is historical training data, keep it separate from current inference coverage checks.",
                ))

    lineups = sources.get("lineups", pd.DataFrame())
    squads = sources.get("squads", pd.DataFrame())
    if not lineups.empty and not squads.empty and {"team", "player"}.issubset(lineups.columns) and {"team", "player"}.issubset(squads.columns):
        lineup_pairs = _entity_pairs(lineups, "team", "player")
        current_squads = squads.copy()
        if "status" in current_squads.columns:
            current_squads = current_squads[current_squads["status"].astype(str).str.lower().str.strip().eq("current")]
        squad_pairs = _entity_pairs(current_squads, "team", "player")
        missing_pairs = sorted(lineup_pairs - squad_pairs)
        for team, player in missing_pairs[:100]:
            rows.append(_entity_issue(
                entity_type="player",
                dataset="lineups",
                entity_value=player,
                status="blocked_for_current_player_props",
                issue="lineup_player_not_in_current_squad",
                evidence=f"team={team}",
                recommendation="Resolve aliases or add current squad row before current player-prop inference.",
            ))

    if not squads.empty and "status" in squads.columns and {"team", "player"}.issubset(squads.columns):
        inactive = squads[~squads["status"].astype(str).str.lower().str.strip().eq("current")]
        for _, r in inactive.head(100).iterrows():
            rows.append(_entity_issue(
                entity_type="player",
                dataset="squads",
                entity_value=_norm_text(r.get("player")),
                status="not_current",
                issue="squad_player_status_not_current",
                evidence=f"team={_norm_text(r.get('team'))}; status={r.get('status')}",
                recommendation="Exclude from current player-prop candidates unless current input confirms eligibility.",
            ))

    if not rows:
        rows.append(_entity_issue(
            entity_type="all",
            dataset="all",
            entity_value="",
            status="ok",
            issue="no_entity_guardrail_failures_detected",
            evidence="basic checks only",
            recommendation="Proceed to provider IDs and alias resolution in the next data phase.",
        ))

    return pd.DataFrame(rows, columns=ENTITY_QUALITY_COLUMNS)


def _build_feature_availability_matrix(*, report: pd.DataFrame, coverage: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    def dataset_status(dataset: str) -> str:
        rows = report[report["dataset"].eq(dataset)]
        if rows.empty:
            return "not_available"
        return str(rows.iloc[0]["data_quality_flag"])

    def coverage_rate(area: str) -> float:
        rows = coverage[coverage["coverage_area"].eq(area)]
        if rows.empty:
            return float("nan")
        return _safe_float(rows.iloc[0]["coverage_rate"])

    features = [
        ("1x2_result_evaluation", ["predictions", "actual_results"], "needs predictions and finished results"),
        ("goal_error_evaluation", ["predictions", "actual_results"], "needs expected goals and actual goals"),
        ("scoreline_evaluation", ["scorelines", "actual_results"], "needs scoreline distribution and actual scoreline"),
        ("dynamic_goal_lines_evaluation", ["dynamic_lines", "actual_results"], "needs dynamic goal lines and actual goals"),
        ("matchday_prediction_report", ["fixtures", "predictions"], "needs fixtures and predictions"),
        ("tournament_simulation_report", ["fixtures", "tournament_simulation"], "needs fixtures and tournament simulation output"),
        ("team_event_stats", ["player_events"], "needs reliable event/team-stat history"),
        ("player_props_current_inference", ["lineups", "squads", "player_events"], "needs current lineups/squads and player history"),
        ("golden_boot_projection", ["squads", "player_events", "tournament_simulation"], "needs current squad eligibility, scoring history and team progression probabilities"),
        ("value_betting_future_layer", ["predictions", "odds"], "needs predictions and reliable odds snapshots"),
    ]

    rows: list[dict[str, Any]] = []
    for feature, datasets, notes in features:
        flags = [dataset_status(d) for d in datasets]
        available = [f not in {"not_available", "low"} for f in flags]
        if all(available):
            status = "available"
            flag = "medium" if "medium" in flags else "high"
        elif any(f != "not_available" for f in flags):
            status = "partial"
            flag = "low"
        else:
            status = "not_available"
            flag = "not_available"

        relevant_rates = []
        for area in [
            "actual_results_match_coverage_vs_fixtures",
            "lineups_match_coverage_vs_fixtures",
            "odds_match_coverage_vs_fixtures",
            "predictions_match_coverage_vs_fixtures",
            "dynamic_lines_match_coverage_vs_fixtures",
            "current_squad_player_coverage_for_lineups",
        ]:
            rate = coverage_rate(area)
            if not math.isnan(rate):
                relevant_rates.append(rate)
        avg_rate = round(float(sum(relevant_rates) / len(relevant_rates)), 6) if relevant_rates else float("nan")

        rows.append({
            "feature": feature,
            "required_datasets": ",".join(datasets),
            "status": status,
            "data_quality_flag": flag,
            "coverage_rate": avg_rate if not math.isnan(avg_rate) else "",
            "notes": notes,
        })

    return pd.DataFrame(rows, columns=FEATURE_AVAILABILITY_COLUMNS)


def _build_next_requirements() -> pd.DataFrame:
    rows = [
        {
            "area": "match_results",
            "required_data": "Finished match results",
            "minimum_columns": "match_id,date,competition,home_team,away_team,home_goals,away_goals,status,source",
            "why_it_matters": "Needed for 1X2, goal, line and scoreline evaluation.",
            "future_use": "retrospective_backtest and forward_evaluation",
            "priority": "critical",
        },
        {
            "area": "forward_prediction_snapshots",
            "required_data": "Predictions saved before kickoff",
            "minimum_columns": "run_id,created_at_utc,match_id,p_home_win,p_draw,p_away_win,expected_home_goals,expected_away_goals,model_version,data_version",
            "why_it_matters": "Avoids leakage and separates real forward evaluation from retrospective backtests.",
            "future_use": "calibration, model selection, paper-value audit",
            "priority": "critical",
        },
        {
            "area": "team_event_stats",
            "required_data": "Team-level match stats",
            "minimum_columns": "match_id,team,opponent,date,shots,shots_on_target,corners,fouls,yellow_cards,red_cards",
            "why_it_matters": "Required for team stat simulations and dynamic lines beyond goals.",
            "future_use": "team props, match reports, feature availability",
            "priority": "high",
        },
        {
            "area": "player_events",
            "required_data": "Player match/event history",
            "minimum_columns": "player_id,player,team,match_id,date,minutes,goals,shots,shots_on_target,fouls_committed,fouls_drawn,yellow_cards,assists",
            "why_it_matters": "Required for player props and awards such as Golden Boot.",
            "future_use": "player-prop models, scorer projections, awards simulator",
            "priority": "high",
        },
        {
            "area": "current_squads_lineups",
            "required_data": "Current squads and expected lineups",
            "minimum_columns": "team,player_id,player,status,position,expected_minutes,current_squad_flag",
            "why_it_matters": "Prevents retired/historical players from appearing in current player predictions.",
            "future_use": "player-prop inference and Golden Boot eligibility",
            "priority": "high",
        },
        {
            "area": "provider_identity",
            "required_data": "Stable team/player/provider IDs and aliases",
            "minimum_columns": "provider,provider_entity_id,provider_entity_name,canonical_entity_id,canonical_name,match_confidence,status",
            "why_it_matters": "Reduces duplicates and failed joins across providers.",
            "future_use": "data ingestion, player props, odds joins",
            "priority": "high",
        },
        {
            "area": "odds_snapshots",
            "required_data": "Optional current/forward odds snapshots",
            "minimum_columns": "snapshot_time_utc,fixture_id,match_id,bookmaker,market,outcome,line,price,active",
            "why_it_matters": "Needed later for edge, EV, CLV and paper tracking.",
            "future_use": "value betting layer",
            "priority": "medium",
        },
    ]
    return pd.DataFrame(rows, columns=NEXT_REQUIREMENTS_COLUMNS)


def _overall_status(report: pd.DataFrame, gaps: pd.DataFrame, guardrail_summary: dict[str, Any] | None = None) -> str:
    guardrail_status = str((guardrail_summary or {}).get("status", "ok"))
    if guardrail_status == "blocked":
        return "warning"
    if not gaps.empty and gaps["severity"].eq("critical").any():
        return "warning"
    supplied = report[report["status"].ne("not_available")]
    if supplied.empty:
        return "not_available"
    if not gaps.empty and gaps["severity"].eq("warning").any():
        return "warning"
    return "ok"


def _dataset_quality_flag(*, row_count: int, missing_required: list[str], duplicate_key_rows: int, missing_value_rate: float) -> str:
    if row_count <= 0:
        return "not_available"
    if missing_required:
        return "low"
    if duplicate_key_rows > 0:
        return "low"
    if missing_value_rate >= 0.05:
        return "medium"
    return "high"


def _missing_dataset_severity(dataset: str) -> str:
    if dataset in {"fixtures"}:
        return "critical"
    if dataset in {"actual_results", "predictions", "lineups", "squads", "player_events"}:
        return "warning"
    return "info"


def _dataset_recommendation(dataset: str) -> str:
    recommendations = {
        "fixtures": "Provide fixtures with match_id, date, home_team and away_team before running simulator audits.",
        "actual_results": "Provide finished results to evaluate model quality; otherwise metrics remain not_available.",
        "lineups": "Provide current lineups to support player-current eligibility and player prop inference.",
        "squads": "Provide current squads with player status/current flags for player guardrails.",
        "player_events": "Provide historical player events for player props, Golden Boot and team-stat features.",
        "odds": "Provide odds snapshots only when evaluating future value betting; not required for simulator core.",
    }
    return recommendations.get(dataset, "Provide this dataset when the corresponding feature is in scope.")


def _coverage_row(coverage_area: str, numerator: int, denominator: int, notes: str) -> dict[str, Any]:
    if denominator <= 0:
        rate: Any = ""
        status = "not_available"
    else:
        rate_value = numerator / denominator
        rate = round(float(rate_value), 6)
        status = "ok" if rate_value >= 0.8 else "warning" if rate_value >= 0.5 else "low"
    return {
        "coverage_area": coverage_area,
        "numerator": int(numerator),
        "denominator": int(denominator),
        "coverage_rate": rate,
        "status": status,
        "notes": notes,
    }


def _gap(dataset: str, gap_type: str, severity: str, status: str, message: str, recommendation: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "gap_type": gap_type,
        "severity": severity,
        "status": status,
        "message": message,
        "recommendation": recommendation,
    }


def _entity_issue(
    *,
    entity_type: str,
    dataset: str,
    entity_value: str,
    status: str,
    issue: str,
    evidence: str,
    recommendation: str,
) -> dict[str, Any]:
    return {
        "entity_type": entity_type,
        "dataset": dataset,
        "entity_value": entity_value,
        "status": status,
        "issue": issue,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _ensure_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    return pd.DataFrame(frame)


def _unique_count_from_columns(df: pd.DataFrame, columns: list[str]) -> int:
    for col in columns:
        if col in df.columns:
            return int(df[col].dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique())
    return 0


def _unique_entity_count(df: pd.DataFrame, columns: list[str]) -> int:
    values: set[str] = set()
    for col in columns:
        if col in df.columns:
            values |= _normalized_set(df[col])
    return len(values)


def _team_set_from_fixtures(fixtures: pd.DataFrame) -> set[str]:
    if fixtures.empty:
        return set()
    values: set[str] = set()
    for col in ["home_team", "away_team", "team"]:
        if col in fixtures.columns:
            values |= _normalized_set(fixtures[col])
    return values


def _normalized_set(series: pd.Series | None) -> set[str]:
    if series is None:
        return set()
    return {_norm_text(value) for value in series.dropna().tolist() if _norm_text(value)}


def _entity_pairs(df: pd.DataFrame, first: str, second: str) -> set[tuple[str, str]]:
    if first not in df.columns or second not in df.columns:
        return set()
    pairs = set()
    for _, r in df[[first, second]].dropna().iterrows():
        a = _norm_text(r[first])
        b = _norm_text(r[second])
        if a and b:
            pairs.add((a, b))
    return pairs


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.replace("_", " ").replace("-", " ").split())


def _safe_float(value: Any) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class='small'>No rows.</p>"
    return df.fillna("").to_html(index=False, escape=True)


def _status_class(status: str) -> str:
    if status == "ok":
        return "status-ok"
    if status in {"warning", "partial", "low"}:
        return "status-warning"
    return "status-critical"


def _escape(value: str) -> str:
    return html.escape(str(value))
