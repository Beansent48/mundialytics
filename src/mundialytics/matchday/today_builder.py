from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from mundialytics.data.competition_taxonomy import enrich_competition_metadata
from mundialytics.data.free_fixtures import add_local_kickoff_columns, filter_by_matchday_date
from mundialytics.statistical_core.schemas import canonical_name, stable_match_id, standardize_fixtures

SCHEDULED_STATUS_TOKENS = {
    "notstarted",
    "not started",
    "scheduled",
    "status_scheduled",
    "ns",
    "pre",
    "time tbd",
    "tbd",
}
LIVE_STATUS_TOKENS = {
    "inprogress",
    "in progress",
    "live",
    "1st half",
    "2nd half",
    "first half",
    "second half",
    "halftime",
    "half time",
    "status in progress",
    "status first half",
    "status second half",
    "status halftime",
    "ht",
}
COMPLETED_STATUS_TOKENS = {
    "finished",
    "finish",
    "complete",
    "completed",
    "full time",
    "status full time",
    "final",
    "status final",
    "post",
    "status post",
    "ft",
    "aet",
    "penalties",
}
CANCELLED_STATUS_TOKENS = {"cancelled", "canceled", "postponed", "abandoned", "suspended"}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def _status_bucket(status_short: Any, status_long: Any) -> str:
    text = f"{_norm(status_short)} {_norm(status_long)}".strip()
    if not text:
        return "unknown"
    if any(tok in text for tok in CANCELLED_STATUS_TOKENS):
        return "cancelled"
    if any(tok in text for tok in COMPLETED_STATUS_TOKENS):
        return "completed"
    if any(tok in text for tok in LIVE_STATUS_TOKENS):
        return "live"
    if any(tok in text for tok in SCHEDULED_STATUS_TOKENS):
        return "scheduled"
    return "unknown"


def _parse_stage_group(round_text: Any, competition: Any) -> tuple[str, str]:
    text = f"{round_text or ''} {competition or ''}".strip()
    low = text.lower()
    group = ""
    m = re.search(r"group\s+([a-z0-9]+)", low, flags=re.I)
    if m:
        group = m.group(1).upper()
    if "group" in low:
        stage = "Group"
    elif "round of 16" in low or "last 16" in low:
        stage = "Round of 16"
    elif "quarter" in low:
        stage = "Quarter-final"
    elif "semi" in low:
        stage = "Semi-final"
    elif "final" in low:
        stage = "Final"
    else:
        stage = "Scheduled"
    return stage, group


def build_matchday_fixtures(
    provider_fixtures: pd.DataFrame,
    *,
    local_date: str | None = None,
    timezone: str = "Europe/Madrid",
    date_mode: str = "event_or_user",
    include_live: bool = True,
    include_completed: bool = False,
    include_unknown_status: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert provider fixture rows into run_statistical_matchday-compatible fixtures.

    The function is deliberately conservative: cancelled/postponed games are removed,
    completed games are excluded by default, and all rows keep provider IDs for audit.
    """

    if provider_fixtures is None or provider_fixtures.empty:
        return pd.DataFrame(columns=["match_id", "date", "home_team", "away_team"]), {
            "status": "no_provider_fixtures",
            "rows_raw": 0,
            "rows_final": 0,
            "timezone": timezone,
            "date_mode": date_mode,
            "local_date": local_date,
        }

    work = provider_fixtures.copy()
    rows_raw = len(work)
    if local_date:
        work, local_filter_warning = filter_by_matchday_date(work, local_date=local_date, timezone=timezone, date_mode=date_mode)
    else:
        work = add_local_kickoff_columns(work, timezone=timezone)
        local_filter_warning = None

    rows_after_date = len(work)
    if work.empty:
        return pd.DataFrame(columns=["match_id", "date", "home_team", "away_team"]), {
            "status": "no_fixtures_after_date_filter",
            "rows_raw": int(rows_raw),
            "rows_after_date_filter": int(rows_after_date),
            "rows_final": 0,
            "timezone": timezone,
            "date_mode": date_mode,
            "local_date": local_date,
            "local_filter_warning": local_filter_warning,
        }

    work["status_bucket"] = [
        _status_bucket(r.get("status_short"), r.get("status_long")) for _, r in work.iterrows()
    ]
    allowed = {"scheduled"}
    if include_live:
        allowed.add("live")
    if include_completed:
        allowed.add("completed")
    if include_unknown_status:
        allowed.add("unknown")
    before_status = len(work)
    work = work[work["status_bucket"].isin(allowed)].copy()
    rows_removed_by_status = before_status - len(work)

    rows: list[dict[str, Any]] = []
    for _, r in work.iterrows():
        home = canonical_name(r.get("home_team"))
        away = canonical_name(r.get("away_team"))
        if not home or not away:
            continue
        raw_match_id = r.get("match_id") or r.get("fixture_id") or r.get("provider_match_id")
        match_id = str(raw_match_id) if pd.notna(raw_match_id) and str(raw_match_id).strip() else stable_match_id(r.get("date"), home, away)
        stage, group = _parse_stage_group(r.get("round"), r.get("competition"))
        rows.append({
            "match_id": match_id,
            "date": r.get("date") or r.get("kickoff_utc") or r.get("kickoff_local_date") or local_date or "unknown",
            "home_team": home,
            "away_team": away,
            "neutral": int(float(r.get("neutral", 1))) if str(r.get("neutral", "")).strip() not in {"", "None", "nan"} else 1,
            "competition": r.get("competition") or r.get("unique_tournament_name") or r.get("tournament_name") or "unknown",
            "season": r.get("season", "unknown"),
            "stage": stage,
            "group": group,
            "team_scope": r.get("team_scope", "national" if str(r.get("team_type", "")).lower() == "national_team" else "unknown"),
            "team_type": r.get("team_type", "unknown"),
            "competition_context": r.get("competition_context", "unknown"),
            "gender": r.get("gender", "unknown"),
            "provider": r.get("provider", "unknown"),
            "provider_match_id": r.get("provider_match_id", r.get("fixture_id", "")),
            "fixture_id": r.get("fixture_id", r.get("provider_match_id", "")),
            "kickoff_utc": r.get("kickoff_utc", ""),
            "kickoff_local": r.get("kickoff_local", ""),
            "kickoff_local_date": r.get("kickoff_local_date", ""),
            "kickoff_local_time": r.get("kickoff_local_time", ""),
            "kickoff_user": r.get("kickoff_user", ""),
            "kickoff_user_date": r.get("kickoff_user_date", ""),
            "kickoff_user_time": r.get("kickoff_user_time", ""),
            "kickoff_event": r.get("kickoff_event", ""),
            "kickoff_event_date": r.get("kickoff_event_date", ""),
            "kickoff_event_time": r.get("kickoff_event_time", ""),
            "event_timezone": r.get("event_timezone", ""),
            "event_timezone_source": r.get("event_timezone_source", ""),
            "venue_name": r.get("venue_name", ""),
            "venue_city": r.get("venue_city", ""),
            "venue_country": r.get("venue_country", ""),
            "status_short": r.get("status_short", ""),
            "status_long": r.get("status_long", ""),
            "status_bucket": r.get("status_bucket", ""),
        })

    fixtures = pd.DataFrame(rows)
    if not fixtures.empty:
        fixtures = enrich_competition_metadata(fixtures, overwrite=False)
        fixtures = standardize_fixtures(fixtures)
        # Preserve provider/audit columns after standardization.
        for col in ["provider", "provider_match_id", "fixture_id", "kickoff_utc", "kickoff_utc_date", "kickoff_local", "kickoff_local_date", "kickoff_local_time", "kickoff_user", "kickoff_user_date", "kickoff_user_time", "kickoff_event", "kickoff_event_date", "kickoff_event_time", "event_timezone", "event_timezone_source", "venue_name", "venue_city", "venue_country", "status_short", "status_long", "status_bucket"]:
            if col in pd.DataFrame(rows).columns and col not in fixtures.columns:
                fixtures[col] = pd.DataFrame(rows)[col].values

    report = {
        "status": "matchday_fixtures_ready" if not fixtures.empty else "no_fixtures_ready",
        "rows_raw": int(rows_raw),
        "rows_after_date_filter": int(rows_after_date),
        "rows_removed_by_status": int(rows_removed_by_status),
        "rows_final": int(len(fixtures)),
        "timezone": timezone,
        "date_mode": date_mode,
        "local_date": local_date,
        "include_live": bool(include_live),
        "include_completed": bool(include_completed),
        "include_unknown_status": bool(include_unknown_status),
        "local_filter_warning": local_filter_warning,
        "status_counts": {str(k): int(v) for k, v in work.get("status_bucket", pd.Series(dtype=str)).value_counts(dropna=False).to_dict().items()},
    }
    return fixtures.reset_index(drop=True), report


def empty_current_lineups_template() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "match_id", "team", "opponent", "player", "position", "started", "expected_minutes", "status", "source"
    ])


def empty_squads_template() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "team", "player", "position", "started", "expected_minutes", "status", "source"
    ])


def write_matchday_inputs(
    provider_fixtures: pd.DataFrame,
    *,
    out_dir: str | Path,
    local_date: str | None = None,
    timezone: str = "Europe/Madrid",
    date_mode: str = "event_or_user",
    include_live: bool = True,
    include_completed: bool = False,
    include_unknown_status: bool = True,
    write_empty_player_inputs: bool = True,
    source_report: dict[str, Any] | None = None,
    lineups_df: pd.DataFrame | None = None,
    squads_df: pd.DataFrame | None = None,
    player_input_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    provider_path = out / "today_provider_fixtures.csv"
    provider_fixtures.to_csv(provider_path, index=False)
    fixtures, report = build_matchday_fixtures(
        provider_fixtures,
        local_date=local_date,
        timezone=timezone,
        date_mode=date_mode,
        include_live=include_live,
        include_completed=include_completed,
        include_unknown_status=include_unknown_status,
    )
    fixtures_path = out / "today_fixtures.csv"
    fixtures.to_csv(fixtures_path, index=False)
    result: dict[str, Any] = {
        "version": "v0.33_team_identity_event_timezone_player_confidence",
        "status": report.get("status"),
        "fixtures_csv": str(fixtures_path),
        "provider_fixtures_csv": str(provider_path),
        "fixtures_rows": int(len(fixtures)),
        "builder_report": report,
        "source_report": source_report or {},
        "player_input_report": player_input_report or {},
        "notes": [
            "Generated fixtures are run_statistical_matchday-compatible.",
            "Event-local kickoff columns allow US/Canada/Mexico evening games to be selected by event date as well as user-local date.",
            "Player props use lineups first and squads second; squad fallback is lower confidence than confirmed lineups.",
        ],
    }
    lineups_path = out / "today_current_lineups.csv"
    squads_path = out / "today_squads.csv"
    if lineups_df is not None and not lineups_df.empty:
        lineups_df.to_csv(lineups_path, index=False)
        result["lineups_csv"] = str(lineups_path)
        result["lineups_rows"] = int(len(lineups_df))
    elif write_empty_player_inputs:
        empty_current_lineups_template().to_csv(lineups_path, index=False)
        result["lineups_csv"] = str(lineups_path)
        result["lineups_rows"] = 0

    if squads_df is not None and not squads_df.empty:
        squads_df.to_csv(squads_path, index=False)
        result["squads_csv"] = str(squads_path)
        result["squads_rows"] = int(len(squads_df))
    elif write_empty_player_inputs:
        empty_squads_template().to_csv(squads_path, index=False)
        result["squads_csv"] = str(squads_path)
        result["squads_rows"] = 0

    if "lineups_csv" in result or "squads_csv" in result:
        if (result.get("lineups_rows", 0) or result.get("squads_rows", 0)):
            result["player_input_status"] = "lineups_or_squads_written"
        else:
            result["player_input_status"] = "empty_templates_written"
    audit_path = out / "today_matchday_audit.json"
    result["audit_json"] = str(audit_path)
    audit_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return result
