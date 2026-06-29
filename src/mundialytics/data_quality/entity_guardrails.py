
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import unicodedata

import pandas as pd


ENTITY_GUARDRAILS_VERSION = "v0.49.1_entity_squad_guardrails"

GUARDRAIL_COLUMNS: tuple[str, ...] = (
    "guardrail_scope",
    "dataset",
    "record_key",
    "severity",
    "status",
    "reason_code",
    "entity_type",
    "entity_value",
    "evidence",
    "recommendation",
)


@dataclass(frozen=True)
class EntitySquadGuardrailOutputs:
    entity_guardrails: pd.DataFrame
    squad_guardrails: pd.DataFrame
    summary: dict[str, Any]


def build_entity_squad_guardrails(sources: dict[str, pd.DataFrame]) -> EntitySquadGuardrailOutputs:
    """Build conservative offline guardrails for entity and squad safety.

    This function is intentionally pure and offline-only. It does not fetch data,
    change model behavior, infer missing identities or filter records silently.
    Unsafe conditions are reported with reason codes so downstream workflows can
    block or review current inference.
    """

    normalized_sources = {name: _ensure_frame(frame) for name, frame in sources.items()}
    entity_rows: list[dict[str, Any]] = []
    squad_rows: list[dict[str, Any]] = []

    fixtures = normalized_sources.get("fixtures", pd.DataFrame())
    fixture_match_ids = _normalized_set(fixtures.get("match_id")) if "match_id" in fixtures.columns else set()
    fixture_teams = _fixture_team_set(fixtures)
    fixture_scopes = _scope_values(fixtures)

    entity_rows.extend(_match_identity_guardrails(normalized_sources, fixture_match_ids))
    entity_rows.extend(_provider_fixture_guardrails(normalized_sources))
    entity_rows.extend(_team_scope_guardrails(normalized_sources, fixture_scopes))
    entity_rows.extend(_fixture_team_guardrails(normalized_sources, fixture_teams))
    entity_rows.extend(_provider_entity_ambiguity_guardrails(normalized_sources))

    lineups = normalized_sources.get("lineups", pd.DataFrame())
    squads = normalized_sources.get("squads", pd.DataFrame())
    player_events = normalized_sources.get("player_events", pd.DataFrame())

    squad_rows.extend(_lineup_player_id_guardrails(lineups))
    squad_rows.extend(_current_squad_eligibility_guardrails(lineups, squads))
    squad_rows.extend(_historical_only_player_guardrails(player_events, squads, fixture_teams))
    squad_rows.extend(_player_name_ambiguity_guardrails(normalized_sources))

    if not entity_rows:
        entity_rows.append(_guardrail_row(
            guardrail_scope="entity",
            dataset="all",
            record_key="",
            severity="info",
            status="ok",
            reason_code="no_entity_guardrail_failures_detected",
            entity_type="all",
            entity_value="",
            evidence="provider, match and team checks found no issues in supplied data",
            recommendation="Continue collecting stable provider IDs and current eligibility evidence.",
        ))
    if not squad_rows:
        squad_rows.append(_guardrail_row(
            guardrail_scope="squad",
            dataset="all",
            record_key="",
            severity="info",
            status="ok",
            reason_code="no_squad_guardrail_failures_detected",
            entity_type="all",
            entity_value="",
            evidence="lineup and squad checks found no issues in supplied data",
            recommendation="Keep current_squad_flag, availability_status and player_id populated for current inference.",
        ))

    entity_guardrails = pd.DataFrame(entity_rows, columns=GUARDRAIL_COLUMNS)
    squad_guardrails = pd.DataFrame(squad_rows, columns=GUARDRAIL_COLUMNS)
    summary = _build_guardrail_summary(entity_guardrails, squad_guardrails)

    return EntitySquadGuardrailOutputs(
        entity_guardrails=entity_guardrails,
        squad_guardrails=squad_guardrails,
        summary=summary,
    )


def _match_identity_guardrails(sources: dict[str, pd.DataFrame], fixture_match_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    datasets_requiring_match_id = [
        "fixtures",
        "actual_results",
        "lineups",
        "player_events",
        "odds",
        "predictions",
        "scorelines",
        "dynamic_lines",
        "matchday_summary",
    ]

    for dataset in datasets_requiring_match_id:
        df = sources.get(dataset, pd.DataFrame())
        if df.empty:
            continue

        if "match_id" not in df.columns:
            rows.append(_guardrail_row(
                guardrail_scope="entity",
                dataset=dataset,
                record_key="dataset",
                severity="critical",
                status="blocked",
                reason_code="missing_match_id",
                entity_type="match",
                entity_value="",
                evidence=f"{dataset} was supplied but has no match_id column",
                recommendation="Add a stable match_id before joining this dataset with fixtures, odds, events or predictions.",
            ))
            continue

        match_ids = df["match_id"].map(_norm_text)
        missing_count = int(match_ids.eq("").sum())
        if missing_count:
            rows.append(_guardrail_row(
                guardrail_scope="entity",
                dataset=dataset,
                record_key="match_id",
                severity="critical",
                status="blocked",
                reason_code="missing_match_id",
                entity_type="match",
                entity_value="",
                evidence=f"{missing_count} rows have empty match_id",
                recommendation="Populate or remove rows without stable match_id before inference/evaluation.",
            ))

        if dataset == "fixtures":
            duplicate_mask = match_ids.ne("") & match_ids.duplicated(keep=False)
            for match_id in sorted(match_ids[duplicate_mask].unique())[:100]:
                rows.append(_guardrail_row(
                    guardrail_scope="entity",
                    dataset=dataset,
                    record_key=f"match_id={match_id}",
                    severity="critical",
                    status="blocked",
                    reason_code="duplicate_match_id",
                    entity_type="match",
                    entity_value=match_id,
                    evidence="same fixture match_id appears multiple times",
                    recommendation="Deduplicate fixtures or generate a deterministic fixture-level match_id.",
                ))

        if fixture_match_ids and dataset != "fixtures":
            unknown = sorted(set(match_ids[match_ids.ne("")]) - fixture_match_ids)
            for match_id in unknown[:100]:
                rows.append(_guardrail_row(
                    guardrail_scope="entity",
                    dataset=dataset,
                    record_key=f"match_id={match_id}",
                    severity="warning",
                    status="needs_review",
                    reason_code="match_id_not_in_fixtures",
                    entity_type="match",
                    entity_value=match_id,
                    evidence="match_id appears in dataset but not in supplied fixtures",
                    recommendation="Keep historical/training data separate from current inference inputs or add the matching fixture row.",
                ))

    return rows


def _provider_fixture_guardrails(sources: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    provider_fixture_columns = ["provider_fixture_id", "fixture_id"]

    for dataset, df in sources.items():
        if df.empty or "match_id" not in df.columns:
            continue

        for fixture_col in provider_fixture_columns:
            if fixture_col not in df.columns:
                continue

            working_cols = [fixture_col, "match_id"]
            if "provider" in df.columns:
                working_cols.insert(0, "provider")
            working = df[working_cols].copy()
            working["_provider"] = working["provider"].map(_norm_text) if "provider" in working.columns else ""
            working["_fixture"] = working[fixture_col].map(_norm_text)
            working["_match_id"] = working["match_id"].map(_norm_text)
            working = working[working["_fixture"].ne("") & working["_match_id"].ne("")]
            if working.empty:
                continue

            grouped = working.groupby(["_provider", "_fixture"], dropna=False)["_match_id"].nunique()
            conflicts = grouped[grouped > 1]
            for (provider, fixture_id), _count in conflicts.head(100).items():
                affected = sorted(working[(working["_provider"].eq(provider)) & (working["_fixture"].eq(fixture_id))]["_match_id"].unique())
                provider_label = provider or "unknown_provider"
                rows.append(_guardrail_row(
                    guardrail_scope="entity",
                    dataset=dataset,
                    record_key=f"provider={provider_label}; {fixture_col}={fixture_id}",
                    severity="critical",
                    status="blocked",
                    reason_code="provider_fixture_id_conflict",
                    entity_type="match",
                    entity_value=fixture_id,
                    evidence=f"maps to multiple match_id values: {','.join(affected[:10])}",
                    recommendation="Resolve provider fixture identity before joining odds, events, lineups or predictions.",
                ))

    return rows


def _team_scope_guardrails(sources: dict[str, pd.DataFrame], fixture_scopes: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_inference_datasets = ["fixtures", "lineups", "squads", "predictions", "dynamic_lines", "matchday_summary"]

    for dataset in current_inference_datasets:
        df = sources.get(dataset, pd.DataFrame())
        if df.empty:
            continue

        scopes = _scope_values(df)
        if len(scopes) > 1:
            rows.append(_guardrail_row(
                guardrail_scope="entity",
                dataset=dataset,
                record_key="team_scope",
                severity="critical",
                status="blocked",
                reason_code="team_scope_mismatch",
                entity_type="team_scope",
                entity_value=",".join(sorted(scopes)),
                evidence="club and national/current contexts appear in the same current inference dataset",
                recommendation="Separate club and national-team workflows before model inference or reporting.",
            ))

        if fixture_scopes and scopes and not scopes.issubset(fixture_scopes):
            rows.append(_guardrail_row(
                guardrail_scope="entity",
                dataset=dataset,
                record_key="team_scope",
                severity="critical",
                status="blocked",
                reason_code="team_scope_mismatch",
                entity_type="team_scope",
                entity_value=",".join(sorted(scopes - fixture_scopes)),
                evidence=f"dataset team_scope values do not match fixture scopes: fixtures={','.join(sorted(fixture_scopes))}",
                recommendation="Do not join club and national contexts in the same current inference path.",
            ))

    return rows


def _fixture_team_guardrails(sources: dict[str, pd.DataFrame], fixture_teams: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not fixture_teams:
        return rows

    dataset_team_columns = {
        "lineups": ["team"],
        "squads": ["team"],
        "player_events": ["team", "opponent"],
        "actual_results": ["home_team", "away_team"],
        "predictions": ["home_team", "away_team"],
        "dynamic_lines": ["team", "side"],
        "matchday_summary": ["home_team", "away_team"],
    }

    for dataset, columns in dataset_team_columns.items():
        df = sources.get(dataset, pd.DataFrame())
        if df.empty:
            continue

        for column in columns:
            if column not in df.columns:
                continue
            values = _normalized_set(df[column])
            unknown = sorted(values - fixture_teams)
            for team in unknown[:100]:
                rows.append(_guardrail_row(
                    guardrail_scope="entity",
                    dataset=dataset,
                    record_key=f"{column}={team}",
                    severity="warning",
                    status="needs_review",
                    reason_code="team_not_in_fixture",
                    entity_type="team",
                    entity_value=team,
                    evidence=f"{column} appears in {dataset} but not in supplied fixture teams",
                    recommendation="Resolve aliases or keep historical/non-current data outside current fixture coverage checks.",
                ))

    return rows


def _provider_entity_ambiguity_guardrails(sources: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checks = [
        ("provider_team_id", "team", "team", "ambiguous_team_name"),
        ("provider_player_id", "player", "player", "ambiguous_player_name"),
        ("team_id", "team", "team", "ambiguous_team_name"),
        ("player_id", "player", "player", "ambiguous_player_name"),
    ]

    for dataset, df in sources.items():
        if df.empty:
            continue
        for id_col, name_col, entity_type, reason_code in checks:
            if id_col not in df.columns or name_col not in df.columns:
                continue
            working = df[[id_col, name_col]].copy()
            working["_id"] = working[id_col].map(_norm_text)
            working["_name"] = working[name_col].map(_norm_text)
            working = working[working["_id"].ne("") & working["_name"].ne("")]
            if working.empty:
                continue

            ids_per_name = working.groupby("_name")["_id"].nunique()
            ambiguous = ids_per_name[ids_per_name > 1]
            for name, _count in ambiguous.head(100).items():
                ids = sorted(working[working["_name"].eq(name)]["_id"].unique())
                rows.append(_guardrail_row(
                    guardrail_scope="entity",
                    dataset=dataset,
                    record_key=f"{name_col}={name}",
                    severity="warning",
                    status="needs_review",
                    reason_code=reason_code,
                    entity_type=entity_type,
                    entity_value=name,
                    evidence=f"same normalized {name_col} maps to multiple IDs in {dataset}: {','.join(ids[:10])}",
                    recommendation="Resolve aliases or require stable provider/canonical IDs before joining records.",
                ))

    return rows


def _lineup_player_id_guardrails(lineups: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if lineups.empty:
        return rows

    player_id_columns = [column for column in ["player_id", "provider_player_id"] if column in lineups.columns]
    if not player_id_columns:
        rows.append(_guardrail_row(
            guardrail_scope="squad",
            dataset="lineups",
            record_key="player_id",
            severity="warning",
            status="needs_review",
            reason_code="lineup_player_without_player_id",
            entity_type="player",
            entity_value="",
            evidence="lineups were supplied without player_id or provider_player_id",
            recommendation="Use stable player IDs where possible; names alone are unsafe for current player-prop inference.",
        ))
        return rows

    player_col = "player" if "player" in lineups.columns else None
    for idx, row in lineups.iterrows():
        has_any_id = any(_norm_text(row.get(column)) for column in player_id_columns)
        if has_any_id:
            continue
        player = _norm_text(row.get(player_col)) if player_col else ""
        rows.append(_guardrail_row(
            guardrail_scope="squad",
            dataset="lineups",
            record_key=f"row={idx}",
            severity="warning",
            status="needs_review",
            reason_code="lineup_player_without_player_id",
            entity_type="player",
            entity_value=player,
            evidence="lineup row has player name but no stable player ID",
            recommendation="Map the player to provider_player_id or canonical player_id before high-confidence player props.",
        ))

    return rows


def _current_squad_eligibility_guardrails(lineups: pd.DataFrame, squads: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if lineups.empty:
        return rows

    if squads.empty:
        rows.append(_guardrail_row(
            guardrail_scope="squad",
            dataset="squads",
            record_key="dataset",
            severity="critical",
            status="blocked",
            reason_code="missing_current_eligibility",
            entity_type="player",
            entity_value="",
            evidence="lineups were supplied but current squads were not supplied",
            recommendation="Provide current squads before treating player-prop candidates as safe.",
        ))
        return rows

    if not {"team", "player"}.issubset(lineups.columns) or not {"team", "player"}.issubset(squads.columns):
        rows.append(_guardrail_row(
            guardrail_scope="squad",
            dataset="lineups/squads",
            record_key="team_player_columns",
            severity="critical",
            status="blocked",
            reason_code="missing_current_eligibility",
            entity_type="player",
            entity_value="",
            evidence="lineups or squads are missing team/player columns",
            recommendation="Provide team and player columns in both lineups and squads before current player-prop inference.",
        ))
        return rows

    eligibility_columns = [col for col in ["current_squad_flag", "availability_status", "status", "lineup_status"] if col in squads.columns]
    if not eligibility_columns:
        rows.append(_guardrail_row(
            guardrail_scope="squad",
            dataset="squads",
            record_key="current_eligibility_columns",
            severity="warning",
            status="needs_review",
            reason_code="missing_current_eligibility",
            entity_type="player",
            entity_value="",
            evidence="squads lack current_squad_flag, availability_status, status or lineup_status",
            recommendation="Add explicit current eligibility fields; names in a squad file alone are lower confidence.",
        ))

    current_pairs = _eligible_squad_pairs(squads)
    squad_pairs_all = _entity_pairs(squads, "team", "player")
    lineup_pairs = _entity_pairs(lineups, "team", "player")

    for team, player in sorted(lineup_pairs - current_pairs)[:200]:
        if (team, player) in squad_pairs_all:
            reason_code = "missing_current_eligibility"
            evidence = "player appears in squads but lacks current/available eligibility evidence"
        else:
            reason_code = "player_not_in_current_squad"
            evidence = "lineup player is not present in current squad rows"
        rows.append(_guardrail_row(
            guardrail_scope="squad",
            dataset="lineups",
            record_key=f"team={team}; player={player}",
            severity="critical",
            status="blocked",
            reason_code=reason_code,
            entity_type="player",
            entity_value=player,
            evidence=evidence,
            recommendation="Resolve aliases or add a current, available squad row before current player-prop inference.",
        ))

    return rows


def _historical_only_player_guardrails(player_events: pd.DataFrame, squads: pd.DataFrame, fixture_teams: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if player_events.empty or squads.empty or not {"team", "player"}.issubset(player_events.columns) or not {"team", "player"}.issubset(squads.columns):
        return rows

    current_pairs = _eligible_squad_pairs(squads)
    event_pairs = _entity_pairs(player_events, "team", "player")

    if fixture_teams:
        event_pairs = {pair for pair in event_pairs if pair[0] in fixture_teams}

    historical_only = sorted(event_pairs - current_pairs)
    for team, player in historical_only[:200]:
        rows.append(_guardrail_row(
            guardrail_scope="squad",
            dataset="player_events",
            record_key=f"team={team}; player={player}",
            severity="warning",
            status="unsafe_for_current_player_props",
            reason_code="historical_only_player_for_current_inference",
            entity_type="player",
            entity_value=player,
            evidence="player appears in historical events for a current fixture team but lacks current squad eligibility",
            recommendation="Use this row for historical modelling only; do not surface as a current player-prop candidate without current evidence.",
        ))

    return rows


def _player_name_ambiguity_guardrails(sources: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for dataset in ["lineups", "squads", "player_events"]:
        df = sources.get(dataset, pd.DataFrame())
        if df.empty or "player" not in df.columns or "team" not in df.columns:
            continue

        working = df[["team", "player"]].copy()
        working["_team"] = working["team"].map(_norm_text)
        working["_player"] = working["player"].map(_norm_text)
        working = working[working["_team"].ne("") & working["_player"].ne("")]
        if working.empty:
            continue

        teams_per_player = working.groupby("_player")["_team"].nunique()
        ambiguous = teams_per_player[teams_per_player > 1]
        for player, _count in ambiguous.head(100).items():
            teams = sorted(working[working["_player"].eq(player)]["_team"].unique())
            rows.append(_guardrail_row(
                guardrail_scope="squad",
                dataset=dataset,
                record_key=f"player={player}",
                severity="warning",
                status="needs_review",
                reason_code="ambiguous_player_name",
                entity_type="player",
                entity_value=player,
                evidence=f"same normalized player name appears for multiple teams in {dataset}: {','.join(teams[:10])}",
                recommendation="Use stable player_id/provider_player_id to distinguish same-name or transferred players.",
            ))

    return rows


def _eligible_squad_pairs(squads: pd.DataFrame) -> set[tuple[str, str]]:
    if squads.empty or not {"team", "player"}.issubset(squads.columns):
        return set()

    working = squads.copy()
    eligibility_columns = [col for col in ["current_squad_flag", "availability_status", "status", "lineup_status"] if col in working.columns]

    if "current_squad_flag" in working.columns:
        working = working[working["current_squad_flag"].map(_truthy_current_flag)]

    if "status" in working.columns:
        statuses = working["status"].map(_norm_text)
        working = working[~statuses.isin({"inactive", "retired", "former", "old", "not current", "out"})]

    if "availability_status" in working.columns:
        availability = working["availability_status"].map(_norm_text)
        working = working[~availability.isin({"unavailable", "injured", "suspended", "out", "doubtful"})]

    if "lineup_status" in working.columns:
        lineup_status = working["lineup_status"].map(_norm_text)
        working = working[~lineup_status.isin({"out", "inactive", "unavailable"})]

    if not eligibility_columns and not working.empty:
        # Fallback for legacy v0.49.0 sample files: if no explicit eligibility
        # field exists, squad membership is treated as lower-confidence evidence.
        return _entity_pairs(working, "team", "player")

    return _entity_pairs(working, "team", "player")


def _build_guardrail_summary(entity_guardrails: pd.DataFrame, squad_guardrails: pd.DataFrame) -> dict[str, Any]:
    combined = pd.concat([entity_guardrails, squad_guardrails], ignore_index=True)
    issue_rows = combined[~combined["status"].isin(["ok"])].copy()
    critical_count = int(issue_rows["severity"].eq("critical").sum()) if not issue_rows.empty else 0
    warning_count = int(issue_rows["severity"].eq("warning").sum()) if not issue_rows.empty else 0

    if critical_count:
        status = "blocked"
    elif warning_count:
        status = "warning"
    else:
        status = "ok"

    reason_codes = sorted(set(issue_rows["reason_code"].astype(str).tolist())) if not issue_rows.empty else []

    return {
        "version": ENTITY_GUARDRAILS_VERSION,
        "status": status,
        "entity_guardrail_rows": int(len(entity_guardrails)),
        "squad_guardrail_rows": int(len(squad_guardrails)),
        "critical_count": critical_count,
        "warning_count": warning_count,
        "reason_codes": reason_codes,
        "unsafe_for_player_props": bool(
            combined["status"].astype(str).isin(["blocked", "unsafe_for_current_player_props"]).any()
            and not issue_rows.empty
        ),
        "principles": {
            "offline_only": True,
            "model_logic_changed": False,
            "external_api_calls": False,
            "missing_data_policy": "not_available",
            "current_player_props_require_current_eligibility": True,
        },
    }


def _guardrail_row(
    *,
    guardrail_scope: str,
    dataset: str,
    record_key: str,
    severity: str,
    status: str,
    reason_code: str,
    entity_type: str,
    entity_value: str,
    evidence: str,
    recommendation: str,
) -> dict[str, Any]:
    return {
        "guardrail_scope": guardrail_scope,
        "dataset": dataset,
        "record_key": record_key,
        "severity": severity,
        "status": status,
        "reason_code": reason_code,
        "entity_type": entity_type,
        "entity_value": entity_value,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _ensure_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    return pd.DataFrame(frame)


def _fixture_team_set(fixtures: pd.DataFrame) -> set[str]:
    if fixtures.empty:
        return set()
    values: set[str] = set()
    for col in ["home_team", "away_team", "team"]:
        if col in fixtures.columns:
            values |= _normalized_set(fixtures[col])
    return values


def _scope_values(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    values: set[str] = set()
    for col in ["team_scope", "team_type", "competition_context"]:
        if col in df.columns:
            # For team_type/competition_context only keep explicit club/national
            col_values = _normalized_set(df[col])
            values |= {value for value in col_values if value in {"club", "national"}}
    return values


def _entity_pairs(df: pd.DataFrame, first: str, second: str) -> set[tuple[str, str]]:
    if first not in df.columns or second not in df.columns:
        return set()
    pairs: set[tuple[str, str]] = set()
    for _, row in df[[first, second]].dropna().iterrows():
        first_value = _norm_text(row[first])
        second_value = _norm_text(row[second])
        if first_value and second_value:
            pairs.add((first_value, second_value))
    return pairs


def _normalized_set(series: pd.Series | None) -> set[str]:
    if series is None:
        return set()
    return {_norm_text(value) for value in series.dropna().tolist() if _norm_text(value)}


def _truthy_current_flag(value: Any) -> bool:
    text = _norm_text(value)
    if text in {"1", "true", "yes", "y", "current", "active", "available", "starter", "bench", "squad"}:
        return True
    if text in {"0", "false", "no", "n", "inactive", "retired", "former", "out", "unavailable", "nan", "none", ""}:
        return False
    if isinstance(value, bool):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(numeric)


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "<na>"}:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.replace("_", " ").replace("-", " ").split())
