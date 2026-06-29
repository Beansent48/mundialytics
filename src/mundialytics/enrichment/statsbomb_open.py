from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

CANONICAL_XG_COLUMNS = [
    "provider",
    "provider_match_id",
    "date",
    "competition",
    "season",
    "home_team",
    "away_team",
    "home_xg",
    "away_xg",
    "home_npxg",
    "away_npxg",
    "xg_match_confidence",
]


STATSBOMB_OPEN_XG_VERSION = "v0.49.9_statsbomb_open_xg_import"
STATSBOMB_PROVIDER = "statsbomb_open_data"


@dataclass(frozen=True)
class StatsBombOpenXGOutputs:
    xg_matches: pd.DataFrame
    xg_shots: pd.DataFrame
    report: dict[str, Any]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "")
    return str(value or "")


def _normalize_filter_values(values: Iterable[str] | None) -> set[str]:
    return {str(v) for v in values or [] if str(v).strip()}


def _load_competitions(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "competitions.json"
    if not path.exists():
        raise FileNotFoundError(f"StatsBomb competitions.json not found: {path}")
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected competitions.json to contain a list: {path}")
    return pd.DataFrame(rows)


def _iter_match_files(
    data_dir: Path,
    *,
    competition_ids: set[str] | None = None,
    season_ids: set[str] | None = None,
) -> list[Path]:
    matches_dir = data_dir / "matches"
    if not matches_dir.exists():
        raise FileNotFoundError(f"StatsBomb matches directory not found: {matches_dir}")

    files: list[Path] = []
    for competition_dir in sorted(matches_dir.iterdir()):
        if not competition_dir.is_dir():
            continue
        if competition_ids and competition_dir.name not in competition_ids:
            continue
        for season_file in sorted(competition_dir.glob("*.json")):
            season_id = season_file.stem
            if season_ids and season_id not in season_ids:
                continue
            files.append(season_file)
    return files


def _load_match_rows(data_dir: Path, *, competition_ids: set[str] | None = None, season_ids: set[str] | None = None) -> pd.DataFrame:
    competitions = _load_competitions(data_dir)
    comp_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in competitions.iterrows():
        comp_lookup[(str(row.get("competition_id")), str(row.get("season_id")))] = row.to_dict()

    rows: list[dict[str, Any]] = []
    for path in _iter_match_files(data_dir, competition_ids=competition_ids, season_ids=season_ids):
        competition_id = path.parent.name
        season_id = path.stem
        meta = comp_lookup.get((competition_id, season_id), {})
        matches = _read_json(path)
        if not isinstance(matches, list):
            continue
        for item in matches:
            match_id = item.get("match_id")
            if match_id is None:
                continue
            rows.append(
                {
                    "provider_match_id": str(match_id),
                    "match_id": str(match_id),
                    "date": item.get("match_date"),
                    "competition": meta.get("competition_name") or _as_name(item.get("competition")) or competition_id,
                    "season": meta.get("season_name") or _as_name(item.get("season")) or season_id,
                    "competition_id": str(competition_id),
                    "season_id": str(season_id),
                    "home_team": _as_name(item.get("home_team")),
                    "away_team": _as_name(item.get("away_team")),
                    "home_score": item.get("home_score"),
                    "away_score": item.get("away_score"),
                    "match_file": str(path),
                }
            )
    return pd.DataFrame(rows)


def _shot_row(event: dict[str, Any], match: dict[str, Any]) -> dict[str, Any] | None:
    shot = event.get("shot")
    if not isinstance(shot, dict):
        return None

    team = _as_name(event.get("team"))
    xg = shot.get("statsbomb_xg")
    try:
        xg_value = float(xg) if xg is not None else 0.0
    except (TypeError, ValueError):
        xg_value = 0.0

    shot_type = _as_name(shot.get("type"))
    is_penalty = shot_type.lower() == "penalty"

    location = event.get("location") or []
    x = location[0] if isinstance(location, list) and len(location) > 0 else None
    y = location[1] if isinstance(location, list) and len(location) > 1 else None

    return {
        "provider": STATSBOMB_PROVIDER,
        "provider_match_id": match["provider_match_id"],
        "date": match["date"],
        "competition": match["competition"],
        "season": match["season"],
        "team": team,
        "opponent": match["away_team"] if team == match["home_team"] else match["home_team"],
        "is_home_team": bool(team == match["home_team"]),
        "player": _as_name(event.get("player")),
        "minute": event.get("minute"),
        "second": event.get("second"),
        "xg": xg_value,
        "npxg": 0.0 if is_penalty else xg_value,
        "shot_type": shot_type,
        "body_part": _as_name(shot.get("body_part")),
        "outcome": _as_name(shot.get("outcome")),
        "technique": _as_name(shot.get("technique")),
        "under_pressure": bool(event.get("under_pressure", False)),
        "x": x,
        "y": y,
    }


def _load_event_shots(data_dir: Path, matches: pd.DataFrame, *, max_matches: int | None = None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    events_dir = data_dir / "events"
    if not events_dir.exists():
        raise FileNotFoundError(f"StatsBomb events directory not found: {events_dir}")

    shots: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    iterable = matches.to_dict("records")
    if max_matches is not None:
        iterable = iterable[: max(0, int(max_matches))]

    for match in iterable:
        match_id = str(match["provider_match_id"])
        path = events_dir / f"{match_id}.json"
        if not path.exists():
            failures.append({"provider_match_id": match_id, "error": "events_file_missing", "path": str(path)})
            continue
        try:
            events = _read_json(path)
            if not isinstance(events, list):
                failures.append({"provider_match_id": match_id, "error": "events_file_not_list", "path": str(path)})
                continue
            for event in events:
                if _as_name(event.get("type")) != "Shot":
                    continue
                row = _shot_row(event, match)
                if row is not None:
                    shots.append(row)
        except Exception as exc:  # pragma: no cover - defensive report path
            failures.append({"provider_match_id": match_id, "error": str(exc), "path": str(path)})
    return pd.DataFrame(shots), failures


def _aggregate_match_xg(matches: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = shots.groupby(["provider_match_id", "team"], dropna=False).agg(xg=("xg", "sum"), npxg=("npxg", "sum")).reset_index() if not shots.empty else pd.DataFrame()
    for match in matches.to_dict("records"):
        match_id = str(match["provider_match_id"])
        home = str(match["home_team"])
        away = str(match["away_team"])

        home_xg = 0.0
        away_xg = 0.0
        home_npxg = 0.0
        away_npxg = 0.0
        if not grouped.empty:
            subset = grouped[grouped["provider_match_id"].astype(str).eq(match_id)]
            for _, row in subset.iterrows():
                team = str(row["team"])
                if team == home:
                    home_xg = float(row["xg"])
                    home_npxg = float(row["npxg"])
                elif team == away:
                    away_xg = float(row["xg"])
                    away_npxg = float(row["npxg"])

        rows.append(
            {
                "provider": STATSBOMB_PROVIDER,
                "provider_match_id": match_id,
                "date": match["date"],
                "competition": match["competition"],
                "season": match["season"],
                "home_team": home,
                "away_team": away,
                "home_xg": home_xg,
                "away_xg": away_xg,
                "home_npxg": home_npxg,
                "away_npxg": away_npxg,
                "xg_match_confidence": "statsbomb_event_aggregate",
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_XG_COLUMNS)


def import_statsbomb_open_xg(
    data_dir: str | Path,
    *,
    competition_ids: Iterable[str] | None = None,
    season_ids: Iterable[str] | None = None,
    max_matches: int | None = None,
) -> StatsBombOpenXGOutputs:
    """Import free StatsBomb Open Data shot xG from a local open-data checkout.

    The importer is intentionally offline-first: it reads the user's local
    `statsbomb/open-data/data` directory and never calls the StatsBomb API.
    Coverage depends entirely on the competitions and matches present in that
    local checkout.
    """

    data_path = Path(data_dir)
    comp_filter = _normalize_filter_values(competition_ids) or None
    season_filter = _normalize_filter_values(season_ids) or None

    matches = _load_match_rows(data_path, competition_ids=comp_filter, season_ids=season_filter)
    if matches.empty:
        shots = pd.DataFrame()
        xg_matches = pd.DataFrame(columns=CANONICAL_XG_COLUMNS)
        failures: list[dict[str, Any]] = []
    else:
        matches = matches.sort_values(["date", "provider_match_id"]).reset_index(drop=True)
        if max_matches is not None:
            matches = matches.head(max(0, int(max_matches))).copy()
        shots, failures = _load_event_shots(data_path, matches, max_matches=None)
        xg_matches = _aggregate_match_xg(matches, shots)

    report = {
        "version": STATSBOMB_OPEN_XG_VERSION,
        "status": "ok" if not xg_matches.empty else "warning",
        "provider": STATSBOMB_PROVIDER,
        "mode": "free_official_open_data_local_import",
        "data_dir": str(data_path),
        "competition_ids": sorted(comp_filter) if comp_filter else [],
        "season_ids": sorted(season_filter) if season_filter else [],
        "matches_discovered": int(len(matches)),
        "xg_match_rows": int(len(xg_matches)),
        "shot_rows": int(len(shots)),
        "failures": failures[:50],
        "failure_count": int(len(failures)),
        "competitions": sorted(xg_matches["competition"].dropna().astype(str).unique().tolist()) if not xg_matches.empty else [],
        "seasons": sorted(xg_matches["season"].dropna().astype(str).unique().tolist()) if not xg_matches.empty else [],
        "terms_note": "StatsBomb Open Data is official free open data, but coverage is partial and usage should respect the repository license/terms.",
        "raw_data_changed": False,
    }

    return StatsBombOpenXGOutputs(
        xg_matches=xg_matches,
        xg_shots=shots,
        report=report,
    )
