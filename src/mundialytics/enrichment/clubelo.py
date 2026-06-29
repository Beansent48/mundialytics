from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote
import time

import pandas as pd
import requests

from mundialytics.data_quality.team_registry import normalize_provider_name, provider_alias_map, team_id_from_name


CLUBELO_ENRICHMENT_VERSION = "v0.49.7_clubelo_enrichment"
CLUBELO_API_BASE = "https://api.clubelo.com"


@dataclass(frozen=True)
class ClubEloDownloadOutputs:
    downloaded_files: list[Path]
    report: dict[str, Any]


@dataclass(frozen=True)
class ClubEloEnrichmentOutputs:
    enriched_matches: pd.DataFrame
    match_features: pd.DataFrame
    join_report: pd.DataFrame
    summary: dict[str, Any]


def clubelo_daily_path(out_dir: str | Path, date: str | pd.Timestamp) -> Path:
    d = pd.to_datetime(date, errors="raise").date().isoformat()
    return Path(out_dir) / "daily" / f"clubelo_{d}.csv"


def clubelo_team_history_path(out_dir: str | Path, alias: str) -> Path:
    """Stable cache path for one ClubElo team-history API response."""
    alias_id = team_id_from_name(alias)
    return Path(out_dir) / "teams" / f"clubelo_team_{alias_id}.csv"


def _validate_clubelo_csv_text(text: str, *, required: set[str]) -> pd.DataFrame:
    from io import StringIO

    if not text or not text.strip():
        raise RuntimeError("empty ClubElo response")
    df = pd.read_csv(StringIO(text))
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"response did not look like expected ClubElo CSV; missing={missing}")
    return df


def _collect_requested_teams(matches: pd.DataFrame, registry: pd.DataFrame | None = None) -> list[dict[str, str]]:
    alias_map = provider_alias_map(registry, "clubelo_name") if registry is not None else {}
    teams = sorted(set(matches.get("home_team", pd.Series(dtype=str)).dropna().astype(str)) | set(matches.get("away_team", pd.Series(dtype=str)).dropna().astype(str)))
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for team in teams:
        alias = alias_map.get(normalize_provider_name(team), team)
        alias_norm = normalize_provider_name(alias)
        if not alias_norm or alias_norm in seen:
            continue
        seen.add(alias_norm)
        rows.append({
            "football_data_name": team,
            "canonical_team_id": team_id_from_name(team),
            "clubelo_alias": alias,
            "clubelo_alias_norm": alias_norm,
        })
    return rows


def download_clubelo_team_histories(
    matches: pd.DataFrame,
    out_dir: str | Path,
    *,
    registry: pd.DataFrame | None = None,
    force: bool = False,
    timeout: int = 30,
    sleep_seconds: float = 0.05,
) -> ClubEloDownloadOutputs:
    """Download one full ClubElo history per team alias.

    This is the practical default. ClubElo supports full team-history API calls
    such as ``api.clubelo.com/Arsenal``. Downloading one file per team is much
    faster than one full ranking snapshot per match date and is enough to
    recover the pre-match Elo via the history ``From``/``To`` intervals.
    """
    out_dir = Path(out_dir)
    team_dir = out_dir / "teams"
    team_dir.mkdir(parents=True, exist_ok=True)

    requested = _collect_requested_teams(matches, registry)
    downloaded: list[Path] = []
    failures: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    for item in requested:
        alias = item["clubelo_alias"]
        path = clubelo_team_history_path(out_dir, alias)
        if path.exists() and not force:
            skipped.append({"team": item["football_data_name"], "clubelo_alias": alias, "path": str(path)})
            downloaded.append(path)
            continue

        url = f"{CLUBELO_API_BASE}/{quote(alias, safe='')}"
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": "mundialytics-data-enrichment/0.49.7"})
            response.raise_for_status()
            df = _validate_clubelo_csv_text(response.text, required={"Club", "Elo", "From", "To"})
            if df.empty:
                raise RuntimeError("empty ClubElo team history")
            path.write_text(response.text.strip() + "\n", encoding="utf-8")
            downloaded.append(path)
        except Exception as exc:  # pragma: no cover - depends on internet/API
            failures.append({
                "team": item["football_data_name"],
                "clubelo_alias": alias,
                "url": url,
                "error": str(exc),
                "recommendation": "Edit data/processed/entities/team_registry.csv clubelo_name for this team, then re-run.",
            })

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    report = {
        "version": CLUBELO_ENRICHMENT_VERSION,
        "status": "ok" if not failures else ("warning" if downloaded else "blocked"),
        "provider": "clubelo",
        "mode": "team_histories",
        "requested_team_aliases": len(requested),
        "downloaded_or_cached_files": len(downloaded),
        "skipped_existing": len(skipped),
        "failures": failures,
        "output_dir": str(out_dir),
        "cache_dir": str(team_dir),
        "why_this_mode": "one API request per team instead of one full snapshot per match date",
        "principle": "external_provider_data_is_cached_and_never_written_to_data_raw",
    }
    return ClubEloDownloadOutputs(downloaded, report)


def download_clubelo_daily_ratings(
    dates: list[str] | pd.Series,
    out_dir: str | Path,
    *,
    force: bool = False,
    timeout: int = 30,
    sleep_seconds: float = 0.25,
) -> ClubEloDownloadOutputs:
    """Download ClubElo full daily rating snapshots for the provided dates.

    This legacy mode is accurate but can be slow for multi-season datasets
    because it needs one full snapshot per unique match date. Prefer
    ``download_clubelo_team_histories`` unless you specifically need daily
    full-table snapshots.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = out_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    unique_dates = sorted({
        pd.to_datetime(d, errors="coerce").date().isoformat()
        for d in dates
        if pd.notna(pd.to_datetime(d, errors="coerce"))
    })

    downloaded: list[Path] = []
    failures: list[dict[str, str]] = []
    skipped: list[str] = []

    for d in unique_dates:
        path = clubelo_daily_path(out_dir, d)
        if path.exists() and not force:
            skipped.append(d)
            downloaded.append(path)
            continue

        url = f"{CLUBELO_API_BASE}/{d}"
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": "mundialytics-data-enrichment/0.49.7"})
            response.raise_for_status()
            _validate_clubelo_csv_text(response.text, required={"Club", "Elo"})
            path.write_text(response.text.strip() + "\n", encoding="utf-8")
            downloaded.append(path)
        except Exception as exc:  # pragma: no cover - depends on internet/API
            failures.append({"date": d, "url": url, "error": str(exc)})
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    report = {
        "version": CLUBELO_ENRICHMENT_VERSION,
        "status": "ok" if not failures else "warning",
        "provider": "clubelo",
        "mode": "daily_rating_snapshots_legacy",
        "requested_dates": len(unique_dates),
        "downloaded_or_cached_files": len(downloaded),
        "skipped_existing": len(skipped),
        "failures": failures,
        "output_dir": str(out_dir),
        "warning": "This mode can be slow for multi-season datasets; prefer --mode team-history.",
        "principle": "external_provider_data_is_cached_and_never_written_to_data_raw",
    }
    return ClubEloDownloadOutputs(downloaded, report)


def _read_daily_snapshot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Club" not in df.columns or "Elo" not in df.columns:
        raise ValueError(f"ClubElo snapshot missing Club/Elo columns: {path}")
    out = df.copy()
    out["clubelo_norm"] = out["Club"].map(normalize_provider_name)
    out["Elo"] = pd.to_numeric(out["Elo"], errors="coerce")
    return out


def _read_team_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted({"Club", "Elo", "From", "To"} - set(df.columns))
    if missing:
        raise ValueError(f"ClubElo team history missing columns {missing}: {path}")
    out = df.copy()
    out["clubelo_norm"] = out["Club"].map(normalize_provider_name)
    out["Elo"] = pd.to_numeric(out["Elo"], errors="coerce")
    out["from_date"] = pd.to_datetime(out["From"], errors="coerce")
    out["to_date"] = pd.to_datetime(out["To"], errors="coerce")
    return out.sort_values(["from_date", "to_date"], na_position="last").reset_index(drop=True)


def _best_clubelo_match(alias: str, snapshot: pd.DataFrame, *, fuzzy_threshold: float = 0.94) -> tuple[float | None, str | None, str]:
    alias_norm = normalize_provider_name(alias)
    if not alias_norm:
        return None, None, "missing_alias"

    exact = snapshot.loc[snapshot["clubelo_norm"] == alias_norm]
    if not exact.empty:
        row = exact.iloc[0]
        return float(row["Elo"]) if pd.notna(row["Elo"]) else None, str(row["Club"]), "exact"

    best_name = None
    best_score = 0.0
    for club_name, norm in zip(snapshot["Club"].astype(str), snapshot["clubelo_norm"].astype(str), strict=False):
        score = SequenceMatcher(None, alias_norm, norm).ratio()
        if score > best_score:
            best_score = score
            best_name = club_name
    if best_name is not None and best_score >= fuzzy_threshold:
        row = snapshot.loc[snapshot["Club"].astype(str) == best_name].iloc[0]
        return float(row["Elo"]) if pd.notna(row["Elo"]) else None, best_name, f"fuzzy_{best_score:.3f}"

    return None, best_name, f"unmatched_best_{best_score:.3f}" if best_name else "unmatched"


def _lookup_history_elo(alias: str, date_value: Any, clubelo_dir: str | Path) -> tuple[float | None, str | None, str]:
    alias_norm = normalize_provider_name(alias)
    if not alias_norm:
        return None, None, "missing_alias"
    d = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(d):
        return None, None, "invalid_match_date"

    path = clubelo_team_history_path(clubelo_dir, alias)
    if not path.exists():
        return None, None, "missing_team_history"

    history = _read_team_history(path)
    if history.empty:
        return None, None, "empty_team_history"

    valid = history.loc[history["Elo"].notna()].copy()
    if valid.empty:
        return None, None, "history_without_elo"

    # ClubElo histories expose rating intervals through From/To. Use the row
    # whose interval contains the match date. Fallback to the latest known prior
    # rating if the interval is not available for that exact day.
    in_interval = valid.loc[(valid["from_date"].notna()) & (valid["to_date"].notna()) & (valid["from_date"] <= d) & (valid["to_date"] >= d)]
    if not in_interval.empty:
        row = in_interval.sort_values("from_date").iloc[-1]
        return float(row["Elo"]), str(row["Club"]), "history_interval"

    prior = valid.loc[(valid["from_date"].notna()) & (valid["from_date"] <= d)]
    if not prior.empty:
        row = prior.sort_values("from_date").iloc[-1]
        return float(row["Elo"]), str(row["Club"]), "history_latest_prior"

    return None, str(valid.iloc[0].get("Club", "")), "no_rating_before_match_date"


def _has_team_histories(clubelo_dir: str | Path) -> bool:
    team_dir = Path(clubelo_dir) / "teams"
    return team_dir.exists() and any(team_dir.glob("clubelo_team_*.csv"))


def enrich_matches_with_clubelo(
    matches: pd.DataFrame,
    registry: pd.DataFrame,
    clubelo_dir: str | Path,
    *,
    dataset_name: str = "clubelo_enriched_matches",
    fuzzy_threshold: float = 0.94,
    source_mode: str = "auto",
) -> ClubEloEnrichmentOutputs:
    """Attach pre-match ClubElo values from cached histories or daily snapshots."""
    if matches.empty:
        summary = {
            "version": CLUBELO_ENRICHMENT_VERSION,
            "dataset_name": dataset_name,
            "status": "blocked",
            "reason": "empty_matches",
        }
        return ClubEloEnrichmentOutputs(matches.copy(), pd.DataFrame(), pd.DataFrame(), summary)

    required = {"match_id", "date", "home_team", "away_team"}
    missing = sorted(required - set(matches.columns))
    if missing:
        summary = {
            "version": CLUBELO_ENRICHMENT_VERSION,
            "dataset_name": dataset_name,
            "status": "blocked",
            "missing_required_columns": missing,
        }
        return ClubEloEnrichmentOutputs(matches.copy(), pd.DataFrame(), pd.DataFrame(), summary)

    if source_mode not in {"auto", "team-history", "daily-snapshot"}:
        raise ValueError("source_mode must be one of: auto, team-history, daily-snapshot")
    resolved_mode = "team-history" if (source_mode == "team-history" or (source_mode == "auto" and _has_team_histories(clubelo_dir))) else "daily-snapshot"

    alias_map = provider_alias_map(registry, "clubelo_name")
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    features: list[dict[str, Any]] = []
    join_rows: list[dict[str, Any]] = []
    snapshot_cache: dict[str, pd.DataFrame] = {}

    for _, row in df.sort_values(["date", "match_id"]).iterrows():
        d = row["date"]
        date_key = d.date().isoformat() if pd.notna(d) else ""

        feature_row: dict[str, Any] = {
            "match_id": row["match_id"],
            "clubelo_snapshot_date": date_key or pd.NA,
            "clubelo_provider": "clubelo",
            "clubelo_source_mode": resolved_mode,
            "clubelo_snapshot_status": "available" if resolved_mode == "team-history" else "unknown",
        }

        snapshot = None
        snapshot_status = "available"
        if resolved_mode == "daily-snapshot":
            path = clubelo_daily_path(clubelo_dir, date_key) if date_key else None
            if not date_key or path is None or not path.exists():
                snapshot_status = "missing_snapshot"
            else:
                if date_key not in snapshot_cache:
                    snapshot_cache[date_key] = _read_daily_snapshot(path)
                snapshot = snapshot_cache[date_key]
            feature_row["clubelo_snapshot_status"] = snapshot_status

        for side in ["home", "away"]:
            team = str(row[f"{side}_team"])
            alias = alias_map.get(normalize_provider_name(team), team)
            elo: float | None = None
            matched_name: str | None = None
            join_status = "unknown"

            if resolved_mode == "team-history":
                elo, matched_name, join_status = _lookup_history_elo(alias, d, clubelo_dir)
            else:
                join_status = snapshot_status
                if snapshot is not None:
                    elo, matched_name, join_status = _best_clubelo_match(alias, snapshot, fuzzy_threshold=fuzzy_threshold)

            feature_row[f"{side}_clubelo"] = elo
            feature_row[f"{side}_external_elo"] = elo
            feature_row[f"{side}_clubelo_name"] = matched_name
            feature_row[f"{side}_clubelo_join_status"] = join_status
            join_rows.append({
                "match_id": row["match_id"],
                "date": date_key,
                "side": side,
                "team": team,
                "clubelo_alias": alias,
                "clubelo_matched_name": matched_name,
                "join_status": join_status,
                "elo": elo,
                "source_mode": resolved_mode,
            })

        if feature_row.get("home_clubelo") is not None and feature_row.get("away_clubelo") is not None:
            feature_row["clubelo_diff"] = float(feature_row["home_clubelo"]) - float(feature_row["away_clubelo"])
            feature_row["clubelo_available"] = True
        else:
            feature_row["clubelo_diff"] = pd.NA
            feature_row["clubelo_available"] = False
        features.append(feature_row)

    match_features = pd.DataFrame(features)
    enriched = df.merge(match_features, on="match_id", how="left", validate="one_to_one")
    # Keep compatibility with existing loaders/model features.
    if "home_external_elo" not in enriched.columns:
        enriched["home_external_elo"] = enriched["home_clubelo"]
    if "away_external_elo" not in enriched.columns:
        enriched["away_external_elo"] = enriched["away_clubelo"]

    join_report = pd.DataFrame(join_rows)
    total_sides = len(join_report)
    matched_sides = int(join_report["elo"].notna().sum()) if total_sides else 0
    matched_matches = int(match_features["clubelo_available"].sum()) if not match_features.empty else 0
    status = "ok" if matched_matches == len(match_features) else ("warning" if matched_matches > 0 else "blocked")
    summary = {
        "version": CLUBELO_ENRICHMENT_VERSION,
        "dataset_name": dataset_name,
        "status": status,
        "provider": "clubelo",
        "source_mode": resolved_mode,
        "input_rows": int(len(matches)),
        "output_rows": int(len(enriched)),
        "matches_with_full_clubelo": matched_matches,
        "coverage_rate": float(matched_matches / len(matches)) if len(matches) else 0.0,
        "matched_team_sides": matched_sides,
        "team_side_coverage_rate": float(matched_sides / total_sides) if total_sides else 0.0,
        "unmatched_rows": int(total_sides - matched_sides),
        "unmatched_examples": join_report.loc[join_report["elo"].isna(), ["team", "clubelo_alias", "join_status"]].drop_duplicates().head(20).to_dict("records")
        if total_sides else [],
        "raw_data_changed": False,
        "model_logic_changed": False,
        "leakage_policy": "clubelo_team_histories_are_resolved_as_of_match_date_using_provider_From_To_intervals",
    }
    return ClubEloEnrichmentOutputs(enriched, match_features, join_report, summary)
