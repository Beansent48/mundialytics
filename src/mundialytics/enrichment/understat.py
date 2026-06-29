from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pandas as pd
import requests

from mundialytics.enrichment.xg import CANONICAL_XG_COLUMNS, canonicalize_xg_matches


UNDERSTAT_DOWNLOAD_VERSION = "v0.49.8_xg_provider_fallbacks"
UNDERSTAT_BASE_URL = "https://understat.com/league"

UNDERSTAT_LEAGUES = {
    "EPL": "EPL",
    "Premier League": "EPL",
    "LaLiga": "La_liga",
    "La Liga": "La_liga",
    "Serie A": "Serie_A",
    "Serie_A": "Serie_A",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue_1",
    "Ligue_1": "Ligue_1",
    "SP1": "La_liga",
    "E0": "EPL",
    "I1": "Serie_A",
    "D1": "Bundesliga",
    "F1": "Ligue_1",
}


@dataclass(frozen=True)
class UnderstatDownloadOutputs:
    xg_matches: pd.DataFrame
    report: dict[str, Any]


def _empty_xg_matches() -> pd.DataFrame:
    """Return an empty canonical xG dataframe.

    Downstream enrichment scripts should not fail with FileNotFoundError when a
    research scraper is blocked by provider markup/rate limits. Empty canonical
    output lets the pipeline continue while reporting zero xG coverage.
    """
    return pd.DataFrame(columns=CANONICAL_XG_COLUMNS)


def _decode_understat_json(raw: str) -> Any:
    """Decode the JavaScript-escaped JSON used by historic Understat pages."""
    # Historic pages used JSON.parse('...') where the payload is JS escaped.
    # unquote handles occasional URL-escaped exports; unicode_escape handles
    # \xNN and escaped quotes from the inline script representation.
    raw = unquote(raw)
    try:
        decoded = raw.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        decoded = raw
    return json.loads(decoded)


def _extract_understat_var(html: str, variable: str) -> Any:
    """Extract an Understat JS variable from several known markup variants."""
    patterns = [
        # Current format: JSON.parse('...') with single quotes
        rf"(?:var\s+)?{re.escape(variable)}\s*=\s*JSON\.parse\('(?P<data>.*?)'\)",
        # Double-quote variant
        rf"(?:var\s+)?{re.escape(variable)}\s*=\s*JSON\.parse\(\"(?P<data>.*?)\"\)",
        # Direct array assignment (no JSON.parse)
        rf"(?:var\s+)?{re.escape(variable)}\s*=\s*(?P<data>\[.*?\])\s*;",
        # Inline object assignment
        rf"(?:var\s+)?{re.escape(variable)}\s*=\s*(?P<data>\{{.*?\}})\s*;",
        # Encoded with decodeURIComponent
        rf"decodeURIComponent\('(?P<data>%5B.*?)'\)",
        # JSON inside script tag with variable on its own line
        rf"{re.escape(variable)}\s*=\s*JSON\.parse\(decodeURIComponent\('(?P<data>.*?)'\)\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.DOTALL)
        if not match:
            continue
        raw = match.group("data")
        raw_stripped = raw.lstrip()
        if raw_stripped.startswith("[") or raw_stripped.startswith("{"):
            try:
                return json.loads(raw_stripped)
            except json.JSONDecodeError:
                pass
        try:
            return _decode_understat_json(raw)
        except Exception:
            continue
    raise ValueError(
        f"Could not find Understat {variable} JSON in page. "
        "Understat may have changed/removed inline JSON; use a provider/manual xG CSV import."
    )


def _extract_dates_data(html: str) -> list[dict[str, Any]]:
    return _extract_understat_var(html, "datesData")


def fetch_understat_league_season(league: str, season_start_year: int, *, timeout: int = 30) -> pd.DataFrame:
    league_slug = UNDERSTAT_LEAGUES.get(league, league)
    url = f"{UNDERSTAT_BASE_URL}/{league_slug}/{int(season_start_year)}"
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
    )
    response.raise_for_status()
    data = _extract_dates_data(response.text)

    rows: list[dict[str, Any]] = []
    season = f"{season_start_year}-{season_start_year + 1}"
    for item in data:
        if not item.get("isResult", False):
            continue
        xg = item.get("xG", {}) or {}
        home = item.get("h", {}) or {}
        away = item.get("a", {}) or {}
        rows.append({
            "provider": "understat",
            "provider_match_id": item.get("id"),
            "date": str(item.get("datetime", ""))[:10],
            "competition": league_slug,
            "season": season,
            "home_team": home.get("title"),
            "away_team": away.get("title"),
            "home_xg": xg.get("h"),
            "away_xg": xg.get("a"),
            "home_npxg": pd.NA,
            "away_npxg": pd.NA,
            "xg_match_confidence": "understat_league_page_match_xg",
        })
    return pd.DataFrame(rows)


def normalize_provider_xg_csv(
    input_csv: str | Path,
    out_dir: str | Path,
    *,
    provider: str = "provider_csv",
    output_filename: str = "understat_xg_matches.csv",
) -> UnderstatDownloadOutputs:
    """Normalize an already-downloaded provider/manual xG CSV.

    This is the supported fallback when Understat direct scraping is blocked.
    It accepts common column aliases through canonicalize_xg_matches and writes
    the same canonical file expected by enrich_matches_with_xg.py.
    """
    input_csv = Path(input_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_csv)
    xg_matches = canonicalize_xg_matches(raw, provider=provider)
    output_path = out_dir / output_filename
    xg_matches.to_csv(output_path, index=False)

    valid_rows = int((xg_matches["home_xg"].notna() & xg_matches["away_xg"].notna()).sum())
    report = {
        "version": UNDERSTAT_DOWNLOAD_VERSION,
        "status": "ok" if valid_rows else "blocked",
        "provider": provider,
        "mode": "provider_or_manual_csv_import",
        "input_csv": str(input_csv),
        "requested_league_seasons": 0,
        "output_rows": int(len(xg_matches)),
        "rows_with_home_away_xg": valid_rows,
        "files": [str(output_path)],
        "failures": [],
        "terms_note": "User is responsible for provider terms/licensing for imported xG data.",
        "raw_data_changed": False,
    }
    return UnderstatDownloadOutputs(xg_matches, report)


def download_understat_xg(
    league_seasons: list[tuple[str, int]],
    out_dir: str | Path,
    *,
    timeout: int = 30,
    sleep_seconds: float = 1.0,
    write_empty_combined: bool = True,
) -> UnderstatDownloadOutputs:
    """Download match-level xG from Understat league pages for offline research.

    Direct Understat scraping is best-effort. If provider markup blocks scraping,
    this function still writes an empty canonical combined CSV by default so the
    rest of the enrichment pipeline can continue and report zero xG coverage.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    files: list[str] = []
    for league, season in league_seasons:
        league_slug = UNDERSTAT_LEAGUES.get(league, league)
        try:
            df = fetch_understat_league_season(league_slug, season, timeout=timeout)
            frames.append(df)
            path = out_dir / f"understat_xg_{league_slug}_{season}.csv"
            df.to_csv(path, index=False)
            files.append(str(path))
        except Exception as exc:  # pragma: no cover - depends on internet/provider markup
            failures.append({"league": league, "season": season, "error": str(exc)})
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    xg_matches = pd.concat(frames, ignore_index=True, sort=False) if frames else _empty_xg_matches()
    combined_path = out_dir / "understat_xg_matches.csv"
    if not xg_matches.empty or write_empty_combined:
        xg_matches.to_csv(combined_path, index=False)
        files.append(str(combined_path))

    status = "ok" if not failures and not xg_matches.empty else ("warning" if not xg_matches.empty else "blocked")
    report = {
        "version": UNDERSTAT_DOWNLOAD_VERSION,
        "status": status,
        "provider": "understat",
        "mode": "optional_research_scrape_cached_csv",
        "requested_league_seasons": len(league_seasons),
        "output_rows": int(len(xg_matches)),
        "files": files,
        "failures": failures,
        "direct_scrape_blocked": bool(failures and xg_matches.empty),
        "fallback": {
            "recommended": "import_provider_or_manual_xg_csv",
            "canonical_output": str(combined_path),
            "command": (
                "python scripts/import_xg_csv.py --input <provider_xg.csv> "
                "--provider <provider_name> --out-dir data/external/xg/understat"
            ),
        },
        "terms_note": "User should confirm provider terms/licensing before relying on scraped/imported data beyond local research.",
        "raw_data_changed": False,
    }
    return UnderstatDownloadOutputs(xg_matches, report)
