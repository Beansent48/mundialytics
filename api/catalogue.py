"""
Competitions, calendars and slugs — the shared vocabulary of the HTTP API.

A slug is what appears in a URL (`laliga`, `valencia-barcelona`) and it has to
round-trip: every page the site links to must be reachable by parsing its own
address back into the names the engine uses. Doing that in one module is what
stops a match page from 404-ing because two files disagreed on how to spell a
team with an accent.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

COMPETITIONS: dict[str, dict] = {
    "laliga": {"id": "LaLiga", "name": "LaLiga", "country": "ES", "type": "league"},
    "premier-league": {"id": "Premier League", "name": "Premier League", "country": "EN", "type": "league"},
    "serie-a": {"id": "Serie A", "name": "Serie A", "country": "IT", "type": "league"},
    "bundesliga": {"id": "Bundesliga", "name": "Bundesliga", "country": "DE", "type": "league"},
    "ligue-1": {"id": "Ligue 1", "name": "Ligue 1", "country": "FR", "type": "league"},
}

BY_COMP_ID = {c["id"]: slug for slug, c in COMPETITIONS.items()}


def slugify(value: str) -> str:
    """`Ath Bilbao` -> `ath-bilbao`, accents folded, so it survives a URL."""
    s = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def fixture_slug(home: str, away: str) -> str:
    return f"{slugify(home)}-vs-{slugify(away)}"


@lru_cache(maxsize=32)
def season_calendar(comp_id: str, season: str) -> pd.DataFrame:
    """The published calendar for a league-season: every fixture, played or not.

    The foundation holds only played matches, so a calendar derived from it can
    never contain a fixture before kick-off — the one thing this product is for.
    Returns an empty frame when the provider is unreachable; callers fall back.
    """
    try:
        from mundialytics.providers.espn_fixtures import fetch_season_fixtures

        fx = fetch_season_fixtures(comp_id, season, root=ROOT)
    except Exception:
        return pd.DataFrame()
    if fx is None or fx.empty or "matchday" not in fx.columns:
        return pd.DataFrame()
    out = fx[["date", "matchday", "home_team", "away_team", "completed"]].copy()
    out["home_goals"] = pd.to_numeric(fx.get("home_goals"), errors="coerce")
    out["away_goals"] = pd.to_numeric(fx.get("away_goals"), errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def current_season(df: pd.DataFrame, comp_id: str) -> str:
    seasons = df.loc[df["competition"] == comp_id, "season"]
    return str(sorted(seasons.unique())[-1]) if len(seasons) else ""


def seasons_for(df: pd.DataFrame, comp_id: str) -> list[str]:
    seasons = df.loc[df["competition"] == comp_id, "season"].unique()
    return [str(s) for s in sorted(seasons, reverse=True)]


def trusted_calendar(comp_id: str, season: str, df: pd.DataFrame) -> pd.DataFrame | None:
    """The calendar, but only when it accounts for every result we already hold.

    ESPN's older seasons fail to map about a tenth of club names; an unmapped row
    comes back goalless, which would present a match settled years ago as a
    fixture still to play, with a prediction attached. Coverage is the test.
    """
    cal = season_calendar(comp_id, season)
    if cal.empty:
        return None
    played = df[(df["competition"] == comp_id) & (df["season"] == season)]
    merged = cal.merge(
        played[["home_team", "away_team", "home_goals"]].rename(
            columns={"home_goals": "_found"}
        ),
        on=["home_team", "away_team"],
        how="left",
    )
    if int(merged["_found"].notna().sum()) < len(played):
        return None
    return cal


def upcoming_window(df: pd.DataFrame, days: int = 8) -> pd.DataFrame:
    """Every fixture across the covered leagues between today and `days` ahead.

    This is what the matchday screen opens with: what is about to be played,
    everywhere, rather than a competition the visitor has to choose first.
    """
    today = pd.Timestamp(date.today())
    horizon = today + timedelta(days=days)
    frames = []
    for slug, meta in COMPETITIONS.items():
        season = current_season(df, meta["id"])
        if not season:
            continue
        cal = season_calendar(meta["id"], season)
        if cal.empty:
            continue
        window = cal[(cal["date"] >= today - timedelta(days=1)) & (cal["date"] <= horizon)].copy()
        if window.empty:
            continue
        window["competition"] = slug
        window["season"] = season
        frames.append(window)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date")

def window(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Every fixture across the covered leagues between two dates, inclusive."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    for slug, meta in COMPETITIONS.items():
        season = current_season(df, meta["id"])
        if not season:
            continue
        cal = season_calendar(meta["id"], season)
        if cal.empty:
            continue
        sel = cal[(cal["date"] >= lo) & (cal["date"] <= hi)].copy()
        if sel.empty:
            continue
        sel["competition"] = slug
        sel["season"] = season
        frames.append(sel)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date")


def day_counts(df: pd.DataFrame, start: date, end: date) -> dict[str, int]:
    """Fixtures per calendar day — what the date strip needs to mark a day.

    Deliberately does not price anything: the strip only has to know whether a
    day has football on it, and predicting eighty matches to draw a dot would
    make moving through the calendar cost seconds.
    """
    w = window(df, start, end)
    if w.empty:
        return {}
    counts = w.groupby(w["date"].dt.date).size()
    return {d.isoformat(): int(n) for d, n in counts.items()}
