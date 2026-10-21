"""
Competitions, calendars and slugs — the shared vocabulary of the HTTP API.

A slug is what appears in a URL (`laliga`, `valencia-barcelona`) and it has to
round-trip: every page the site links to must be reachable by parsing its own
address back into the names the engine uses. Doing that in one module is what
stops a match page from 404-ing because two files disagreed on how to spell a
team with an accent.
"""
from __future__ import annotations

import os
import re
import threading
import time
import unicodedata
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CALENDAR_DIR = ROOT / "data/processed/fixture_calendars"

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


CALENDAR_TTL_SECONDS = 1800
CALENDAR_COLUMNS = ["date", "matchday", "home_team", "away_team", "completed",
                    "home_goals", "away_goals"]


def _snapshot_path(comp_id: str, season: str) -> Path:
    return CALENDAR_DIR / f"{slugify(comp_id)}_{season}.csv"


def _shape(fx: pd.DataFrame | None) -> pd.DataFrame:
    """A provider frame in the shape every caller here expects."""
    if fx is None or fx.empty or "matchday" not in fx.columns:
        return pd.DataFrame(columns=CALENDAR_COLUMNS)
    out = fx[["date", "matchday", "home_team", "away_team", "completed"]].copy()
    out["home_goals"] = pd.to_numeric(fx.get("home_goals"), errors="coerce")
    out["away_goals"] = pd.to_numeric(fx.get("away_goals"), errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["matchday"] = pd.to_numeric(out["matchday"], errors="coerce")
    # `completed` has to survive a CSV round trip as a BOOLEAN. Read back as the
    # string "False" it would pass `astype(bool)` -- any non-empty string does --
    # and `trusted_calendar` and the Golden Boot race both ask exactly that
    # question, so a settled season would read as 380 fixtures still to play.
    out["completed"] = (out["completed"].astype(str).str.strip().str.lower()
                        .isin(("true", "1", "yes")))
    return out


def _fetch_calendar(comp_id: str, season: str) -> pd.DataFrame:
    """One league-season straight from the provider. Empty if it is unreachable."""
    try:
        from mundialytics.providers.espn_fixtures import fetch_season_fixtures

        return _shape(fetch_season_fixtures(comp_id, season, root=ROOT))
    except Exception:
        return pd.DataFrame(columns=CALENDAR_COLUMNS)


# One refresh at a time per league-season. `lru_cache` does not de-duplicate
# calls in flight and FastAPI runs sync endpoints in a threadpool, so the two
# requests the matchday screen fires together used to fetch all five calendars
# twice over: 14 s of work measured as 83 s of contention.
_REFRESH_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
# When the fetch fails there is nothing on disk to serve, and without this every
# single request would start another 45-second provider timeout. The lru_cache
# this replaced got that right by accident: it cached the empty answer too.
_LAST_TRY: dict[tuple[str, str], float] = {}


def _refresh_lock(key: tuple[str, str]) -> threading.Lock:
    with _LOCKS_GUARD:
        return _REFRESH_LOCKS.setdefault(key, threading.Lock())


def _age(path: Path) -> float:
    """Seconds since the snapshot was written; `inf` when there is none."""
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return float("inf")


def refresh_calendar(comp_id: str, season: str, block: bool = True,
                     max_age: float = CALENDAR_TTL_SECONDS) -> bool:
    """Fetch one league-season and write its snapshot.

    Returns True when a usable snapshot is on disk afterwards — written now, or
    already younger than `max_age` because a concurrent caller got there first.
    False means the provider gave us nothing and there is still no snapshot (or
    only the old one), which is a thing to report, not to serve.

    An empty answer never replaces a snapshot: `fetch_season_fixtures` swallows
    its errors and returns an empty frame, and letting that through is how a
    matchday renders blank on a bad provider day.
    """
    key = (comp_id, season)
    path = _snapshot_path(comp_id, season)
    lock = _refresh_lock(key)
    if not lock.acquire(blocking=block):
        return False                    # someone else is already on it
    try:
        if _age(path) <= max_age:
            return True                 # a waiting caller found it already done
        cal = _fetch_calendar(comp_id, season)
        _LAST_TRY[key] = time.time()
        if cal.empty:
            return False
        CALENDAR_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".csv.tmp")
        cal.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, path)           # a reader never sees half a file
        return True
    except OSError:
        return False
    finally:
        lock.release()


def refresh_calendars(pairs) -> dict[tuple[str, str], bool]:
    """Write every snapshot, blocking, however fresh the last one is.

    Takes (competition id, season) pairs — `seasons_to_snapshot` builds them.
    What the daily refresh calls: that pass talks to ESPN anyway, so it is the
    right place to pay for this, and the serving path then only ever reads.
    """
    return {(comp, season): refresh_calendar(comp, season, max_age=0.0)
            for comp, season in pairs}


def has_snapshot(comp_id: str, season: str) -> bool:
    """Whether a calendar can be served for this league-season without a fetch."""
    return _snapshot_path(comp_id, season).exists()


@lru_cache(maxsize=32)
def _read_snapshot(comp_id: str, season: str, _stamp: int) -> pd.DataFrame:
    """The snapshot on disk, keyed on its mtime so a refresh is picked up at once."""
    path = _snapshot_path(comp_id, season)
    try:
        return _shape(pd.read_csv(path))
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, KeyError):
        return pd.DataFrame(columns=CALENDAR_COLUMNS)


def season_calendar(comp_id: str, season: str) -> pd.DataFrame:
    """The published calendar for a league-season: every fixture, played or not.

    The foundation holds only played matches, so a calendar derived from it can
    never contain a fixture before kick-off — the one thing this product is for.
    Returns an empty frame when there is no snapshot and the provider is
    unreachable; callers fall back.

    Served from a snapshot on disk, refreshed from the provider *behind* the
    answer. ESPN used to be the serving source — two HTTP calls per
    league-season, no disk cache of any kind, so 10.4 s for the five leagues on
    every cold start (measured 2026-10-21) and minutes on a slow day, with the
    visitor waiting for all of it. Now the provider is the refresh source and
    the disk is the serving source, which is how the UEFA calendar has worked
    since September. Staleness is bounded by CALENDAR_TTL_SECONDS and the
    freshness that matters on a matchday arrives the same way it always did:
    the next request after the refresh lands, keyed on the file's mtime.
    """
    path = _snapshot_path(comp_id, season)
    age = _age(path)
    # At most one provider attempt per league-season per window, whether it
    # succeeds or not: a dead provider must not cost a timeout per request.
    may_try = time.time() - _LAST_TRY.get((comp_id, season), 0.0) > CALENDAR_TTL_SECONDS
    if age == float("inf"):
        # nothing on disk at all: the very first run has to wait for the fetch
        if may_try:
            refresh_calendar(comp_id, season, block=True)
    elif age > CALENDAR_TTL_SECONDS and may_try:
        # stale: serve what we have and refresh behind the visitor's back
        threading.Thread(target=refresh_calendar, args=(comp_id, season, False),
                         daemon=True, name=f"calendar-{slugify(comp_id)}").start()
    stamp = 0
    try:
        stamp = int(path.stat().st_mtime)
    except OSError:
        pass
    return _read_snapshot(comp_id, season, stamp)


def serving_season(today: date | None = None) -> str:
    """The season the serving path asks for, read off the calendar date.

    Deliberately not `current_season`, which is the newest season in the
    foundation. On matchday one the foundation holds no match of the new season
    at all, and the new season's clubs are what a price is about — so `_predict`
    asks for the season the date implies. The two agree for eleven months of the
    year and disagree in July and August, which is exactly when it matters.

    This is the one definition: anything that has to agree with what the serving
    path requests — above all whatever writes the calendar snapshots it reads —
    must call this rather than repeat the arithmetic.
    """
    d = today or date.today()
    y = d.year if d.month >= 7 else d.year - 1
    return f"{y}-{y + 1}"


def current_season(df: pd.DataFrame, comp_id: str) -> str:
    seasons = df.loc[df["competition"] == comp_id, "season"]
    return str(sorted(seasons.unique())[-1]) if len(seasons) else ""


def seasons_to_snapshot(df: pd.DataFrame, today: date | None = None) -> list[tuple[str, str]]:
    """Every (competition, season) pair the serving path can ask for.

    Both seasons, when they differ. `/fixtures`, `/match`, `/league` and
    `/awards` work from `current_season` (the foundation's newest), while
    `_predict` works from `serving_season` (the date's). Writing only the first
    left the second with no snapshot every July and August: the site would fall
    back to a blocking provider fetch once per window, and the daily log would
    report that everything went fine.
    """
    serving = serving_season(today)
    pairs: list[tuple[str, str]] = []
    for meta in COMPETITIONS.values():
        comp = meta["id"]
        for season in dict.fromkeys(s for s in (current_season(df, comp), serving) if s):
            pairs.append((comp, season))
    return pairs


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
