"""The league calendar is served from disk, not from the provider.

ESPN used to be the serving source for `api.catalogue.season_calendar`: two HTTP
calls per league-season with no disk cache of any kind, so every cold start paid
10.4 s for the five leagues (measured 2026-10-21) and a slow provider day cost
minutes — with the visitor waiting for all of it. Worse,
`fetch_season_fixtures` swallows its errors and returns an empty frame, which is
how a matchday renders blank.

What is pinned here is the part that makes the snapshot safe rather than the
part that makes it fast: the boolean that decides whether a fixture has been
played has to survive a CSV round trip, an empty answer must never replace a
good snapshot, and a stale snapshot must be served rather than waited on.

Run:  .venv/Scripts/python.exe -m pytest tests/test_calendar_snapshot.py -q
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

pytest.importorskip("fastapi")

from api import catalogue as cat  # noqa: E402


def _calendar() -> pd.DataFrame:
    """Two played fixtures and two still to come, as the provider shapes them."""
    return cat._shape(pd.DataFrame({
        "date": ["2026-08-15", "2026-08-16", "2027-05-20", "2027-05-21"],
        "matchday": [1, 1, 38, 38],
        "home_team": ["valencia", "barcelona", "sevilla", "real madrid"],
        "away_team": ["betis", "getafe", "osasuna", "girona"],
        "completed": [True, True, False, False],
        "home_goals": [2, 1, None, None],
        "away_goals": [0, 1, None, None],
    }))


@pytest.fixture
def snapshots(tmp_path, monkeypatch):
    """Point the snapshot store at a temporary directory, caches cleared."""
    monkeypatch.setattr(cat, "CALENDAR_DIR", tmp_path)
    monkeypatch.setattr(cat, "_LAST_TRY", {})
    cat._read_snapshot.cache_clear()
    yield tmp_path
    cat._read_snapshot.cache_clear()


def test_completed_survives_the_csv_round_trip(snapshots, monkeypatch):
    """The one that bites: read back as the string "False" it passes
    `astype(bool)`, which is exactly the question `trusted_calendar` and the
    Golden Boot race ask — a settled season would read as all still to play."""
    monkeypatch.setattr(cat, "_fetch_calendar", lambda c, s: _calendar())
    assert cat.refresh_calendar("LaLiga", "2026-2027") is True

    cal = cat.season_calendar("LaLiga", "2026-2027")
    assert cal["completed"].dtype == bool
    assert list(cal["completed"]) == [True, True, False, False]
    assert int(cal["completed"].astype(bool).sum()) == 2
    assert len(cal[~cal["completed"].astype(bool)]) == 2


def test_the_rest_of_the_frame_round_trips_too(snapshots, monkeypatch):
    monkeypatch.setattr(cat, "_fetch_calendar", lambda c, s: _calendar())
    cat.refresh_calendar("LaLiga", "2026-2027")

    cal = cat.season_calendar("LaLiga", "2026-2027")
    assert list(cal.columns) == cat.CALENDAR_COLUMNS
    assert str(cal["date"].dtype).startswith("datetime64")
    assert sorted(int(m) for m in cal["matchday"].unique()) == [1, 38]
    assert cal["home_goals"].notna().sum() == 2        # NaN stays NaN, not 0
    assert list(cal["home_team"]) == ["valencia", "barcelona", "sevilla", "real madrid"]


def test_an_empty_answer_never_replaces_a_snapshot(snapshots, monkeypatch):
    """A blank matchday is worse than a stale one. `fetch_season_fixtures`
    returns an empty frame for every failure, so this is the common case."""
    monkeypatch.setattr(cat, "_fetch_calendar", lambda c, s: _calendar())
    cat.refresh_calendar("LaLiga", "2026-2027")
    path = cat._snapshot_path("LaLiga", "2026-2027")
    before = path.read_bytes()

    monkeypatch.setattr(cat, "_fetch_calendar",
                        lambda c, s: pd.DataFrame(columns=cat.CALENDAR_COLUMNS))
    assert cat.refresh_calendar("LaLiga", "2026-2027", max_age=0.0) is False
    assert path.read_bytes() == before
    assert len(cat.season_calendar("LaLiga", "2026-2027")) == 4


def test_a_fresh_snapshot_does_not_touch_the_provider(snapshots, monkeypatch):
    monkeypatch.setattr(cat, "_fetch_calendar", lambda c, s: _calendar())
    cat.refresh_calendar("LaLiga", "2026-2027")

    calls = []
    monkeypatch.setattr(cat, "_fetch_calendar",
                        lambda c, s: calls.append((c, s)) or _calendar())
    for _ in range(3):
        assert len(cat.season_calendar("LaLiga", "2026-2027")) == 4
    assert calls == []


def test_a_stale_snapshot_is_served_not_waited_for(snapshots, monkeypatch):
    """Stale-while-revalidate: the answer comes from disk and the refresh
    happens behind it, so a slow provider never becomes a slow page."""
    monkeypatch.setattr(cat, "_fetch_calendar", lambda c, s: _calendar())
    cat.refresh_calendar("LaLiga", "2026-2027")
    path = cat._snapshot_path("LaLiga", "2026-2027")
    old = time.time() - (cat.CALENDAR_TTL_SECONDS + 60)
    os.utime(path, (old, old))

    def slow(_c, _s):
        time.sleep(2.0)
        return _calendar()

    monkeypatch.setattr(cat, "_fetch_calendar", slow)
    t0 = time.perf_counter()
    cal = cat.season_calendar("LaLiga", "2026-2027")
    assert time.perf_counter() - t0 < 1.0, "the visitor waited for the refresh"
    assert len(cal) == 4


def test_no_snapshot_and_a_dead_provider_is_an_empty_frame(snapshots, monkeypatch):
    """Callers fall back on an empty calendar; they must not see an exception."""
    monkeypatch.setattr(cat, "_fetch_calendar",
                        lambda c, s: pd.DataFrame(columns=cat.CALENDAR_COLUMNS))
    cal = cat.season_calendar("Serie A", "2026-2027")
    assert cal.empty
    assert list(cal.columns) == cat.CALENDAR_COLUMNS


# ── the season the snapshot is written for ───────────────────────────────────
def _foundation(last_season: str) -> pd.DataFrame:
    """A foundation frame whose newest season is `last_season`, all five leagues."""
    return pd.DataFrame([
        {"competition": meta["id"], "season": s}
        for meta in cat.COMPETITIONS.values()
        for s in ("2025-2026", last_season)
    ])


def test_the_serving_season_follows_the_date_not_the_foundation():
    """July starts the new season: the foundation still holds none of it, and
    the new season's clubs are what a price is about."""
    assert cat.serving_season(date(2027, 6, 30)) == "2026-2027"
    assert cat.serving_season(date(2027, 7, 1)) == "2027-2028"
    assert cat.serving_season(date(2027, 12, 31)) == "2027-2028"
    assert cat.serving_season(date(2028, 1, 1)) == "2027-2028"


def test_the_daily_pass_writes_the_season_the_serving_path_asks_for(monkeypatch):
    """The bug this pins: the pass wrote `current_season` (the foundation's
    newest) while `_predict` asked for `serving_season` (the date's). In July
    they differ, the snapshot the serving path wanted did not exist, and the
    site fell back to a blocking provider fetch while the daily log said every
    calendar was fine."""
    import api.main as m

    df = _foundation("2026-2027")
    july = date(2027, 7, 15)
    pairs = cat.seasons_to_snapshot(df, today=july)

    wanted = cat.serving_season(july)
    assert wanted == "2027-2028"
    for meta in cat.COMPETITIONS.values():
        assert (meta["id"], wanted) in pairs, f"no snapshot planned for {meta['id']} {wanted}"
        # and the foundation's own season is still written, for the endpoints
        # that work from it (/fixtures, /match, /league, /awards)
        assert (meta["id"], "2026-2027") in pairs

    # the serving path really does ask for that season, and through this helper
    asked = []
    monkeypatch.setattr(cat, "serving_season", lambda today=None: "1999-2000")
    monkeypatch.setattr(cat, "season_calendar",
                        lambda comp, season: asked.append((comp, season)) or pd.DataFrame())
    monkeypatch.setattr(m, "_models", lambda: (object(), [], df))
    m._predict("valencia", "betis", "LaLiga")
    assert asked == [("LaLiga", "1999-2000")]


def test_one_season_is_planned_once(monkeypatch):
    """In the eleven months when the two agree, no duplicate fetch."""
    df = _foundation("2026-2027")
    pairs = cat.seasons_to_snapshot(df, today=date(2027, 3, 1))
    assert len(pairs) == len(set(pairs)) == len(cat.COMPETITIONS)


def test_a_dead_provider_is_retried_at_most_once_per_window(snapshots, monkeypatch):
    """Without this every request starts another 45 s provider timeout — the
    lru_cache this replaced got that right by accident, caching the failure."""
    calls = []

    def dead(c, s):
        calls.append((c, s))
        return pd.DataFrame(columns=cat.CALENDAR_COLUMNS)

    monkeypatch.setattr(cat, "_fetch_calendar", dead)
    for _ in range(4):
        assert cat.season_calendar("Serie A", "2026-2027").empty
    assert len(calls) == 1
