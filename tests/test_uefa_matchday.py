"""European ties appear on the day screen, priced by the European model.

The matchday browser only ever knew the five domestic leagues, because
everything it touches is keyed on cat.COMPETITIONS -- which assumes a big-five
foundation, a trusted calendar and a match page of its own. Europe is served
from its own catalogue and merged at the edge instead, so these tests pin the
two things that merge has to get right: the payload the client already renders,
and an honest account of what priced it.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import pytest

from api import european as eu

DOMESTIC_KEYS = {
    "slug", "competition", "competitionName", "season", "matchday", "kickoff",
    "home", "away", "homeSlug", "awaySlug", "played", "score", "probabilities",
}


def _a_busy_day() -> date | None:
    """The day in the season with the most European football on it."""
    start = date(2026, 7, 1)
    counts = eu.day_counts(start, start + timedelta(days=365))
    if not counts:
        return None
    return date.fromisoformat(max(counts, key=lambda k: counts[k]))


@pytest.fixture(scope="module")
def busy() -> date:
    day = _a_busy_day()
    if day is None:
        pytest.skip("no European fixtures on disk")
    return day


def test_the_payload_is_the_one_the_day_screen_renders(busy: date) -> None:
    """Same keys as the domestic fixture, or the client needs a second path."""
    fixtures = eu.day_fixtures(busy)
    assert fixtures, f"no European fixtures on {busy}"
    for f in fixtures:
        assert DOMESTIC_KEYS <= set(f), f"missing {DOMESTIC_KEYS - set(f)}"


def test_a_european_tie_says_it_has_no_analysis_page(busy: date) -> None:
    """Without this the row links to /matchday/champions-league/<slug>, which
    does not exist and answers 404."""
    for f in eu.day_fixtures(busy):
        assert f["analysis"] is False
        assert f["model"] == "elo"


def test_a_played_tie_carries_its_score_and_no_price(busy: date) -> None:
    """Quoting a pre-match price beside a final score invites reading one as
    the other -- the same rule the domestic payload follows."""
    seen_played = False
    for day in (busy, date(2026, 9, 16), date(2026, 9, 8)):
        for f in eu.day_fixtures(day):
            if f["played"]:
                seen_played = True
                assert f["probabilities"] is None
                assert f["score"] is not None
            else:
                assert f["score"] is None
    assert seen_played, "no played European tie anywhere; test proved nothing"


def test_an_unplayed_tie_is_priced(busy: date) -> None:
    pending = [f for f in eu.day_fixtures(busy) if not f["played"]]
    if not pending:
        pytest.skip("every tie on the busiest day has been played")
    priced = [f for f in pending if f["probabilities"]]
    assert len(priced) >= len(pending) * 0.8, "most pending ties have no price"
    for f in priced:
        p = f["probabilities"]
        assert abs(p["home"] + p["draw"] + p["away"] - 1.0) < 0.02


def test_the_day_count_agrees_with_the_day_itself(busy: date) -> None:
    counts = eu.day_counts(busy, busy)
    assert counts.get(busy.isoformat(), 0) == len(eu.day_fixtures(busy))


def test_the_fingerprint_moves_with_the_files(tmp_path, monkeypatch) -> None:
    """It is a cache key: a European result landing has to push the cached day
    aside, and the domestic feed's own clock will not do that.

    Driven against a temporary directory rather than the real one. The fixture
    files are not in the repository, so on a clean checkout the honest answer is
    an empty fingerprint -- asserting it was non-empty tested that someone had
    run the pipeline, not that the mechanism works.
    """
    monkeypatch.setattr(eu, "UEFA_DIR", tmp_path)
    assert eu.fixtures_fingerprint() == "", "an empty directory has no fingerprint"

    fixture = tmp_path / "raw_champions-league_2026.csv"
    fixture.write_text("Round Number,Date,Home Team,Away Team,Result" + chr(10),
                       encoding="utf-8")
    os.utime(fixture, (1_700_000_000, 1_700_000_000))
    first = eu.fixtures_fingerprint()
    assert first, "a fixture file on disk must produce a fingerprint"
    assert first == eu.fixtures_fingerprint(), "unchanged files must hash the same"

    # a result landing rewrites the file; the cached day has to notice
    os.utime(fixture, (1_700_003_600, 1_700_003_600))
    assert eu.fixtures_fingerprint() != first


def test_a_january_date_belongs_to_the_season_that_started_in_july() -> None:
    """European seasons run July-June; asking for the calendar year would look
    up a season that has not been drawn."""
    assert eu._season_year(date(2027, 1, 20)) == 2026
    assert eu._season_year(date(2026, 9, 16)) == 2026
    assert eu._season_year(date(2026, 7, 1)) == 2026
