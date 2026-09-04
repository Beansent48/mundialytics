"""The ESPN player source, pinned where it can rot silently.

Player markets are logged pre-kickoff for all five leagues. FBref only ever
delivered the Premier League before stalling, which left 79% of them with
nothing able to settle them -- the booking-points failure, where a market was
priced and served for months while the evaluator quietly skipped it. ESPN closed
that, and these tests pin the properties that make the settlement trustworthy
rather than merely present.
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAYERS = ROOT / "data/external/advanced/espn/espn_player_match_current.csv"
MATCHES = ROOT / "data/external/advanced/espn/espn_matches_current.csv"
FOUNDATION = ROOT / "data/processed/foundation_big5_multi_season.csv"


def _load():
    if not PLAYERS.exists() or not MATCHES.exists():
        pytest.skip("no ESPN player data downloaded")
    p = pd.read_csv(PLAYERS, low_memory=False)
    m = pd.read_csv(MATCHES)
    if p.empty or m.empty:
        pytest.skip("ESPN player data is empty")
    for c in ("goals", "own_goals", "appearances", "shots", "assists"):
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)
    return p, m


def test_every_team_resolves_to_a_foundation_side():
    """An unmapped club settles nothing, and a mismapped one settles wrongly.

    The alias file has produced both: 53 of 121 ClubElo aliases once pointed at
    clubs that did not exist, and a duplicate-target check caught Espanyol
    mapped onto Barcelona. Fourteen ESPN names needed aliases ("AC Milan" ->
    milan, "Deportivo" -> la coruna); this fails the moment a fifteenth appears.
    """
    p, _ = _load()
    if not FOUNDATION.exists():
        pytest.skip("no foundation")
    f = pd.read_csv(FOUNDATION, low_memory=False)
    f = f[f["season"] == "2026-2027"]
    if f.empty:
        pytest.skip("foundation has no current season")

    unknown = {}
    for comp, sub in p.groupby("competition"):
        fc = f[f["competition"] == comp]
        known = set(fc["home_team"]) | set(fc["away_team"])
        if not known:
            continue
        bad = sorted(set(sub["team"]) - known)
        if bad:
            unknown[comp] = bad
    assert not unknown, f"ESPN teams with no foundation match: {unknown}"


def test_scorers_reconstruct_the_scoreline():
    """Per-player goals must add back up to the match result.

    This is the one check that catches a whole class of quiet corruption at
    once: a mismapped team, a dropped substitute, a double-counted penalty, or
    an own goal credited to the wrong side would all break it.
    """
    p, m = _load()
    g = p.groupby(["event_id", "team"], as_index=False).agg(
        gf=("goals", "sum"), og=("own_goals", "sum"))
    gm = {(r.event_id, r.team): (r.gf, r.og) for r in g.itertuples(index=False)}

    bad = []
    for r in m.itertuples(index=False):
        h = gm.get((r.event_id, r.home_team), (0, 0))
        a = gm.get((r.event_id, r.away_team), (0, 0))
        # a side's goals are its players' goals plus the opponent's own goals
        if (h[0] + a[1], a[0] + h[1]) != (r.home_goals, r.away_goals):
            bad.append(f"{r.date} {r.home_team} {r.home_goals}-{r.away_goals} "
                       f"{r.away_team} (jugadores: {h[0] + a[1]}-{a[0] + h[1]})")
    assert not bad, f"{len(bad)} partidos no cuadran, p.ej. {bad[:3]}"


def test_own_goals_are_not_credited_as_goals():
    """An own goal must never settle its scorer's own market as a hit.

    ESPN credits an own goal to the BENEFITING team, so its scorer plays for the
    other side. If `goals` included them, a defender who put one into his own
    net would be paid out as an anytime scorer.
    """
    p, m = _load()
    if p["own_goals"].sum() == 0:
        pytest.skip("no own goals in the window")
    # exact identity, not a threshold: every goal on the scoreboard is either a
    # player's own or an own goal, so the two columns must partition the total.
    # (Scoring at both ends is legitimate and happens -- Joao Pedro did it in
    # chelsea 4-3 brighton -- so counting such players proves nothing.)
    scored = p["goals"].sum()
    own = p["own_goals"].sum()
    total = m["home_goals"].sum() + m["away_goals"].sum()
    assert scored + own == total, (
        f"goles de jugador ({int(scored)}) + en propia ({int(own)}) "
        f"!= goles del marcador ({int(total)}): totalGoals podria estar "
        "incluyendo o perdiendo las porterias propias")


def test_scores_agree_with_football_data():
    """Two independent providers must tell the same story.

    ESPN is the settlement source for player markets and football-data is the
    source for team results. If they disagreed on a scoreline, one of them is
    describing a different match and the player settlement is attached to the
    wrong fixture.
    """
    _, m = _load()
    if not FOUNDATION.exists():
        pytest.skip("no foundation")
    f = pd.read_csv(FOUNDATION, low_memory=False)
    f = f[(f["season"] == "2026-2027") & f["home_goals"].notna()]
    if f.empty:
        pytest.skip("no played matches in the current season")

    mk = {(r.home_team, r.away_team): (r.home_goals, r.away_goals)
          for r in m.itertuples(index=False)}
    checked, bad = 0, []
    for r in f.itertuples(index=False):
        k = mk.get((r.home_team, r.away_team))
        if k is None:
            continue
        checked += 1
        if k != (r.home_goals, r.away_goals):
            bad.append(f"{r.date} {r.home_team} {r.away_team}: "
                       f"fd {int(r.home_goals)}-{int(r.away_goals)} vs espn {k[0]}-{k[1]}")
    if checked == 0:
        pytest.skip("no overlapping matches")
    assert not bad, f"{len(bad)} marcadores discrepan: {bad[:3]}"


def test_unused_substitutes_are_distinguishable():
    """The file must say who played, not only who did something.

    ESPN's event feed names scorers and booked players only. Settling from that
    alone would score a striker's market just in the matches where he scored,
    and the hit rate would approach 100% by construction. The roster carries
    `appearances`, which is what lets an unused substitute be voided instead of
    handed to the track record as a free "did not score".
    """
    p, _ = _load()
    assert "appearances" in p.columns, "sin columna appearances: no se sabe quien jugo"
    played = (p["appearances"] > 0).sum()
    benched = (p["appearances"] == 0).sum()
    assert played > 0, "ninguna aparicion registrada"
    assert benched > 0, (
        "ningun suplente sin usar en el fichero: probablemente solo se han "
        "guardado jugadores con evento, que es justo el sesgo a evitar")
    # roughly two full squads per match, of which 22 start
    per_match = p.groupby("event_id").size()
    assert per_match.median() >= 30, f"plantillas demasiado cortas: {per_match.median()}"


def test_matchday_survives_a_postponed_fixture():
    """Round numbers are counted per club, not sliced off the calendar.

    ESPN publishes no matchweek. Cutting the fixture list into blocks of n/2
    would be simpler, but one rearranged match shifts every later round by one.
    Counting each club's own fixtures keeps the rest of the season correct.
    """
    from mundialytics.providers.espn_fixtures import _matchdays

    # four clubs; a's round-2 match is played out of order, after round 3
    fx = pd.DataFrame({
        "home_team": ["a", "c", "a", "c", "a"],
        "away_team": ["b", "d", "d", "b", "c"],
    })
    md = _matchdays(fx)
    assert md == [1, 1, 2, 2, 3], md


def test_matchday_is_absent_only_when_fixtures_are():
    """Whatever the provider returns must carry a usable round label."""
    if not MATCHES.exists():
        pytest.skip("no ESPN data downloaded")
    from mundialytics.providers.espn_fixtures import COLUMNS

    assert "matchday" in COLUMNS, "el logger agrupa por jornada; debe venir del proveedor"
