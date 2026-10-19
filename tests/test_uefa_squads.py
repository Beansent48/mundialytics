"""The squad table reaches Europe without disturbing the domestic half.

Before this, current_squads.csv held the big five and nothing else, so two
thirds of the clubs drawn into Europe had no players at all -- which is why a
Champions run in SquadLab could only ever name the drafted eleven and the other
35 clubs scored anonymously.

Adding 70 clubs to a file this many things read is the risky half. These tests
pin the two promises that make it safe: the domestic rows mean exactly what
they meant before, and nothing downstream starts treating an unmeasured club as
if it were measured.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mundialytics.props.player_props import UEFA_COMPETITIONS

ROOT = Path(__file__).resolve().parents[1]
SQUADS = ROOT / "data/processed/current_squads.csv"
UEFA_DIR = ROOT / "data/external/uefa"

pytestmark = pytest.mark.skipif(not SQUADS.exists(), reason="no squad table")


@pytest.fixture(scope="module")
def squads() -> pd.DataFrame:
    return pd.read_csv(SQUADS, low_memory=False)


def test_the_european_columns_are_additive(squads: pd.DataFrame) -> None:
    """Europe adds columns; it never redefines a domestic one. The scorer race,
    the props and squad_share all read `goals` and `apps` as league numbers."""
    for col in ("uefa_competition", "uefa_matches", "uefa_apps", "uefa_goals",
                "uefa_assists"):
        assert col in squads.columns, f"missing {col}"


def test_every_drawn_club_has_a_squad(squads: pd.DataFrame) -> None:
    """All 108 participants, or a Champions run still has anonymous clubs."""
    have = set(squads["team"])
    missing = {}
    for path in sorted(UEFA_DIR.glob("raw_*-league_2026.csv")):
        raw = pd.read_csv(path)
        from mundialytics.identity.normalization import canonical_team_name
        alias_path = ROOT / "data/curated/fixture_team_aliases.csv"
        alias = {}
        if alias_path.exists():
            a = pd.read_csv(alias_path)
            alias = dict(zip(a.iloc[:, 0].astype(str), a.iloc[:, 1].astype(str)))
        drawn = set(raw["Home Team"]) | set(raw["Away Team"])
        gone = sorted(t for t in drawn
                      if alias.get(t, canonical_team_name(t)) not in have)
        if gone:
            missing[path.name] = gone
    assert not missing, f"drawn clubs with no squad: {missing}"


def test_a_club_is_listed_once(squads: pd.DataFrame) -> None:
    """A big-five club appears in its league AND in Europe; the two rosters
    have to collapse, or Vinicius is listed twice for Real Madrid."""
    dupes = squads.duplicated(subset=["team", "player"])
    assert not dupes.any(), f"{int(dupes.sum())} duplicated (team, player) rows"


def test_a_domestic_club_keeps_its_league(squads: pd.DataFrame) -> None:
    """Filing Real Madrid under "Champions League" would drop it out of every
    path keyed on LaLiga."""
    big5 = {"LaLiga", "Premier League", "Serie A", "Bundesliga", "Ligue 1"}
    in_europe = squads[squads["uefa_competition"].fillna("") != ""]
    assert len(in_europe), "nobody is in Europe; the merge did nothing"

    domestic = in_europe[in_europe["competition"].isin(big5)]
    assert len(domestic), "no big-five club is marked as playing in Europe"
    assert set(domestic["uefa_competition"]) <= UEFA_COMPETITIONS


def test_european_only_clubs_carry_no_domestic_numbers(squads: pd.DataFrame) -> None:
    """Sabah have not scored in LaLiga, and a zero says so. Folding their
    European goals into `goals` would quietly change what the column means."""
    euro_only = squads[squads["competition"].isin(UEFA_COMPETITIONS)]
    assert len(euro_only) > 1000, "the European half is suspiciously small"
    for col in ("goals", "apps", "starts", "assists", "squad_matches"):
        assert (euro_only[col].fillna(0) == 0).all(), f"{col} is not domestic-only"


def test_europe_is_where_the_european_numbers_live(squads: pd.DataFrame) -> None:
    """The point of the whole exercise: real goals, by real players, in Europe."""
    assert squads["uefa_goals"].sum() > 0, "no European goals were recorded"
    scored = squads[squads["uefa_goals"] > 0]
    assert scored["team"].nunique() >= 10


def test_an_unmodelled_club_gets_no_prop_shortlist() -> None:
    """The failure this guard exists for: one coincidental name match used to
    be enough to install a "squad", and predict_fixture would then serve an
    authoritative-looking shortlist of one player for a club never measured."""
    from mundialytics.props.player_props import PlayerPropsModel

    players = pd.DataFrame({
        "player": ["Known Man"], "team": ["real madrid"], "pgroup": ["FW"],
        "last_date": [pd.Timestamp("2026-09-01")],
    })
    model = PlayerPropsModel.__new__(PlayerPropsModel)
    model._players = players
    model._rosters = {}
    model._fd_to_us = {}
    model._stale_rosters = {}

    cs = pd.DataFrame({
        "team": ["sabah", "sabah"],
        "player": ["Known Man", "Someone Else"],
        "pos_group": ["Forward", "Defender"],
        "competition": ["Champions League"] * 2,
    })
    model._apply_current_squads(cs)
    assert "sabah" not in model._rosters, "installed a squad for an unmodelled club"
    assert "sabah" in model._unmodelled_squads


def test_a_club_is_filed_under_one_competition(squads: pd.DataFrame) -> None:
    """The two-squads signature, and the one check that would have caught it.

    Stuttgart shipped on 2026-10-19 with 58 rows: 31 first-teamers under
    "Bundesliga" and 27 VfB Stuttgart II players under "Champions League",
    because the UEFA roster endpoint answers with the club's whole registered
    pool and the European half unioned its extra 27 in. Every row of this file
    is a distinct (team, player) -- the duplicate check above pins that -- so
    two competitions on one club means two DISJOINT sets of players under one
    name, which is exactly the pathology. The design has no other reading of
    the column: a big-five club carries its league, a club with no domestic
    coverage carries the tournament it was drawn into, and neither carries
    both.
    """
    split = squads.groupby("team")["competition"].nunique()
    bad = split[split > 1]
    detail = {
        t: squads[squads["team"] == t].groupby("competition").size().to_dict()
        for t in bad.index
    }
    assert bad.empty, f"clubs filed under two competitions: {detail}"


def test_no_club_carries_a_second_squad(squads: pd.DataFrame) -> None:
    """The other half of the same guard: too MANY players, not too few.

    The build only ever warned about thin squads (`clubes con plantilla corta`,
    and `clubes con menos de 16 cartas` downstream), so a club holding two
    squads passed every check it had. A registered first-team list runs to
    about 30 and tops out at 39 (Torreense, all 39 numbered); 46 is a reserve
    side merged in, not a deep bench.
    """
    size = squads.groupby("team").size()
    too_many = size[size > 45]
    assert too_many.empty, f"squads too large to be one squad: {too_many.to_dict()}"
