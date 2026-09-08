"""The ESPN UEFA fixture path, pinned at its contract.

The European layer used to read fixtures from fixturedownload and fall back to a
cached CSV. When that host went dark the cache kept the page looking healthy
while results silently stopped arriving, so ESPN became the source and the cache
became a fallback. These tests hold the two things that make that a drop-in
replacement rather than an almost-one.
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _rows():
    """A season shaped like ESPN's, with the dates that once broke it."""
    ts = pd.to_datetime
    return [
        # league phase: two rounds of two matches among four clubs
        {"date": ts("2026-09-08T18:45Z"), "stage": None,
         "Home Team": "AEK Athens", "Away Team": "LASK Linz", "Result": "2 - 1"},
        {"date": ts("2026-09-08T20:00Z"), "stage": None,
         "Home Team": "Club Brugge", "Away Team": "Aston Villa", "Result": ""},
        # December: the day is 09, which a dayfirst parser reads as September
        {"date": ts("2026-12-09T20:00Z"), "stage": None,
         "Home Team": "LASK Linz", "Away Team": "Club Brugge", "Result": ""},
        {"date": ts("2026-12-09T20:00Z"), "stage": None,
         "Home Team": "Aston Villa", "Away Team": "AEK Athens", "Result": ""},
        # a two-legged knockout tie, second leg at the other ground
        {"date": ts("2027-03-10T20:00Z"), "stage": "QF",
         "Home Team": "AEK Athens", "Away Team": "Club Brugge", "Result": ""},
        {"date": ts("2027-03-17T20:00Z"), "stage": "QF",
         "Home Team": "Club Brugge", "Away Team": "AEK Athens", "Result": ""},
    ]


def test_dates_survive_a_dayfirst_parser():
    """The Date column must mean the same thing to every existing consumer.

    fixturedownload writes DD/MM/YYYY, so the app, the round logger and the Elo
    roll-forward all parse this column with dayfirst=True. Emitting ISO instead
    reinterprets every date whose day is 12 or under: the December matchday
    (09/12) came back as 9 September and was logged as an upcoming fixture,
    while matchday 1 on 08/09 was read as 8 August and dropped as past. The
    fixtures were real, the count looked plausible, and the round was wrong.
    """
    from mundialytics.statistical_core.competition.european import shape_uefa_rows

    df = shape_uefa_rows(_rows())
    parsed = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    assert parsed.notna().all(), "fechas que un parser dayfirst no entiende"

    original = pd.Series([r["date"] for r in _rows()]).dt.tz_localize(None).sort_values()
    assert list(parsed) == list(original), (
        "las fechas cambian de significado al pasar por dayfirst=True")
    # the specific pair that broke it
    assert (parsed.dt.month == 12).sum() == 2, "la jornada de diciembre se ha movido"
    assert (parsed.dt.month == 9).sum() == 2, "la jornada 1 de septiembre se ha movido"


def test_league_rounds_are_numbered_per_club():
    """parse_fixturedownload splits league from knockout on Round Number being
    numeric, so the league phase must carry a number and the knockout must not.
    """
    from mundialytics.statistical_core.competition.european import shape_uefa_rows

    df = shape_uefa_rows(_rows())
    rn = pd.to_numeric(df["Round Number"], errors="coerce")
    assert rn.notna().sum() == 4, "la fase de liga debe llevar numero de jornada"
    assert sorted(rn.dropna().astype(int)) == [1, 1, 2, 2]
    assert rn.isna().sum() == 2, "la eliminatoria no debe llevar numero"


def test_knockout_legs_are_numbered_by_the_tie():
    """Leg 2 is the return, whoever is at home.

    The pairing is keyed on the unordered pair: taking the row order alone would
    call both matches leg 1, because the home and away clubs swap between legs.
    """
    from mundialytics.statistical_core.competition.european import shape_uefa_rows

    df = shape_uefa_rows(_rows())
    ko = df[pd.to_numeric(df["Round Number"], errors="coerce").isna()]
    assert list(ko["Round Number"]) == ["QF Game 1", "QF Game 2"], list(ko["Round Number"])


def test_it_parses_back_into_the_shape_the_simulator_expects():
    """End of the contract: what comes out must go through the existing parser."""
    from mundialytics.statistical_core.competition.european import (
        parse_fixturedownload, shape_uefa_rows)

    df = shape_uefa_rows(_rows())
    clubs = {"AEK Athens", "LASK Linz", "Club Brugge", "Aston Villa"}
    league, ko = parse_fixturedownload(df, lambda n: n if n in clubs else None)

    assert len(league) == 4 and len(ko) == 2
    assert list(league.columns) == ["home", "away", "home_goals", "away_goals"]
    played = league.dropna(subset=["home_goals"])
    assert len(played) == 1 and played.iloc[0]["home_goals"] == 2
    assert set(ko["round"]) == {"qf"} and sorted(ko["leg"]) == [1, 2]


def test_empty_input_is_none_not_a_crash():
    """A quiet provider must fall through to the next source, never raise."""
    from mundialytics.statistical_core.competition.european import shape_uefa_rows

    assert shape_uefa_rows([]) is None


def test_uefa_clubs_resolve_onto_the_rating_table():
    """A club with no rating is dropped; a club with the WRONG one is invisible.

    ESPN spells these differently again -- "Heart of Midlothian" where ClubElo
    has "Hearts", "Union St.-Gilloise" which fuzzy-matches onto Union Berlin.
    Two stay deliberately unresolved because ClubElo carries no rating for them
    at all: Slavia Praha (it has Sparta and Bohemians, and "Slavia Sofia" is a
    different club in a different country) and Torreense.
    """
    cache_dir = ROOT / "data/external/uefa"
    elo_file = ROOT / "data/processed/clubelo_local.csv"
    if not elo_file.exists() or not cache_dir.exists():
        pytest.skip("no local ClubElo / UEFA cache")

    from mundialytics.statistical_core.competition.european import make_resolver

    elo = pd.read_csv(elo_file)
    col = next((c for c in elo.columns if c.lower() in ("club", "team", "name")), None)
    if col is None:
        pytest.skip("unexpected ClubElo layout")
    resolver = make_resolver(set(elo[col].astype(str)))

    known_gaps = {"slavia prague", "slavia praha", "torreense"}
    unresolved, targets = [], {}
    for f in sorted(cache_dir.glob("raw_*_2026.csv")):
        raw = pd.read_csv(f)
        if not {"Home Team", "Away Team"} <= set(raw.columns):
            continue
        for club in set(raw["Home Team"]) | set(raw["Away Team"]):
            hit = resolver(str(club))
            if hit is None:
                if str(club).lower() not in known_gaps:
                    unresolved.append((f.name, club))
            else:
                targets.setdefault((f.name, hit), set()).add(str(club))

    assert not unresolved, f"clubes UEFA sin alias: {unresolved[:6]}"
    # two different clubs sharing one rating is the failure that put Espanyol
    # onto Barcelona; it must never happen silently
    clashes = {k: sorted(v) for k, v in targets.items() if len(v) > 1}
    assert not clashes, f"dos clubes apuntando al mismo Elo: {clashes}"
