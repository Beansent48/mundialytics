"""The current-squad layer, pinned where it rots silently.

Both player-facing surfaces used to read the squad off appearance history, and
history is not a squad list. The props model asked "who featured in this club's
last ten Understat games" — for Hull, promoted after nine years away, that
returned the 2016/17 side and priced Harry Maguire to score in 2026/27.
SquadLab grouped career profiles by the club a player is remembered for and
fielded Xavi, Puyol and Messi for Barcelona.

Nothing failed loudly when it happened, which is the point of these tests: the
squad going stale looks exactly like the squad being fine until someone reads
the names. They pin the properties that make a squad current, not merely
present.
"""
from pathlib import Path

import pandas as pd
import pytest

from mundialytics.identity.current_squads import (
    NameIndex, full_key, load_current_squads, position_group, short_key,
    squads_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
SQUADS = ROOT / "data/processed/current_squads.csv"


def _squads() -> pd.DataFrame:
    df = load_current_squads()
    if df.empty:
        pytest.skip("current_squads.csv not built")
    return df


# ── name keys ──────────────────────────────────────────────────────────────────
def test_short_key_survives_a_hyphenated_surname():
    """The providers spell Mbappé two ways and only the short key reunites them.

    Folding the hyphen to a space before splitting makes the surname "Lottin",
    which matches nothing — the first version of this did exactly that and
    dropped Real Madrid's top scorer out of his own squad.
    """
    assert short_key("Kylian Mbappe-Lottin") == short_key("Kylian Mbappé")
    assert full_key("Kylian Mbappe-Lottin") != full_key("Kylian Mbappé")


def test_short_key_does_not_merge_two_players_with_one_surname():
    assert short_key("Ethan Mbappé") != short_key("Kylian Mbappé")


def test_keys_fold_accents_and_punctuation():
    assert full_key("Vinícius Júnior") == "vinicius junior"
    assert full_key("Nico O'Reilly") == "nico o reilly"
    assert short_key("Angeliño") == "angelino"


# ── positions ──────────────────────────────────────────────────────────────────
def test_sub_is_not_a_position():
    """ESPN labels every substitute "SUB" in its MATCH rosters.

    Read as an abbreviation it ends in a B, lands on the wing-back rule, and
    turns two thirds of the league into defenders — which is what the first
    build of the squad table did.
    """
    assert position_group("SUB") == "Unknown"


@pytest.mark.parametrize("abbrev,group", [
    ("G", "Goalkeeper"), ("CD-L", "Defender"), ("LWB", "Defender"),
    ("DM", "Midfielder"), ("AM", "Midfielder"), ("RW", "Forward"),
    ("F", "Forward"), ("", "Unknown"),
])
def test_position_abbreviations_map_to_roles(abbrev, group):
    assert position_group(abbrev) == group


# ── the name index ─────────────────────────────────────────────────────────────
def test_index_prefers_the_full_key_over_the_short_one():
    idx = NameIndex()
    idx.add("Adrián López Rodríguez", "forward", rank=100.0)
    idx.add("Adrián Rodríguez", "keeper", rank=1.0)
    assert idx.lookup_detail("Adrián Rodríguez") == ("keeper", "full")


def test_index_reports_a_short_match_as_such():
    """Callers hold the short key to a higher standard, so it has to be visible.

    Without the "short" label, Alavés' goalkeeper Adrián Rodríguez resolves to
    the retired forward Adrián López Rodríguez and keeps goal for them all
    season.
    """
    idx = NameIndex()
    idx.add("Adrián López Rodríguez", "forward", rank=100.0)
    value, kind = idx.lookup_detail("Adrián Rodríguez")
    assert (value, kind) == ("forward", "short")


def test_index_breaks_a_tie_on_rank():
    idx = NameIndex()
    idx.add("Pape Sarr", "old", rank=1.0)
    idx.add("Pape Matar Sarr", "recent", rank=9.0)
    assert idx.lookup("Pape Sarr") == "old"          # exact name wins outright
    assert idx.lookup("Pape M. Sarr") == "recent"    # short key, highest rank


def test_invisible_characters_do_not_split_a_name():
    """A soft hyphen (U+00AD) inside a surname renders as nothing, so the name
    looks identical on screen and never joins. The audit found one in the FBref
    tables; the ratings file carries `Andy O"Brien` with a straight double quote
    where an apostrophe should be."""
    assert full_key("Mat­ias Rocha") == full_key("Matias Rocha")
    assert full_key('Andy O"Brien') == full_key("Andy O'Brien")
    idx = NameIndex()
    idx.add('Bruno N"Gotty', "ngotty", rank=1.0)
    assert idx.lookup("Bruno N'Gotty") == "ngotty"


def test_every_apostrophe_is_the_same_apostrophe():
    """U+2019 is not decomposable, so the ascii strip DELETES it rather than
    folding it: "N’Golo Kanté" keyed on "ngolo kante" and "N'Golo Kanté" on
    "n golo kante", and the two spellings of one man never met. Three of this
    project's files mix both characters."""
    idx = NameIndex()
    idx.add("N'Golo Kanté", "kante", rank=318.0)
    assert idx.lookup("N’Golo Kanté") == "kante"
    assert full_key("N’Golo") == full_key("N'Golo")


def test_a_shortened_name_matches_its_long_form():
    """The compound-surname miss, and the biggest single data hole found.

    The ratings file writes "Kylian Mbappé Lottin" and "Lamine Yamal Nasraoui
    Ebana"; ESPN writes "Kylian Mbappé" and "Lamine Yamal". `short_key` takes
    the first name and the LAST surname, so it compared "kylian lottin" with
    "kylian mbappe" and missed — Mbappé (rated 93.8) and Yamal (90.8) both fell
    through to the projection. A leading run of the parts catches them.
    """
    idx = NameIndex()
    idx.add("Kylian Mbappé Lottin", "mbappe", rank=397.0)
    idx.add("Lamine Yamal Nasraoui Ebana", "yamal", rank=108.0)
    assert idx.lookup("Kylian Mbappé") == "mbappe"
    assert idx.lookup("Lamine Yamal") == "yamal"
    value, kind = idx.lookup_detail("Kylian Mbappé")
    assert kind == "contains", "el emparejamiento largo debe declararse como tal"


def test_a_containment_match_will_not_cross_two_people():
    """Same guard as the short key, one tier down: a name that keeps the first
    and last part and throws away the middle is a different person, not a
    shortening."""
    idx = NameIndex()
    idx.add("Luis Alberto Suárez Díaz", "suarez", rank=100.0)
    assert idx.lookup("Luis Díaz") is None
    # and an ambiguous containment is refused rather than guessed
    amb = NameIndex()
    amb.add("Rafael Silva Costa", "a", rank=50.0)
    amb.add("Rafael Silva Moreira", "b", rank=55.0)
    assert amb.lookup("Rafael Silva") is None


def test_short_key_refuses_to_drop_two_name_parts():
    """The maternal surname is a live grenade in the short key.

    "Luis Alberto Suárez Díaz" keys on Díaz, so Bayern's Luis Díaz resolved to
    Luis Suárez's career and his card read "Bayern Munich". One missing middle
    name is a spelling difference between providers; two is a different player.
    """
    idx = NameIndex()
    idx.add("Luis Alberto Suárez Díaz", "suarez", rank=100.0)
    assert idx.lookup("Luis Díaz") is None
    idx.add("Pape Matar Sarr", "sarr", rank=1.0)
    assert idx.lookup("Pape Sarr") == "sarr"


def test_index_never_compares_the_stored_objects():
    """Two entries tied on rank must not fall through to comparing values —
    a PlayerStrengthProfile is not orderable and the lookup raises."""
    class Unorderable:
        def __lt__(self, other):
            raise AssertionError("values must never be compared")

    idx = NameIndex()
    idx.add("Same Name", Unorderable(), rank=1.0)
    idx.add("Same Name", Unorderable(), rank=1.0)
    assert idx.lookup("Same Name") is not None


def test_fingerprint_tracks_content_not_file_time():
    a = pd.DataFrame({"team": ["x", "y"], "player": ["A", "B"]})
    b = pd.DataFrame({"team": ["y", "x"], "player": ["B", "A"]})
    c = pd.DataFrame({"team": ["x", "y"], "player": ["A", "C"]})
    assert squads_fingerprint(a) == squads_fingerprint(b)
    assert squads_fingerprint(a) != squads_fingerprint(c)
    assert squads_fingerprint(pd.DataFrame()) == "nosquads"


# ── the built table ────────────────────────────────────────────────────────────
def test_every_club_has_a_playable_squad():
    """A club short of a goalkeeper or of outfield players cannot field an XI,
    and the consumers silently fall back to whatever history they have."""
    df = _squads()
    by_team = df.groupby("team")["pos_group"].value_counts().unstack(fill_value=0)
    assert (by_team.get("Goalkeeper", 0) >= 1).all(), \
        f"clubs with no keeper: {list(by_team.index[by_team.get('Goalkeeper', 0) < 1])}"
    outfield = by_team.drop(columns=["Goalkeeper"], errors="ignore").sum(axis=1)
    assert (outfield >= 10).all(), f"clubs under 10 outfielders: {list(outfield[outfield < 10].index)}"


def test_positions_are_resolved():
    """Unknown positions cost a player his slot in SquadLab's eleven. A handful
    is tolerable; a wave of them means ESPN changed its vocabulary."""
    df = _squads()
    unknown = (df["pos_group"] == "Unknown").mean()
    assert unknown < 0.05, f"{unknown:.1%} of squad members have no position"


def test_squads_cover_the_current_season_of_the_foundation():
    """The table is only useful if it names the clubs actually being played."""
    df = _squads()
    found = ROOT / "data/processed/foundation_big5_multi_season.csv"
    if not found.exists():
        pytest.skip("no foundation")
    f = pd.read_csv(found, low_memory=False)
    season = sorted(f["season"].dropna().unique())[-1]
    cur = f[f["season"] == season]
    teams = set(cur["home_team"]) | set(cur["away_team"])
    missing = teams - set(df["team"])
    assert not missing, f"{season}: no squad for {sorted(missing)}"


def test_no_player_is_listed_at_two_clubs_in_the_same_row_set():
    df = _squads()
    dupes = df.duplicated(subset=["team", "player"])
    assert not dupes.any(), f"{int(dupes.sum())} duplicated (team, player) rows"
