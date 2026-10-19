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
import os
import sys
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


# The name shapes that put two EQUALLY RARE tokens in front of one query, with
# the historical side trimmed to the records that create the tie. Every one of
# them was found in the card build of 2026-10-19: with the tie broken by `set`
# iteration order, the same inputs dealt different cards from run to run.
#
# `(name, rank)` on the left, and the one answer each query may ever give on the
# right. Two tokens tying means neither is better evidenced, so both lists have
# to be read — the first two shapes lose the player outright if the losing
# token's list is the one taken, and the last one hands his history to a
# different man.
_TIED_HISTORY = [
    # `dias` and `freitas` carry two records each, `tiago` three
    ("Rúben Dias", 300.0), ("Tiago Dias", 40.0),
    ("Afonso Freitas", 20.0), ("Marlon Freitas", 25.0),
    ("Tiago Santos", 60.0), ("Tiago Gouveia", 30.0),
    # `ze` and `figueiredo` one each, `pedro` three
    ("Zé Pedro", 90.0), ("Miguel Figueiredo", 15.0),
    ("Pedro Neto", 200.0), ("Pedro Porro", 180.0),
    # `oyarzabal` and `ugarte` one each, `mikel` three
    ("Mikel Oyarzabal", 250.0), ("Manuel Ugarte", 150.0),
    ("Mikel Merino", 210.0), ("Mikel Jauregizar", 70.0),
    # `zubimendi` and `ibañez` one each, `martin` three
    ("Martín Zubimendi", 240.0), ("Pablo Ibáñez", 35.0),
    ("Martín Satriano", 45.0), ("Martín Aguirregabiria", 55.0),
    # CONSTRUCTED rather than measured, for the case the measured ones do not
    # reach: two DIFFERENT men admit the query and the tied tokens hold one
    # each, so a single list means no arbitration at all. `pinto` admits only
    # the four-part name, `mota` admits both and lets the rank margin decide —
    # and both tokens carry two records, so which of the two men got the
    # history came down to the hash seed.
    ("Carlos Mota", 400.0), ("Carlos Mota Pinto Silva", 10.0),
    ("Dinis Pinto", 60.0), ("Carlos Vicente", 120.0),
]
_TIED_ANSWERS = {
    "Tiago Dias Freitas": "Tiago Dias",
    "Zé Pedro Figueiredo": "Zé Pedro",
    "Mikel Oyarzabal Ugarte": "Mikel Oyarzabal",
    "Martín Zubimendi Ibáñez": "Martín Zubimendi",
    "Carlos Mota Pinto": "Carlos Mota",
}


def _tied_index() -> NameIndex:
    idx = NameIndex()
    for name, rank in _TIED_HISTORY:
        idx.add(name, name, rank=rank)
    return idx


def test_a_tie_between_equally_rare_tokens_goes_to_the_evidence():
    """Two tokens of one name bringing the same number of candidates.

    The containment tier reads the rarest token's candidates, and when two
    tokens are equally rare there is nothing to choose between them — so both
    are read. Measured over the card build, the losing token's list held no
    admissible record at all in all nine queries where the two disagreed:
    "Tiago Dias Freitas" resolved to Tiago Dias through `dias` and to nobody
    through `freitas`, so taking one list meant deciding by coin flip between
    the right answer and no answer. "Carlos Mota Pinto" is the shape that is worse
    still — a coin flip between two different men, which the rank margin
    exists to settle and cannot when it sees only one of them.
    """
    idx = _tied_index()
    for query, want in _TIED_ANSWERS.items():
        assert idx.lookup(query) == want, f"{query} deberia resolver a {want}"
    # and the tiers above the containment one are untouched by any of this
    assert idx.lookup("Rúben Dias") == "Rúben Dias"
    assert idx.lookup("Luis Díaz") is None


# Run in a child process on purpose: the string hash seed is fixed when the
# interpreter starts, so the only way to ask "does this depend on the hash
# seed?" is to start interpreters that disagree about it.
#
# The names carry accents and the child's stdout is cp1252 on this machine, so
# both ends are pinned to utf-8 rather than left to the console codepage — one
# name outside cp1252 would otherwise crash the probe instead of answering the
# question it was asked.
_SEED_PROBE = """
import sys
sys.path.insert(0, {src!r})
from mundialytics.identity.current_squads import NameIndex
idx = NameIndex()
for name, rank in {hist!r}:
    idx.add(name, name, rank=rank)
for q in {queries!r}:
    print(q, "->", idx.lookup(q))
"""


def test_the_index_answers_the_same_under_any_hash_seed():
    """The guard on the card catalogue being rebuildable.

    Two rebuilds off identical inputs used to deal different cards — 228
    overalls moved by up to 2.1 points and one card lost its measured axes —
    because the containment tier picked its candidate list with `min` over a
    `set`, and a tie in length came out of the set's iteration order, which is
    the hash seed. The symptom was in the catalogue; the test belongs here,
    where the tie is, because rebuilding the catalogue to find out costs half a
    minute and names that tie can be written down in twenty lines.
    """
    import subprocess

    src = str(ROOT / "src")
    code = _SEED_PROBE.format(src=src, hist=_TIED_HISTORY,
                              queries=list(_TIED_ANSWERS))
    seen: dict[str, list[str]] = {}
    for seed in ("0", "1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, "-c", code], env=env,
                           capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, r.stderr
        seen.setdefault(r.stdout, []).append(seed)
    assert len(seen) == 1, (
        "el emparejamiento de nombres depende de la semilla de hash:\n"
        + "\n".join(f"  semillas {s}:\n{o}" for o, s in seen.items()))


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


# ── season tallies ────────────────────────────────────────────────────────────

def test_season_tallies_ignore_rows_from_older_seasons():
    """What the 2026-10-06 backfill broke.

    fetch_espn_match_events.py appends to these files over whatever --from/--to
    window it is given, so older rows can end up in them. match_tallies used to
    sum the whole file: a historical UEFA backfill left two extra seasons behind,
    the daily job rebuilt the squads over them, and uefa_apps went from 1 to 35
    for European players -- the number SquadLab divides its per-appearance rates
    by. Nothing failed; the counts were just wrong.
    """
    import sys
    sys.path[:0] = [str(ROOT / "scripts")]
    import build_current_squads as bcs

    rows = []
    for season, n_events in (("2024-2025", 12), ("2025-2026", 11), ("2026-2027", 1)):
        for ev in range(n_events):
            rows.append({"event_id": f"{season}-{ev}", "season": season,
                         "competition": "Champions League", "date": f"{season[:4]}-09-17",
                         "team": "paris sg", "player": "Vitinha", "position": "Midfielder",
                         "starter": True, "subbed_in": False, "goals": 1, "shots": 2,
                         "sot": 1, "assists": 0, "yellow_cards": 0, "red_cards": 0,
                         "own_goals": 0, "appearances": 1, "sub_ins": 0})
    src = ROOT / "tests" / "fixtures" / "_tmp_match_rows.csv"
    src.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(src, index=False)
    try:
        out = bcs.match_tallies(src)
    finally:
        src.unlink(missing_ok=True)

    assert len(out) == 1
    got = out.iloc[0]
    assert got["season"] == "2026-2027"
    assert got["apps"] == 1, "older seasons must not be counted as this season's appearances"
    assert got["goals"] == 1
    assert got["squad_matches"] == 1
