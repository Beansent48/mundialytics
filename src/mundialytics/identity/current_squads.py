"""Who plays for which club RIGHT NOW.

Every player-facing number in the product — the scorer shortlist on a fixture,
the Golden Boot race, the league SquadLab plays out — was answering "who is in
this squad?" from historical appearance data, and history is not a squad list.
The two consumers failed in the same direction for different reasons:

  * the props model read the squad off "players seen in this club's last ten
    Understat games". For a club that has not been in a top-five league for
    years, that window silently reaches back to whenever it last was: Hull's
    2026/27 squad came back as the 2016/17 one, Harry Maguire and all.
  * SquadLab read it off career-aggregated profiles, whose `team` is the club a
    player is remembered for. It fielded Xavi, Puyol and Messi for Barcelona.

Understat stopped serving its embedded JSON some time after May 2026 (the league
pages now render client-side), so the answer had to come from somewhere else.
ESPN already carries the current season's full match rosters — the same feed the
player markets are settled against — including a per-match position. That makes
it the one source that knows a squad as of this week.

This module builds that squad table and matches its players back to the two
historical datasets by name, which is the only key the sources share.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CURRENT_SQUADS_PATH = ROOT / "data/processed/current_squads.csv"

# ESPN position abbreviations, most specific rule first. Wing backs carry both a
# W and a B and belong with the defenders, so the B test has to come before the
# W one; "AM"/"DM" are midfielders despite the attacking/defensive prefix.
#
# "SUB" is not a position. ESPN's MATCH rosters label every substitute that way,
# which is why the squad table takes positions from the team-roster endpoint
# instead — read as a position it ends in a B, and two thirds of the league
# turns into defenders.
_NON_POSITIONS = {"SUB", "SUBSTITUTE", "BENCH", "NA", "N/A", "UNKNOWN"}
_POS_RULES: tuple[tuple[str, str], ...] = (
    ("G", "Goalkeeper"), ("GK", "Goalkeeper"),
    ("SW", "Defender"), ("CB", "Defender"), ("CD", "Defender"), ("D", "Defender"),
    ("LB", "Defender"), ("RB", "Defender"), ("LWB", "Defender"), ("RWB", "Defender"),
    ("WB", "Defender"),
    ("DM", "Midfielder"), ("CM", "Midfielder"), ("AM", "Midfielder"),
    ("LM", "Midfielder"), ("RM", "Midfielder"), ("M", "Midfielder"),
    ("ST", "Forward"), ("CF", "Forward"), ("SS", "Forward"),
    ("LW", "Forward"), ("RW", "Forward"), ("W", "Forward"), ("F", "Forward"),
)


def position_group(abbrev: object) -> str:
    """ESPN's per-match position abbreviation -> SquadLab's four-way role.

    ESPN spells the slot, not the role ("CD-L", "RWB", "AM"), and the side
    suffix carries no positional information, so it is dropped first.
    """
    s = str(abbrev or "").strip().upper()
    if s in _NON_POSITIONS:
        return "Unknown"
    s = re.split(r"[-/]", s)[0].strip()
    if not s or s in _NON_POSITIONS:
        return "Unknown"
    for token, group in _POS_RULES:
        if s == token:
            return group
    # unseen spelling: fall back to the letters it does contain, in the same
    # precedence order the explicit table uses
    if s.startswith("G"):
        return "Goalkeeper"
    if s.endswith("B") or (s.startswith("D") and not s.startswith("DM")):
        return "Defender"
    if "M" in s:
        return "Midfielder"
    if s[0] in "FWS":
        return "Forward"
    return "Unknown"


def _fold(name: object, keep_hyphen: bool = False) -> str:
    """Accent-stripped, punctuation-flattened, lowercase name."""
    s = unicodedata.normalize("NFKD", str(name))
    # Combining marks AND invisible formatting characters. The audit found a
    # SOFT HYPHEN (U+00AD) sitting inside a player's surname: it renders as
    # nothing, so the name looks identical on screen and never joins.
    s = "".join(c for c in s
                if not unicodedata.combining(c) and unicodedata.category(c) != "Cf")
    # Every apostrophe variant first, and BEFORE the ascii strip: the
    # typographic one (U+2019) is not decomposable, so `encode("ascii",
    # "ignore")` silently DELETES it — "N’Golo" folds to "ngolo" while
    # "N'Golo" folds to "n golo", and the two spellings of the same man never
    # meet. Three files in this project mix both.
    # Every apostrophe the sources use, the straight double quote included:
    # the ratings file carries `Andy O"Brien` and `Bruno N"Gotty`, where the
    # apostrophe was mangled into a quote somewhere upstream.
    for _q in "’‘ʼ´`\"":
        s = s.replace(_q, "'")
    s = s.encode("ascii", "ignore").decode().lower()
    s = s.replace("'", " ").replace(".", " ")
    if not keep_hyphen:
        s = s.replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def full_key(name: object) -> str:
    """Whole-name key. Exact enough that a hit is almost certainly the player."""
    return _fold(name)


def short_key(name: object) -> str:
    """First name + the first segment of the surname.

    The providers disagree on the tail of a name — Understat's "Kylian
    Mbappe-Lottin" against ESPN's "Kylian Mbappé" — so the full key misses
    players the eye matches instantly. The hyphen survives folding precisely so
    it can be cut here: flattening it to a space first would make the surname
    "Lottin" and leave the two spellings as far apart as before. Two given names
    can collapse onto one key, which is why this is the fallback and why
    candidates are ranked before one is taken.
    """
    parts = _fold(name, keep_hyphen=True).split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0].split("-")[0]
    return f"{parts[0].split('-')[0]} {parts[-1].split('-')[0]}"


def load_current_squads(path: str | Path | None = None) -> pd.DataFrame:
    """The current-squad table, or an empty frame if it has not been built.

    Every consumer treats absence as "carry on with the historical roster", so
    a missing file degrades the squads rather than breaking the page.
    """
    p = Path(path) if path else DEFAULT_CURRENT_SQUADS_PATH
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    if df.empty or "team" not in df.columns or "player" not in df.columns:
        return pd.DataFrame()
    for col, fn in (("full_key", full_key), ("short_key", short_key)):
        if col not in df.columns:
            df[col] = df["player"].map(fn)
    # consumers index pos_group unconditionally; derive it rather than making
    # every caller guard a column the builder has always written
    if "pos_group" not in df.columns:
        df["pos_group"] = (df["position"].map(position_group)
                           if "position" in df.columns else "Unknown")
    df["pos_group"] = df["pos_group"].fillna("Unknown")
    return df


def squads_fingerprint(squads: "pd.DataFrame | str | Path | None" = None) -> str:
    """Short content hash of a squad table, for cache keys.

    Content rather than mtime: rebuilding the table on a quiet week produces the
    same squads, and a fitted props model costs ~45s to rebuild for nothing.
    """
    df = squads if isinstance(squads, pd.DataFrame) else load_current_squads(squads)
    if df is None or df.empty:
        return "nosquads"
    pairs = sorted(f"{t}|{p}" for t, p in zip(df["team"], df["player"]))
    return f"{len(pairs)}-{hashlib.sha1('~'.join(pairs).encode('utf-8')).hexdigest()[:10]}"


# How many name parts the short key is allowed to drop. One is a middle name
# the other provider left out — "Pape Sarr" for "Pape Matar Sarr". Two is a
# different player: "Luis Alberto Suárez Díaz" keys on his maternal surname and
# lands squarely on Bayern's Luis Díaz, whose own history is nowhere near it.
MAX_DROPPED_NAME_PARTS = 1
# A containment match needs at least this many shared name parts. Two is the
# floor that still means something: one shared token is a given name, and
# "Luis" is not a person.
MIN_SHARED_TOKENS = 2
# When several different people contain the query, the best is taken only if it
# is this many times better evidenced than the runner-up.
CONTAIN_RANK_MARGIN = 2.0
# How many parts a name may drop once the CALLER can confirm the hit. "Bruno
# Fernandes" is "Bruno Miguel Borges Fernandes" and "Vinícius Júnior" is
# "Vinícius José Paixão de Oliveira Júnior" — both keep the first and last part
# and throw away the middle, which is exactly the shape that makes "Luis Díaz"
# collide with "Luis Alberto Suárez Díaz". Name alone cannot separate them, so
# this tier exists only for callers that hold other evidence (the position he
# plays today) and pass it as `confirm`.
MAX_LOOSE_NAME_PARTS = 3


def _n_parts(name: object) -> int:
    return len(_fold(name, keep_hyphen=True).split())


def name_parts(name: object) -> tuple:
    """The name as an ORDERED tuple of parts.

    `short_key` takes the first given name and the LAST surname, which is the
    wrong end for a Spanish, Portuguese or French compound: the ratings file
    writes "Kylian Mbappé Lottin" and "Lamine Yamal Nasraoui Ebana" while ESPN
    writes "Kylian Mbappé" and "Lamine Yamal", so the short key becomes
    "kylian lottin" against "kylian mbappe" and misses both.

    Order is kept because it is the only thing separating a shortened name from
    a different person. "Lamine Yamal" is the LEADING PART of "Lamine Yamal
    Nasraoui Ebana" — that is how these names shorten, by dropping from the
    right. "Luis Díaz" is a subset of "Luis Alberto Suárez Díaz" too, but not a
    leading part: it keeps the first token and the last and throws away what is
    between, which is a different person's name, not a shortening.
    """
    return tuple(_fold(name).split())


class NameIndex:
    """Name -> historical record, for matching a current squad to past data.

    Built once over the historical side, then queried per squad member. When a
    short key is ambiguous the highest-ranked record wins: the key exists to
    reunite one player with his own history, and rank is how the caller says
    which candidate is likelier to be that player.
    """

    def __init__(self) -> None:
        self._full: dict[str, list[tuple[float, int, object]]] = {}
        self._short: dict[str, list[tuple[float, int, object]]] = {}
        self._by_token: dict[str, list[tuple[frozenset, float, object]]] = {}

    def add(self, name: object, value: object, rank: float = 0.0,
            birth: float | None = None) -> None:
        """`rank` breaks ties — pass recency (or sample size) so the live
        player beats a namesake who stopped playing years ago. `birth` is the
        player's birth year: when the caller knows the birth year of the man it
        is looking for, it separates two namesakes that rank alone cannot."""
        fk, sk, n = full_key(name), short_key(name), _n_parts(name)
        if fk:
            self._full.setdefault(fk, []).append((rank, n, value, birth))
        if sk:
            self._short.setdefault(sk, []).append((rank, n, value, birth))
        parts = name_parts(name)
        if len(parts) >= MIN_SHARED_TOKENS:
            for t in set(parts):
                self._by_token.setdefault(t, []).append((parts, rank, value))

    def lookup(self, name: object, confirm=None, birth: float | None = None):
        """Best historical record for `name`, or None. See `lookup_detail`."""
        return self.lookup_detail(name, confirm, birth)[0]

    @staticmethod
    def _prefer_birth(hits: list, birth: float | None) -> list:
        """Keep only the candidates whose birth year matches (±1), if any do.

        A birthday falls either side of the season boundary, so age->year can be
        off by one; ±1 absorbs that while still separating two players born years
        apart. Falls back to all hits when none carries a birth year or none
        matches, so this can only help, never lose a real match."""
        if birth is None:
            return hits
        near = [h for h in hits if len(h) > 3 and h[3] is not None
                and abs(float(h[3]) - float(birth)) <= 1]
        return near or hits

    def lookup_detail(self, name: object, confirm=None,
                      birth: float | None = None) -> tuple[object, str]:
        """As `lookup`, plus which key matched: "full", "short", or "".

        Callers use it to hold the short key to a higher standard. It is the
        fallback that reunites "Kylian Mbappé" with "Kylian Mbappe-Lottin", and
        by the same rule it will happily read Alavés' goalkeeper Adrián
        Rodríguez as the retired forward Adrián López Rodríguez — two Spanish
        surnames, one dropped in the middle. A caller that knows the position
        can reject that; the index itself cannot tell them apart.
        """
        hits = self._full.get(full_key(name))
        if hits:
            # keyed, never a bare max: the third element is the caller's own
            # object and comparing two of them raises rather than tie-breaks.
            # Birth year separates two same-named players before rank decides.
            cand = self._prefer_birth(hits, birth)
            return max(cand, key=lambda h: h[0])[2], "full"
        hits = self._short.get(short_key(name))
        if hits:
            want = _n_parts(name)
            near = [h for h in hits if abs(h[1] - want) <= MAX_DROPPED_NAME_PARTS]
            if near:
                # closest in length first, then birth year, then the caller's rank
                near = self._prefer_birth(near, birth)
                return max(near, key=lambda h: (-abs(h[1] - want), h[0]))[2], "short"
        hit = self._contained(name)
        if hit is not None:
            return hit, "contains"
        if confirm is not None:
            hit = self._contained(name, confirm=confirm, max_drop=MAX_LOOSE_NAME_PARTS)
            if hit is not None:
                return hit, "confirmed"
        return None, ""

    def _contained(self, name: object, confirm=None, max_drop: int | None = None):
        """One spelling of a name that is a SHORTENING of another.

        Deliberately the last resort and deliberately strict. The shorter name
        must be a leading run of the longer one — "Kylian Mbappé" inside
        "Kylian Mbappé Lottin" — or differ by at most MAX_DROPPED_NAME_PARTS.
        Plain set containment is not enough and the test suite says why: "Luis
        Díaz" sits inside "Luis Alberto Suárez Díaz" as a set, and taking that
        match is how Bayern's Luis Díaz ended up with Luis Suárez's career.

        When two different people both admit the query, the better-evidenced one
        wins only by CONTAIN_RANK_MARGIN.
        """
        q = name_parts(name)
        if len(q) < MIN_SHARED_TOKENS:
            return None
        posts = min((self._by_token.get(t, []) for t in set(q)), key=len, default=[])
        best: dict[tuple, tuple[float, object]] = {}
        for parts, rank, value in posts:
            short, long = (q, parts) if len(q) <= len(parts) else (parts, q)
            if len(short) < MIN_SHARED_TOKENS:
                continue
            prefix = long[:len(short)] == short
            limit = MAX_DROPPED_NAME_PARTS if max_drop is None else max_drop
            # a name that keeps the first part and the last and throws away the
            # middle — "Bruno Fernandes" inside "Bruno Miguel Borges Fernandes"
            # — is how Portuguese names shorten and how "Luis Díaz" collides
            # with "Luis Alberto Suárez Díaz". The two are indistinguishable by
            # name alone, so the loose form is only reachable with `confirm`.
            ends = short[0] == long[0] and short[-1] == long[-1]
            if not (prefix or ((set(short) <= set(long) or ends)
                               and len(long) - len(short) <= limit)):
                continue
            if confirm is not None and not confirm(value):
                continue
            cur = best.get(parts)
            if cur is None or rank > cur[0]:
                best[parts] = (rank, value)
        if not best:
            return None
        ranked = sorted(best.values(), key=lambda h: -h[0])
        if len(ranked) == 1:
            return ranked[0][1]
        if ranked[0][0] >= CONTAIN_RANK_MARGIN * max(ranked[1][0], 1.0):
            return ranked[0][1]
        return None

    def __len__(self) -> int:
        return len(self._full)
