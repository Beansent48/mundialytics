"""The SquadLab card catalogue: loading it, and dealing from it.

Three kinds of card, built by scripts/build_squadlab_cards.py:

    ACTUAL  a player as he is this season, filed under the club he actually
            plays for. The bulk of the pool and the only kind that covers
            everybody — a draft that cannot deal a card for a third of a squad
            is broken, so every current squad member has one.
    PRIME   one standout SEASON of a player, carrying that year's label and
            that year's rating. Scarce.
    ICONO   a retired generational name. No year, the highest ratings, and the
            rarest thing in the pool.

WHY THE DRAW IS TIERED RATHER THAN RATING-WEIGHTED. The obvious design — weight
each card by exp(-rating) — has no honest setting: gentle enough to ever deal a
90 and it deals them constantly, steep enough to make them rare and they become
unreachable. Packs work in tiers for a reason, and tiers are also the only
version a player can reason about ("élite is 3%") — which is the point, because
the tension in a draft comes from knowing what you are chasing.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CARDS_PATH = ROOT / "data/processed/squadlab_cards.csv"

KINDS = ("actual", "prime", "icono")

# Rating bands. Shared by the draw and by the card art, so a player learns to
# read the colour as a probability rather than as decoration.
TIERS = (
    ("bronce", 0.0, 66.0),
    ("plata", 66.0, 72.0),
    ("oro", 72.0, 79.0),
    ("oro_raro", 79.0, 85.0),
    ("elite", 85.0, 200.0),
)

# P(kind) for a single candidate. Eleven slots dealt five candidates each is 55
# draws, so 2% icons means most drafts show one and some show none — rare enough
# to be the thing you remember, common enough to be worth chasing.
KIND_P = {"actual": 0.880, "prime": 0.100, "icono": 0.020}

# P(tier | kind). The actual-card ladder is deliberately bottom-heavy: if gold
# is the median card then gold stops feeling like anything.
TIER_P = {
    "actual": {"bronce": 0.16, "plata": 0.34, "oro": 0.34, "oro_raro": 0.13, "elite": 0.03},
    "prime": {"oro_raro": 0.52, "elite": 0.48},
    "icono": {"elite": 1.0},
}

# What a pull is worth when scoring how lucky a draft was. Ratio-scaled rather
# than absolute so the number means the same thing at every squad level.
PULL_VALUE = {"actual": 1.0, "prime": 4.0, "icono": 12.0}


@dataclass(frozen=True)
class Card:
    card_id: str
    kind: str
    player: str
    display: str
    club: str
    league: str
    position: str
    role: str
    season_label: str
    overall: float
    attack: float
    defense: float
    creation: float
    gk: float
    # What the ENGINE plays with, on the scale it was calibrated on. The four
    # above are the card face, rebuilt onto the rating's scale so a 95-rated
    # striker does not show 63 in attack; the simulator must not read those or
    # the gap between a striker and a full-back flattens. See
    # scripts/build_squadlab_cards.py:raw_axes.
    raw_attack: float
    raw_defense: float
    raw_creation: float
    raw_gk: float
    n_att: float
    n_def: float
    matches: int
    source: str

    @property
    def measured_defense(self) -> bool:
        """Whether anyone ever measured this player defending.

        Zero for 62% of defenders — StatsBomb's free data released Barcelona's
        whole history and almost nobody else's back line. A card with no
        defensive sample shows exactly what its role implies, which is the
        honest reading, and this flag is how a UI can say so.
        """
        return self.n_def > 0

    @property
    def tier(self) -> str:
        return tier_of(self.overall)

    @property
    def is_special(self) -> bool:
        return self.kind != "actual"

    @property
    def subtitle(self) -> str:
        """What goes under the name on the card face."""
        if self.kind == "icono":
            return self.club
        if self.kind == "prime":
            return f"{self.club} · {self.season_label}" if self.club else self.season_label
        return self.club.title()

    def to_profile(self):
        """A PlayerStrengthProfile the season engine can play with.

        The simulator, the squad-to-lambda bridge and the event attribution all
        speak profiles; the catalogue exists to decide WHICH profile, not to
        replace the type.
        """
        from mundialytics.statistical_core.player_strength import PlayerStrengthProfile

        label = self.display
        if self.kind == "prime" and self.season_label:
            label = f"{self.display} {self.season_label}"
        elif self.kind == "icono":
            label = f"{self.display}{ICON_MARK}"
        return PlayerStrengthProfile(
            player=label,
            team=self.club,
            competition=self.league,
            position=self.position,
            matches=int(self.matches),
            offensive_strength=float(self.raw_attack),
            defensive_strength=float(self.raw_defense),
            creation_strength=float(self.raw_creation),
            gk_strength=float(self.raw_gk),
            overall=float(self.overall),
            role=self.role,
            role_overall=float(self.overall),
            current_team=self.club if self.kind == "actual" else "",
            is_icon=self.kind == "icono",
        )


# What a card appends to a player's name so its version is a distinct identity
# in the scorer race: " 15/16" for a prime, " ★" for an icon.
ICON_MARK = " ★"
_SEASON_MARK = re.compile(r"\s\d{2}/\d{2}$")


def split_card_mark(name: object) -> tuple[str, str]:
    """("Suárez 15/16") -> ("Suárez", " 15/16"). The mark must survive name
    shortening: `display_name` keeps the last word, so a prime came back called
    "15/16" and an icon called "★" — the same failure the clone mark ✦ had."""
    text = str(name)
    if text.endswith(ICON_MARK):
        return text[: -len(ICON_MARK)], ICON_MARK
    m = _SEASON_MARK.search(text)
    if m:
        return text[: m.start()], m.group(0)
    return text, ""


def tier_of(overall: float) -> str:
    for name, lo, hi in TIERS:
        if lo <= overall < hi:
            return name
    return "bronce"


@functools.lru_cache(maxsize=4)
def load_cards(path: str | None = None) -> pd.DataFrame:
    p = Path(path) if path else DEFAULT_CARDS_PATH
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in ("club", "league", "role", "season_label", "source", "display"):
        if c in df.columns:
            df[c] = df[c].fillna("")
    # a catalogue built before the face/engine split has only one set of axes
    for face, raw in (("attack", "raw_attack"), ("defense", "raw_defense"),
                      ("creation", "raw_creation"), ("gk", "raw_gk")):
        if raw not in df.columns:
            df[raw] = df[face]
    for c in ("n_att", "n_def"):
        if c not in df.columns:
            df[c] = 0.0
    df["tier"] = df["overall"].map(tier_of)
    return df


@functools.lru_cache(maxsize=4)
def card_player_names(path: str | None = None) -> dict[str, str]:
    """card_id -> the man's name, for callers holding ids and not Cards.

    Building 3,000 Card objects to read one field off a handful of them is the
    kind of work that turns a 6ms deal into a slow one.
    """
    df = load_cards(path)
    if df.empty:
        return {}
    return dict(zip(df["card_id"].astype(str), df["player"].astype(str)))


def cards_from_frame(df: pd.DataFrame) -> list[Card]:
    cols = Card.__dataclass_fields__.keys()
    return [Card(**{c: r[c] for c in cols}) for _, r in df.iterrows()]


class DraftPool:
    """Deals candidates for one draft, without repeating a player.

    Uniqueness is by PERSON, not by card: drafting Messi must remove his icon,
    his prime and his current card at once, or a squad can field three of him.
    """

    def __init__(self, cards: pd.DataFrame, rng: np.random.Generator | None = None):
        self.df = cards.reset_index(drop=True)
        self.rng = rng or np.random.default_rng()
        self._taken: set[str] = set()
        self._by = {
            (k, t, pos): grp.index.to_numpy()
            for (k, t, pos), grp in self.df.groupby(["kind", "tier", "position"])
        }

    # ── identity ──────────────────────────────────────────────────────────────
    @staticmethod
    def person(player: str) -> str:
        from mundialytics.identity.current_squads import short_key
        return short_key(player)

    def take(self, card: Card) -> None:
        self.take_player(card.player)

    def take_player(self, player: str) -> None:
        """Remove a man by name, for callers holding a pick and not a Card.

        The draft is stateless over HTTP: the client posts back who it has
        already drafted, and rebuilding a Card for each of them just to reach
        his short_key would be work for nothing.
        """
        self._taken.add(self.person(player))

    def is_taken(self, card: Card) -> bool:
        return self.person(card.player) in self._taken

    # ── dealing ───────────────────────────────────────────────────────────────
    def _pick(self, kind: str, tier: str, position: str) -> Card | None:
        idx = self._by.get((kind, tier, position))
        if idx is None or not len(idx):
            return None
        order = self.rng.permutation(idx)
        for i in order[:60]:
            card = Card(**{c: self.df.at[i, c] for c in Card.__dataclass_fields__})
            if not self.is_taken(card):
                return card
        return None

    def deal(self, position: str, n: int = 5) -> list[Card]:
        """`n` candidates for one slot, each rolled kind-then-tier.

        A roll that lands on an empty bucket (no icon goalkeepers left, say)
        falls back down the ladder rather than returning fewer candidates — an
        empty slot would just look like a bug to whoever is drafting.
        """
        out: list[Card] = []
        guard = 0
        while len(out) < n and guard < n * 40:
            guard += 1
            kind = self._roll_kind()
            tier = self._roll_tier(kind)
            card = (self._pick(kind, tier, position)
                    or self._fallback(kind, tier, position))
            if card is None or any(c.card_id == card.card_id for c in out) \
                    or any(self.person(c.player) == self.person(card.player) for c in out):
                continue
            out.append(card)
        return out

    def _roll_kind(self) -> str:
        r = float(self.rng.random())
        acc = 0.0
        for kind in KINDS:
            acc += KIND_P[kind]
            if r < acc:
                return kind
        return "actual"

    def _roll_tier(self, kind: str) -> str:
        table = TIER_P[kind]
        r = float(self.rng.random())
        acc = 0.0
        for tier, p in table.items():
            acc += p
            if r < acc:
                return tier
        return list(table)[-1]

    def _fallback(self, kind: str, tier: str, position: str) -> Card | None:
        """Walk down the tiers, then down to an ordinary card."""
        names = [t[0] for t in TIERS]
        start = names.index(tier) if tier in names else 0
        for t in names[start::-1] + names[start + 1:]:
            card = self._pick(kind, t, position)
            if card is not None:
                return card
        if kind != "actual":
            return self._fallback("actual", tier, position)
        return None


# Formation -> how many of each role the eleven needs. Kept here rather than in
# the page because the season engine, the Elo calibration and the draft all have
# to agree on what "an eleven" is.
FORMATION_SLOTS = {
    "4-3-3": {"Goalkeeper": 1, "Defender": 4, "Midfielder": 3, "Forward": 3},
    "4-4-2": {"Goalkeeper": 1, "Defender": 4, "Midfielder": 4, "Forward": 2},
    "3-4-3": {"Goalkeeper": 1, "Defender": 3, "Midfielder": 4, "Forward": 3},
    "3-5-2": {"Goalkeeper": 1, "Defender": 3, "Midfielder": 5, "Forward": 2},
    "5-2-3": {"Goalkeeper": 1, "Defender": 5, "Midfielder": 2, "Forward": 3},
    "5-3-2": {"Goalkeeper": 1, "Defender": 5, "Midfielder": 3, "Forward": 2},
}
DEFAULT_FORMATION = "4-3-3"


def best_eleven(cards: pd.DataFrame, club: str, formation: str = DEFAULT_FORMATION) -> list[Card]:
    """A real club's strongest eleven out of its own current cards."""
    slots = FORMATION_SLOTS.get(formation, FORMATION_SLOTS[DEFAULT_FORMATION])
    pool = cards[(cards["kind"] == "actual") & (cards["club"].str.lower() == club.lower())]
    out: list[Card] = []
    for pos, n in slots.items():
        sub = pool[pool["position"] == pos].nlargest(n, "overall")
        out += cards_from_frame(sub)
    return out


def squad_rating(cards: list[Card]) -> float:
    """One number for an eleven. Straight mean — every slot plays."""
    return float(np.mean([c.overall for c in cards])) if cards else 0.0


def squad_chemistry(cards: list[Card]) -> dict:
    """Links between the eleven, the way a draft rewards a coherent squad.

    Club and league links are the FUT convention and they carry a real idea:
    players who share a side understand each other. Icons are exempt because
    they belong to no current league, and penalising a player for drafting one
    would work against the thing the pool is built to make exciting.
    """
    if not cards:
        return {"links": 0, "max_links": 0, "pct": 0.0, "top_club": "", "top_league": ""}
    reg = [c for c in cards if c.kind != "icono"]
    clubs = pd.Series([c.club.lower() for c in reg if c.club])
    leagues = pd.Series([c.league.lower() for c in reg if c.league])
    club_links = int(sum(n * (n - 1) // 2 for n in clubs.value_counts())) if len(clubs) else 0
    league_links = int(sum(n * (n - 1) // 2 for n in leagues.value_counts())) if len(leagues) else 0
    n = len(cards)
    max_links = n * (n - 1) // 2
    score = min(1.0, (club_links * 2.0 + league_links * 0.5) / max(max_links, 1))
    return {
        "links": club_links,
        "league_links": league_links,
        "max_links": max_links,
        "pct": round(100 * score, 1),
        "top_club": clubs.value_counts().index[0].title() if len(clubs) else "",
        "top_league": leagues.value_counts().index[0] if len(leagues) else "",
    }


def pack_luck(cards: list[Card]) -> dict:
    """How lucky this draft was, against what the draw probabilities imply.

    Expressed as a ratio so it reads the same for everyone: 1.0 is exactly an
    average draft, 2.0 is twice the special-card value an average draft carries.
    """
    got = sum(PULL_VALUE.get(c.kind, 1.0) for c in cards)
    exp_per_card = sum(KIND_P[k] * PULL_VALUE[k] for k in KINDS)
    expected = exp_per_card * max(len(cards), 1)
    return {
        "value": round(got, 1),
        "expected": round(expected, 1),
        "ratio": round(got / expected, 2) if expected else 0.0,
        "primes": sum(1 for c in cards if c.kind == "prime"),
        "iconos": sum(1 for c in cards if c.kind == "icono"),
    }
