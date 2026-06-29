from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import pandas as pd

PLAYER_TOKEN_STOPWORDS = {
    "de", "del", "da", "das", "dos", "di", "du", "la", "le", "el", "van", "von",
    "jr", "junior", "sr", "senior", "ii", "iii", "iv", "bin", "ibn",
}

TEAM_ALIASES = {
    "arg": "argentina",
    "argentina": "argentina",
    "aus": "australia",
    "australia": "australia",
    "aut": "austria",
    "austria": "austria",
    "bel": "belgium",
    "belgium": "belgium",
    "bih": "bosnia and herzegovina",
    "bosnia": "bosnia and herzegovina",
    "bosnia herzegovina": "bosnia and herzegovina",
    "bosnia and herzegovina": "bosnia and herzegovina",
    "bra": "brazil",
    "brazil": "brazil",
    "can": "canada",
    "canada": "canada",
    "cape verde islands": "cape verde",
    "cabo verde": "cape verde",
    "cape verde": "cape verde",
    "cpv": "cape verde",
    "civ": "cote d ivoire",
    "ivory coast": "cote d ivoire",
    "cote d ivoire": "cote d ivoire",
    "côte d ivoire": "cote d ivoire",
    "cote divoire": "cote d ivoire",
    "cod": "dr congo",
    "congo dr": "dr congo",
    "democratic republic of congo": "dr congo",
    "dr congo": "dr congo",
    "cro": "croatia",
    "croatia": "croatia",
    "cuw": "curacao",
    "curaçao": "curacao",
    "curacao": "curacao",
    "cze": "czechia",
    "czech republic": "czechia",
    "czechia": "czechia",
    "dza": "algeria",
    "algeria": "algeria",
    "ecu": "ecuador",
    "ecuador": "ecuador",
    "egy": "egypt",
    "egypt": "egypt",
    "eng": "england",
    "england nt": "england",
    "england": "england",
    "esp": "spain",
    "espana": "spain",
    "españa": "spain",
    "spain": "spain",
    "fra": "france",
    "france": "france",
    "ger": "germany",
    "deu": "germany",
    "germany": "germany",
    "gha": "ghana",
    "ghana": "ghana",
    "hti": "haiti",
    "haiti": "haiti",
    "irn": "iran",
    "iri": "iran",
    "iran": "iran",
    "irq": "iraq",
    "iraq": "iraq",
    "jor": "jordan",
    "jordan": "jordan",
    "jpn": "japan",
    "japan": "japan",
    "ksa": "saudi arabia",
    "saudi arabia nt": "saudi arabia",
    "saudi arabia": "saudi arabia",
    "mar": "morocco",
    "morocco": "morocco",
    "mex": "mexico",
    "méxico": "mexico",
    "mexico": "mexico",
    "ned": "netherlands",
    "netherlands": "netherlands",
    "holland": "netherlands",
    "nor": "norway",
    "norway": "norway",
    "nzl": "new zealand",
    "new zealand": "new zealand",
    "pan": "panama",
    "panamá": "panama",
    "panama": "panama",
    "par": "paraguay",
    "paraguay": "paraguay",
    "por": "portugal",
    "portugal": "portugal",
    "qat": "qatar",
    "qatar": "qatar",
    "rsa": "south africa",
    "south africa": "south africa",
    "sco": "scotland",
    "scotland": "scotland",
    "sen": "senegal",
    "senegal": "senegal",
    "sui": "switzerland",
    "switzerland": "switzerland",
    "swe": "sweden",
    "sweden": "sweden",
    "tun": "tunisia",
    "tunisia": "tunisia",
    "tur": "turkey",
    "türkiye": "turkey",
    "turkiye": "turkey",
    "turkey": "turkey",
    "uru": "uruguay",
    "uruguay nt": "uruguay",
    "uruguay": "uruguay",
    "usa": "united states",
    "u s a": "united states",
    "united states of america": "united states",
    "united states": "united states",
    "uzb": "uzbekistan",
    "uzbekistan": "uzbekistan",
    "real madrid cf": "real madrid",
    "r madrid": "real madrid",
    "fc barcelona": "barcelona",
    "barca": "barcelona",
    "barça": "barcelona",
    "atletico": "atletico madrid",
    "atl madrid": "atletico madrid",
    "atlético madrid": "atletico madrid",
    "atletico madrid": "atletico madrid",
}


def strip_accents(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def normalize_text(value: object, *, keep_plus: bool = False) -> str:
    """Normalize provider/manual names without losing player identity tokens.

    This is intentionally conservative: accents, hyphens and apostrophes are
    normalized, but words are not reordered and no fuzzy match is performed here.
    """
    text = strip_accents(value).lower().strip()
    text = text.replace("&", " and ")
    # Hyphens/apostrophes in names are separators, not part of the identity key.
    text = re.sub(r"[-–—'’`´]", " ", text)
    allowed = r"[^a-z0-9+]+" if keep_plus else r"[^a-z0-9]+"
    text = re.sub(allowed, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_team_name(name: object) -> str:
    base = normalize_text(name)
    return TEAM_ALIASES.get(base, base)


def canonical_player_name(name: object) -> str:
    return normalize_text(name)


def identity_key(name: object) -> str:
    return normalize_text(name).replace(" ", "_")


def player_tokens(name: object, *, drop_stopwords: bool = True) -> tuple[str, ...]:
    toks = [t for t in normalize_text(name).split() if t]
    if drop_stopwords:
        toks = [t for t in toks if t not in PLAYER_TOKEN_STOPWORDS]
    return tuple(toks)


def token_signature(tokens: Iterable[str]) -> str:
    return " ".join(sorted(set(tokens)))


def add_team_identity_columns(df):
    """Add conservative canonical team identity helper columns."""
    out = df.copy()
    for col in ["team", "home_team", "away_team", "opponent"]:
        if col in out.columns:
            out[f"{col}_canonical"] = out[col].map(canonical_team_name)
    if "team" in out.columns and "team_id" not in out.columns:
        out["team_id"] = out["team"].map(canonical_team_name)
    return out


def add_player_identity_columns(df):
    """Add conservative canonical player identity helper columns."""
    out = df.copy()
    if "player" in out.columns:
        out["player_canonical"] = out["player"].map(canonical_player_name)
    if "player_id" not in out.columns and "player" in out.columns:
        out["player_id"] = out["player"].map(canonical_player_name)
    return out
