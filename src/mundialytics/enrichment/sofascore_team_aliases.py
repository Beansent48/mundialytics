from __future__ import annotations

"""Sofascore team-name -> foundation (football-data.co.uk) name mapping.

Same contract as understat_team_aliases: the modeling foundation stores team names
in football-data.co.uk's short lowercase convention ("man united", "ath madrid",
"ein frankfurt"). Sofascore uses fuller display names ("Manchester United",
"Atletico Madrid", "Eintracht Frankfurt"). Of the Big-5 first-division team names
Sofascore serves for 2026/27, 44 already normalize to a foundation name; the 52
below do not and need an explicit alias so match-level xG joins cleanly.

Keys are the *normalized* Sofascore name (see normalize_provider_name); values are
the exact foundation team string. Use `to_foundation_name` to convert any Sofascore
team name to its foundation counterpart (falls back to the normalized name, which is
correct for the direct-match teams). Built 2026-09-15 for the Sofascore xG connector
(see scripts/build_sofascore_xg_matches.py) after Understat froze 2026-05-24.
"""

from mundialytics.data_quality.team_registry import normalize_provider_name

# normalized-sofascore-name -> foundation-name
SOFASCORE_TO_FOUNDATION: dict[str, str] = {
    # England
    "brighton hove albion": "brighton",
    "coventry city": "coventry",
    "hull city": "hull",
    "ipswich town": "ipswich",
    "leeds united": "leeds",
    "liverpool fc": "liverpool",
    "manchester city": "man city",
    "manchester united": "man united",
    "newcastle united": "newcastle",
    "nottingham forest": "nott m forest",
    "tottenham hotspur": "tottenham",
    # Spain
    "athletic club": "ath bilbao",
    "atletico madrid": "ath madrid",
    "celta vigo": "celta",
    "deportivo alaves": "alaves",
    "deportivo de a coruna": "la coruna",
    "espanyol": "espanol",
    "fc barcelona": "barcelona",
    "levante ud": "levante",
    "malaga cf": "malaga",
    "rayo vallecano": "vallecano",
    "real betis": "betis",
    "real racing club": "santander",
    "real sociedad": "sociedad",
    # Italy
    "ac milan": "milan",
    "as roma": "roma",
    "ssc napoli": "napoli",
    # Germany
    "1 fc koln": "fc koln",
    "1 fc union berlin": "union berlin",
    "1 fsv mainz 05": "mainz",
    "bayer 04 leverkusen": "leverkusen",
    "borussia dortmund": "dortmund",
    "borussia m gladbach": "m gladbach",
    "eintracht frankfurt": "ein frankfurt",
    "fc augsburg": "augsburg",
    "fc bayern munchen": "bayern munich",
    "fc schalke 04": "schalke 04",
    "hamburger sv": "hamburg",
    "sc freiburg": "freiburg",
    "sc paderborn 07": "paderborn",
    "sv 07 elversberg": "elversberg",
    "sv werder bremen": "werder bremen",
    "tsg hoffenheim": "hoffenheim",
    "vfb stuttgart": "stuttgart",
    # France
    "as monaco": "monaco",
    "olympique lyonnais": "lyon",
    "olympique de marseille": "marseille",
    "paris saint germain": "paris sg",
    "rc lens": "lens",
    "rc strasbourg": "strasbourg",
    "stade brestois": "brest",
    "stade rennais": "rennes",
}


def to_foundation_name(sofascore_name: str) -> str:
    """Map a Sofascore team name to its foundation (football-data) counterpart."""
    norm = normalize_provider_name(sofascore_name)
    return SOFASCORE_TO_FOUNDATION.get(norm, norm)
