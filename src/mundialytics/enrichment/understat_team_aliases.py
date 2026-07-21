from __future__ import annotations

"""Understat team-name -> foundation (football-data.co.uk) name mapping.

The modeling foundation (`foundation_big5_multi_season.csv`) stores team names in
football-data.co.uk's short lowercase convention ("paris sg", "m gladbach",
"milan"). Understat uses fuller display names ("Paris Saint Germain", "Borussia
M.Gladbach", "AC Milan"). Of the 161 distinct Understat Big5 team names, 119
already normalize to a foundation name; the 42 below do not and need an explicit
alias so match-level xG joins cleanly onto the foundation.

Keys are the *normalized* Understat name (see normalize_provider_name); values are
the exact foundation team string. Use `to_foundation_name` to convert any Understat
team name to its foundation counterpart (falls back to the normalized name, which
is correct for the 119 direct-match teams).
"""

from mundialytics.data_quality.team_registry import normalize_provider_name

# normalized-understat-name -> foundation-name
UNDERSTAT_TO_FOUNDATION: dict[str, str] = {
    "ac milan": "milan",
    "arminia bielefeld": "bielefeld",
    "athletic club": "ath bilbao",
    "atletico madrid": "ath madrid",
    "bayer leverkusen": "leverkusen",
    "borussia dortmund": "dortmund",
    "borussia m gladbach": "m gladbach",
    "celta vigo": "celta",
    "clermont foot": "clermont",
    "deportivo la coruna": "la coruna",
    "eintracht frankfurt": "ein frankfurt",
    "espanyol": "espanol",
    "fc cologne": "fc koln",
    "fc heidenheim": "heidenheim",
    "fortuna duesseldorf": "fortuna dusseldorf",
    "gfc ajaccio": "ajaccio gfco",
    "greuther fuerth": "greuther furth",
    "hamburger sv": "hamburg",
    "hannover 96": "hannover",
    "hertha berlin": "hertha",
    "mainz 05": "mainz",
    "manchester city": "man city",
    "manchester united": "man united",
    "newcastle united": "newcastle",
    "nottingham forest": "nott m forest",
    "nuernberg": "nurnberg",
    "paris saint germain": "paris sg",
    "parma calcio 1913": "parma",
    "queens park rangers": "qpr",
    "rasenballsport leipzig": "rb leipzig",
    "rayo vallecano": "vallecano",
    "real betis": "betis",
    "real oviedo": "oviedo",
    "real sociedad": "sociedad",
    "real valladolid": "valladolid",
    "sc bastia": "bastia",
    "sd huesca": "huesca",
    "saint etienne": "st etienne",
    "sporting gijon": "sp gijon",
    "vfb stuttgart": "stuttgart",
    "west bromwich albion": "west brom",
    "wolverhampton wanderers": "wolves",
}


def to_foundation_name(understat_name: str) -> str:
    """Map an Understat team name to its foundation (football-data) counterpart."""
    norm = normalize_provider_name(understat_name)
    return UNDERSTAT_TO_FOUNDATION.get(norm, norm)
