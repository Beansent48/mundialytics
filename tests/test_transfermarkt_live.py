"""Live squad values read from transfermarkt.com (providers/transfermarkt.py).

Weekly values instead of the June-frozen dump are worth another -0.0003 RPS in the
backtest. These tests pin the parsing on minimal pages shaped like the site's, and the
mapping of a club new to the Big Five (not in the history club map) to its canonical
name, which is where a silent miss would drop a promoted side's value.

Run:  .venv/Scripts/python.exe -m pytest tests/test_transfermarkt_live.py -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.features.squad_value import build_live, map_live_clubs  # noqa: E402
from mundialytics.providers.transfermarkt import (  # noqa: E402
    parse_league_clubs, parse_squad, parse_value, season_id)

LEAGUE_HTML = """<table class="items"><tbody>
<tr><td class="hauptlink"><a title="Arsenal FC" href="/fc-arsenal/startseite/verein/11/saison_id/2026">Arsenal</a></td></tr>
<tr><td class="hauptlink"><a title="Coventry City" href="/coventry-city/startseite/verein/990/saison_id/2026">Coventry</a></td></tr>
</tbody></table>"""

SQUAD_HTML = """<table class="items"><tbody>
<tr><td class="hauptlink"><a href="/david-raya/profil/spieler/262749">David Raya</a></td>
    <td class="rechts hauptlink"><a>€30.00m</a></td></tr>
<tr><td class="hauptlink"><a href="/young-one/profil/spieler/999">Young One</a></td>
    <td class="rechts hauptlink">-</td></tr>
<tr><td class="hauptlink"><a href="/cheap/profil/spieler/1000">Cheap Player</a></td>
    <td class="rechts hauptlink"><a>€500k</a></td></tr>
</tbody></table>"""


@pytest.mark.parametrize("text,value", [("€30.00m", 30e6), ("€500k", 5e5), ("€1.43bn", 1.43e9),
                                        ("-", None), ("", None), (None, None)])
def test_parse_value(text, value):
    assert parse_value(text) == (pytest.approx(value) if value else None)


def test_parse_league_and_squad():
    clubs = parse_league_clubs(LEAGUE_HTML)
    assert [c["club_id"] for c in clubs] == [11, 990]
    assert clubs[1] == {"club_id": 990, "slug": "coventry-city", "club_name": "Coventry City"}
    squad = parse_squad(SQUAD_HTML)
    assert [(p["player_id"], p["value_eur"]) for p in squad] == [(262749, 30e6), (999, None), (1000, 5e5)]


def test_season_id():
    assert season_id("2026-09-29") == 2026 and season_id("2027-03-01") == 2026


def test_new_club_mapped_by_name_within_its_league():
    tm = pd.DataFrame([
        {"competition": "Premier League", "club_id": 11, "club_name": "Arsenal FC", "player_id": 1, "value_eur": 9e7},
        {"competition": "Premier League", "club_id": 990, "club_name": "Coventry City", "player_id": 2, "value_eur": 5e6},
        {"competition": "Bundesliga", "club_id": 4097, "club_name": "SV 07 Elversberg", "player_id": 3, "value_eur": 2e6},
    ])
    club_map = pd.DataFrame({"club_id": [11], "team": ["arsenal"]})
    current = pd.DataFrame({"competition": ["Premier League", "Premier League", "Bundesliga"],
                            "team": ["arsenal", "coventry", "elversberg"]})
    assert map_live_clubs(tm, club_map, current) == {11: "arsenal", 990: "coventry", 4097: "elversberg"}
    teams, unmapped = build_live(tm, club_map, current, snap="2026-09-29")
    assert unmapped == [] and set(teams.team) == {"arsenal", "coventry", "elversberg"}
    assert teams.set_index("team").v18["arsenal"] == 9e7
