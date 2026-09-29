"""Player markets in a season Understat never published (v0.57.0).

ESPN rows feed the player state (PlayerPropsModel._append_current), the published
shortlist ranks by recent starts (likely_xi), and settlement finds a player
under ESPN's spelling of his name (track_record._player_finder).

Run:  .venv/Scripts/python.exe -m pytest tests/test_player_fresh_form.py -q
"""
import sys
from collections import namedtuple
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.props.player_props import PlayerPropsModel, _espn_pos_group, likely_xi  # noqa: E402
from mundialytics.serving.track_record import _player_finder  # noqa: E402


def _understat() -> pd.DataFrame:
    """Last season: Old Nine starts every game, Young Nine sits on the bench."""
    rows = []
    for g in range(12):
        d = pd.Timestamp("2026-02-01") + pd.Timedelta(days=7 * g)
        rows.append(dict(player_id=1, player="Old Nine", team="Club", game_id=g, date=d, position="FW",
                         minutes=88, xg=0.5, goals=g % 2, shots=3, xa=0.1, assists=0, yellow_cards=0))
        rows.append(dict(player_id=2, player="Young Nine", team="Club", game_id=g, date=d, position="Sub",
                         minutes=10, xg=0.05, goals=0, shots=0, xa=0.0, assists=0, yellow_cards=0))
    for g in range(3):
        d = pd.Timestamp("2026-02-01") + pd.Timedelta(days=7 * g)
        rows.append(dict(player_id=2, player="Young Nine", team="Club", game_id=100 + g, date=d,
                         position="FW", minutes=70, xg=0.3, goals=0, shots=2, xa=0.0, assists=0,
                         yellow_cards=0))
    return pd.DataFrame(rows)


def _espn() -> pd.DataFrame:
    """This season: Young Nine starts and scores, Old Nine is gone from the squad;
    Neue Spieler, never seen by Understat, starts at centre-back."""
    rows = []
    for k in range(5):
        d = pd.Timestamp("2026-08-20") + pd.Timedelta(days=7 * k)
        base = dict(event_id=900 + k, season="2026-2027", competition="LaLiga", date=d.strftime("%Y-%m-%d"),
                    team="club", subbed_in=False, sot=0, red_cards=0, own_goals=0, sub_ins=0)
        rows.append({**base, "player": "Young Nine", "position": "CF-L", "starter": True, "goals": 1,
                     "shots": 4, "assists": 0, "yellow_cards": 0, "appearances": 1})
        rows.append({**base, "player": "Neue Spieler", "position": "CD-L", "starter": True, "goals": 0,
                     "shots": 0, "assists": 0, "yellow_cards": 1, "appearances": 1})
    return pd.DataFrame(rows)


def test_espn_positions_map_to_groups():
    assert _espn_pos_group("G") == "GK"
    assert _espn_pos_group("CD-R") == "DEF"
    assert _espn_pos_group("AM-L") == "ATT"
    assert _espn_pos_group("CF-L") == "FW"
    assert _espn_pos_group("RM") == "MID"
    assert _espn_pos_group("SUB") is None


def test_current_rows_move_the_player_state_and_the_shortlist():
    frozen = PlayerPropsModel().fit(_understat())
    fresh = PlayerPropsModel().fit(_understat(), current=_espn())
    assert fresh._n_current_rows == 10
    f = frozen.predict_team_players("Club").set_index("player")
    n = fresh.predict_team_players("Club").set_index("player")
    # the bench player who now starts and scores: more minutes, better odds
    assert n.loc["Young Nine", "exp_min"] > f.loc["Young Nine", "exp_min"]
    assert n.loc["Young Nine", "p_anytime_scorer"] > f.loc["Young Nine", "p_anytime_scorer"]
    # a player new to the big five gets a record and a defender's rates
    assert "Neue Spieler" in n.index
    assert fresh._players.set_index("player").loc["Neue Spieler", "pgroup"] == "DEF"
    # the shortlist follows this season's starts, not last season's minutes
    top = likely_xi(fresh.predict_team_players("Club"), 2)
    assert set(top["player"]) == {"Young Nine", "Neue Spieler"}


def test_likely_xi_falls_back_to_minutes_without_start_share():
    d = pd.DataFrame({"player": ["a", "b", "c"], "exp_min": [30, 90, 60]})
    assert list(likely_xi(d, 2)["player"]) == ["b", "c"]


Row = namedtuple("Row", "player goals")


def test_settlement_finds_espn_spelling_within_the_team_match_only():
    plk = {("2026-09-20", "real madrid", "Kylian Mbappé"): Row("Kylian Mbappé", 1),
           ("2026-09-20", "getafe", "Pape Gueye"): Row("Pape Gueye", 0)}
    find = _player_finder(plk)
    assert find("2026-09-20", "real madrid", "Kylian Mbappe-Lottin").goals == 1
    # the same name at another club or another date is not a match
    assert find("2026-09-20", "getafe", "Kylian Mbappe-Lottin") is None
    assert find("2026-09-27", "real madrid", "Kylian Mbappe-Lottin") is None
