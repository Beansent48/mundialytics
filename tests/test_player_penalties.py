"""Current-season penalties for the player model (v0.58.0).

ESPN's player file marks penalty goals but never the misses, so attempts come
from the commentary (enrichment.text_xg.penalties_from_commentary) and ride on
the current rows into the pen-taker split (PlayerPropsModel._append_current).

Run:  .venv/Scripts/python.exe -m pytest tests/test_player_penalties.py -q
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.enrichment.text_xg import parse_shot, penalties_from_commentary, shooter  # noqa: E402
from mundialytics.props.player_props import PlayerPropsModel  # noqa: E402

GOAL = ("Goal! Newcastle United 2, Liverpool 2. Dominik Szoboszlai (Liverpool) converts the "
        "penalty with a right footed shot to the top left corner.")
MISS = ("Penalty missed! Still Brentford 3, Tottenham Hotspur 0. Igor Thiago (Brentford) hits "
        "the left post with a right footed shot.")
SAVED_DOT = ("Penalty saved. Enzo Le Fée (Sunderland) fails to capitalise on this great "
             "opportunity, right footed shot saved in the bottom left corner.")


def test_shooter_reads_the_name_before_the_team():
    assert shooter(GOAL) == "Dominik Szoboszlai"
    assert shooter(MISS) == "Igor Thiago"
    assert shooter("Corner, Arsenal. Conceded by William Saliba.") is None


def test_a_saved_penalty_written_with_a_full_stop_is_an_attempt():
    s = parse_shot(SAVED_DOT)
    assert s is not None and s["penalty"] and s["outcome"] == "saved"
    assert shooter(SAVED_DOT) == "Enzo Le Fée"


def test_penalties_from_commentary(tmp_path):
    f = tmp_path / "2026-2027.jsonl"
    g = {"event_id": "7", "date": "2026-09-01", "home": "Brentford", "away": "Sunderland",
         "commentary": [{"t": "10'", "text": MISS}, {"t": "20'", "text": SAVED_DOT},
                        {"t": "30'", "text": "Attempt saved. Igor Thiago (Brentford) header from the centre "
                                             "of the box is saved in the centre of the goal."}]}
    f.write_text(json.dumps(g) + "\n", encoding="utf-8")
    p = penalties_from_commentary(f)
    assert list(p["player"]) == ["Igor Thiago", "Enzo Le Fée"]
    assert not p["scored"].any()


def _history() -> pd.DataFrame:
    rows = []
    for g in range(10):
        d = pd.Timestamp("2026-02-01") + pd.Timedelta(days=7 * g)
        rows.append(dict(player_id=1, player="Spot Kicker", team="Club", game_id=g, date=d, position="FW",
                         minutes=90, xg=0.4, goals=0, shots=2, xa=0.1, assists=0, yellow_cards=0))
    return pd.DataFrame(rows)


def _current() -> pd.DataFrame:
    rows = []
    for k in range(4):
        rows.append(dict(event_id=500 + k, season="2026-2027", date=f"2026-09-0{k + 1}", team="club",
                         player="Spot Kicker", position="CF-L", starter=True, appearances=1,
                         goals=1, shots=2, assists=0, yellow_cards=0))
    return pd.DataFrame(rows)


def test_current_penalties_feed_the_taker_split():
    pens = pd.DataFrame({"event_id": ["500", "501", "502"], "player": ["Spot Kicker"] * 3,
                         "scored": [True, True, False]})
    without = PlayerPropsModel().fit(_history(), current=_current())
    with_p = PlayerPropsModel().fit(_history(), current=_current(), current_penalties=pens)
    a = without._players.set_index("player").loc["Spot Kicker"]
    b = with_p._players.set_index("player").loc["Spot Kicker"]
    assert a["p_pen60"] == 0 and b["p_pen60"] == 3 and b["t_pen60"] == 3
    assert with_p._team_pen_rate["Club"] > without._team_pen_rate["Club"]
    # a penalty is xG 0.76, not an ordinary shot: the non-penalty xG is what is left
    assert b["c_npxg"] < b["c_xg"]
