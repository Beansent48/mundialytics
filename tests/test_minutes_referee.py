"""Real minutes from the commentary and the referee on player yellows (v0.60.0).

Minutes follow Understat's clock (enrichment/espn_minutes.py); the referee's
card deviation moves each player's yellow card and nothing else; the lineup
pass reads the referee from the summary it already fetches for the XIs.

Run:  .venv/Scripts/python.exe -m pytest tests/test_minutes_referee.py -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from mundialytics.enrichment.espn_minutes import clock, minute, minutes_from_commentary  # noqa: E402
from mundialytics.props.player_props import REF_YC_B, PlayerPropsModel  # noqa: E402

COMMENTARY = [
    {"t": "45'+3'", "text": "First Half ends, Club 0, Rivals 0."},
    {"t": "62'", "text": "Substitution, Club. Sub Striker replaces Old Nine."},
    {"t": "70'", "text": "Keeper One (Club) is shown the red card."},
    {"t": "90'+4'", "text": "Substitution, Club. Late Man replaces Wing Back."},
]
STARTERS = ["Old Nine", "Keeper One", "Wing Back", "Full Timer"]
SQUAD = STARTERS + ["Sub Striker", "Late Man", "Unused Guy"]


def test_minute_and_clock():
    assert minute("62'") == 62 and minute("45'+3'") == 45 and minute("") is None
    # first-half stoppage runs into the second half (one minute short of all of it)
    assert clock("62'", 3) == 64
    assert clock("45'+2'", 3) == 47
    assert clock("90'+4'", 3) == 90


def test_minutes_from_commentary():
    m = minutes_from_commentary(COMMENTARY, STARTERS, SQUAD)
    assert m["Full Timer"] == 90
    assert m["Old Nine"] == 64 and m["Sub Striker"] == 26
    assert m["Keeper One"] == 72          # sent off at 70' + 2 of first-half stoppage
    assert m["Late Man"] == 1             # on in stoppage time plays 1, as Understat
    assert m["Unused Guy"] == 0


def _history() -> pd.DataFrame:
    rows = []
    for g in range(12):
        d = pd.Timestamp("2026-02-01") + pd.Timedelta(days=7 * g)
        rows.append(dict(player_id=1, player="Nine", team="Club", game_id=g, date=d, position="FW",
                         minutes=85, xg=0.4, goals=0, shots=3, xa=0.1, assists=0, yellow_cards=g % 4 == 0))
    return pd.DataFrame(rows)


def _current() -> pd.DataFrame:
    return pd.DataFrame([dict(event_id=700 + k, season="2026-2027", date=f"2026-09-0{k + 1}", team="club",
                              player="Nine", position="CF-L", starter=True, appearances=1,
                              goals=0, shots=1, assists=0, yellow_cards=0) for k in range(5)])


def test_real_minutes_replace_the_role_estimate():
    est = PlayerPropsModel().fit(_history(), current=_current())
    real = PlayerPropsModel().fit(_history(), current=_current(), current_minutes=pd.DataFrame(
        {"event_id": [str(700 + k) for k in range(5)], "player": "Nine", "minutes": 30}))
    assert real._players.loc[1, "avg_minp10"] < est._players.loc[1, "avg_minp10"]


def test_referee_moves_only_the_yellow_card():
    pp = PlayerPropsModel().fit(_history())
    base = pp.predict_team_players("Club").iloc[0]
    strict = pp.predict_team_players("Club", ref_dev=1.0).iloc[0]
    assert strict["p_yellow"] > base["p_yellow"]
    assert strict["p_anytime_scorer"] == base["p_anytime_scorer"]
    same = pp.predict_team_players("Club", ref_dev=None).iloc[0]
    assert same["p_yellow"] == base["p_yellow"]
    assert REF_YC_B == pytest.approx(0.16)


def test_lineup_pass_reads_the_referee_and_prices_team_markets():
    import log_lineup_pass as lp

    s = {"gameInfo": {"officials": [{"fullName": "Javier Alberola Rojas",
                                     "position": {"name": "Referee"}, "order": 1}]}}
    assert lp.referee_of(s) == "Javier Alberola Rojas"
    assert lp.referee_of({"gameInfo": {}}) is None

    class FakeTP:
        def predict_fixture(self, h, a, referee=None, lam_home=None, lam_away=None):
            self.referee = referee
            return {"yellows": {"over": {4.5: 0.4}, "over_home": {1.5: 0.6}}}

    tp = FakeTP()
    e = {"event_id": "1", "competition": "LaLiga", "kickoff": pd.Timestamp("2026-10-09T19:00")}
    rows = lp.lineup_team_rows(tp, e, "betis", "sevilla", {"home": 1.4, "away": 1.1},
                               "Javier Alberola Rojas", pd.Timestamp("2026-10-09T18:00"))
    assert tp.referee == "Javier Alberola Rojas"
    assert {(r["ambito"], r["linea"], r["prob"]) for r in rows} == {("Total", 4.5, 0.4), ("Local", 1.5, 0.6)}
