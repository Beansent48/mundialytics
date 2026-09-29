"""The 2026-09-29 event-market upgrades: league-season level, ESPN referees and
lineup-priced player props (props/team_props.py, props/player_props.py).

What makes them safe to serve: every walk-forward feature reads EARLIER dates
only, the EPL keeps its own referee path, an unknown referee means
league-typical, and the morning player price is untouched unless a confirmed
lineup is passed.

Run:  .venv/Scripts/python.exe -m pytest tests/test_event_markets_level_referee.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.props import team_props as tp  # noqa: E402
from mundialytics.props.player_props import PlayerPropsModel  # noqa: E402


def _matches() -> pd.DataFrame:
    """Two LaLiga seasons (fouls 20 then 30 per match) + an EPL season."""
    rows = []
    for season, start, tot, comp in [("2024-2025", "2024-09-01", 20, "LaLiga"),
                                     ("2025-2026", "2025-09-01", 30, "LaLiga"),
                                     ("2025-2026", "2025-09-01", 26, "Premier League")]:
        for k in range(6):
            rows.append({"match_id": f"{comp[:3]}{season}{k}", "competition": comp, "season": season,
                         "date": pd.Timestamp(start) + pd.Timedelta(days=7 * k),
                         "home_team": f"h{k}", "away_team": f"a{k}",
                         "home_fouls": tot / 2, "away_fouls": tot / 2,
                         "referee": "Ref A" if k % 2 == 0 else "Ref B"})
    return pd.DataFrame(rows)


def test_league_level_reads_only_earlier_dates():
    m = _matches()
    lvl, latest = tp._league_levels(m, "home_fouls", "away_fouls", k=70)
    first = lvl["LaL2025-20260"]
    # opening match of a season: nothing played yet -> previous season's side mean
    assert first == pytest.approx(20 / 2)
    # later matches move toward the new level, shrunk by k
    second = lvl["LaL2025-20261"]
    assert second == pytest.approx((30 + 70 * 20) / (1 + 70) / 2)
    assert latest["LaLiga"] == pytest.approx((6 * 30 + 70 * 20) / (6 + 70) / 2)


def test_referee_deviation_is_walk_forward_and_skips_epl():
    m = _matches()
    rd, latest = tp._referee_deviation(m, "home_fouls", "away_fouls")
    # a referee's first match carries no information about him
    assert rd["LaL2024-20250"] == 0.0
    # the EPL keeps its football-data referee model: always 0 there
    assert (rd[[i for i in rd.index if i.startswith("Pre")]] == 0.0).all()
    # LaLiga 2025/26 ran 10 fouls above the previous season: Ref A's deviation
    # after his 3 matches of 2024/25 (dev 0) and 2 of 2025/26 is shrunk, positive
    assert rd["LaL2025-20264"] == pytest.approx(20.0 / (5 + tp.REF_DEV_SHRINK))
    assert tp._ref_key("Alejandro José Hernández Hernández") in {
        "alejandro jose hernandez hernandez"}
    assert set(latest) == {"ref a", "ref b"}


def test_unknown_referee_or_epl_fixture_is_league_typical():
    model = tp.TeamPropsModel()
    model._ref_dev = {"yellows": {"ref a": 0.8}}
    model._team_comp = {"betis": "LaLiga", "arsenal": "Premier League"}
    model._epl_teams = {"arsenal", "chelsea"}
    assert model.referee_deviation("yellows", "Ref A", "betis", "sevilla") == 0.8
    assert model.referee_deviation("yellows", None, "betis", "sevilla") is None
    assert model.referee_deviation("yellows", "Nobody", "betis", "sevilla") is None
    assert model.referee_deviation("yellows", "Ref A", "arsenal", "chelsea") is None


def _player_rows() -> pd.DataFrame:
    """Striker S starts every game (~85'); winger W comes off the bench (~20')."""
    rows = []
    for g in range(20):
        date = pd.Timestamp("2025-08-15") + pd.Timedelta(days=7 * g)
        rows.append(dict(player_id=1, player="Sam Striker", team="Club", game_id=g, date=date,
                         position="FW", minutes=85, xg=0.5, goals=g % 2, shots=3, xa=0.1,
                         assists=0, yellow_cards=0))
        rows.append(dict(player_id=2, player="Will Winger", team="Club", game_id=g, date=date,
                         position="Sub", minutes=20, xg=0.1, goals=0, shots=1, xa=0.05,
                         assists=0, yellow_cards=0))
        rows.append(dict(player_id=3, player="Dan Defender", team="Club", game_id=g, date=date,
                         position="DC", minutes=90, xg=0.02, goals=0, shots=0, xa=0.0,
                         assists=0, yellow_cards=g % 3 == 0))
    return pd.DataFrame(rows)


def test_lineup_prices_each_player_on_his_role_tonight():
    pp = PlayerPropsModel().fit(_player_rows())
    morning = pp.predict_team_players("Club").set_index("player")
    tonight, missing = pp.predict_lineup("Club", ["Sam Striker", "Will Winger"], ["Dan Defender"])
    tonight = tonight.set_index("player")
    assert missing == []
    # the usual sub starting tonight is priced on starter minutes -> goes up
    assert tonight.loc["Will Winger", "exp_min"] > morning.loc["Will Winger", "exp_min"]
    assert tonight.loc["Will Winger", "p_shots_over_1_5"] > morning.loc["Will Winger", "p_shots_over_1_5"]
    # the regular on the bench tonight -> goes down
    assert tonight.loc["Dan Defender", "exp_min"] < morning.loc["Dan Defender", "exp_min"]
    assert bool(tonight.loc["Dan Defender", "started"]) is False
    # names ESPN lists that the model never measured are reported, not invented
    _, missing = pp.predict_lineup("Club", ["Sam Striker", "Nobody Known"])
    assert missing == ["Nobody Known"]


def test_morning_price_is_unchanged_by_the_lineup_machinery():
    pp = PlayerPropsModel().fit(_player_rows())
    a = pp.predict_team_players("Club", 1.2)
    b = pp._price(pp._players.loc[a.index].copy(), "Club", 1.2)
    pd.testing.assert_frame_equal(a, b)
    assert np.isfinite(a["p_anytime_scorer"]).all()
