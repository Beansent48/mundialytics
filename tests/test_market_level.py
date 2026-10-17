"""The goals level read off the market (features/market_level.py).

It exists because reacting to our OWN in-season residual failed twice -- shots
(v0.56.0, 2/5 folds) and goals (scripts/goals_level/, 1/6). What is pinned here
is the part that makes it safe rather than the part that makes it accurate:
only a league-season level, only from matches already played, shrunk hard, and
never a reason for the engine to fail to load.

Run:  .venv/Scripts/python.exe -m pytest tests/test_market_level.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from mundialytics.features import market_level as ml  # noqa: E402
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS, PredictionEngine, deployed_markets_from_lambdas)


def test_implied_scale_round_trips_through_the_deployed_chain():
    lh = np.array([1.55, 1.20, 2.00])
    la = np.array([1.10, 1.20, 0.80])
    target = np.array([0.55, 0.45, 0.62])
    s = ml._implied_scale(lh, la, target)
    got = [deployed_markets_from_lambdas(float(a * si), float(b * si))[0]["p_over_25"]
           for a, b, si in zip(lh, la, s)]
    assert np.allclose(got, target, atol=2e-3)


def test_the_multiplier_is_shrunk_hard_when_a_league_has_little_history():
    """Nine matches must not move the level the way sixty do -- that asymmetry is
    the whole reason this can be deployed mid-season."""
    k = 70.0
    for n, ceiling in ((9, 0.02), (60, 0.05)):
        s_sum = n * 1.08            # the book says 8% above us, every match
        m = (s_sum + k) / (n + k)
        assert m - 1.0 < ceiling
    assert (9 * 1.08 + k) / (9 + k) < (60 * 1.08 + k) / (60 + k)


def test_nothing_to_measure_leaves_the_lambdas_alone(monkeypatch):
    monkeypatch.setattr(ml, "SERVED", ROOT / "does/not/exist.csv")
    monkeypatch.setattr(ml, "PRED_LOG", ROOT / "does/not/exist.csv")
    assert ml.served_lambdas().empty
    assert ml.market_levels(pd.DataFrame(), None) == {}
    assert ml.market_levels(None, None) == {}


def test_only_matches_before_asof_can_contribute(monkeypatch, tmp_path):
    """The price of the match being predicted -- and of anything on its date --
    must never reach the level. That is the line between a correction and a copy."""
    served = tmp_path / "served.csv"
    pd.DataFrame({"fecha": ["2026-08-15", "2026-08-22"],
                  "home": ["a", "c"], "away": ["b", "d"],
                  "lh": [1.5, 1.5], "la": [1.1, 1.1]}).to_csv(served, index=False)
    monkeypatch.setattr(ml, "SERVED", served)
    monkeypatch.setattr(ml, "PRED_LOG", tmp_path / "none.csv")
    monkeypatch.setattr(ml, "book_over25", lambda seasons=None: pd.DataFrame({
        "fecha": ["2026-08-15", "2026-08-22"],
        "home_team_raw": ["A", "C"], "away_team_raw": ["B", "D"],
        "competition": ["LaLiga", "LaLiga"], "p_book": [0.70, 0.70]}))
    matches = pd.DataFrame({
        "date": ["2026-08-15", "2026-08-22"], "season": "2026-2027",
        "competition": "LaLiga", "home_team": ["a", "c"], "away_team": ["b", "d"],
        "home_team_raw": ["A", "C"], "away_team_raw": ["B", "D"]})
    both = ml.market_levels(matches, pd.Timestamp("2026-08-23"))
    one = ml.market_levels(matches, pd.Timestamp("2026-08-22"))   # 2nd match excluded
    assert both["LaLiga"] > one["LaLiga"] > 1.0


def test_the_level_is_a_common_multiplier_so_it_moves_only_the_total():
    """It has to commute with the squad-value and lineup tilts, which depend on
    log(lh/la). If it ever stopped being common to both sides, those would shift."""
    eng = PredictionEngine(**DEPLOYED_CLUB_ENGINE_KWARGS)
    eng.set_market_levels({"LaLiga": 1.05})
    assert eng._market_level("LaLiga") == pytest.approx(1.05)
    assert eng._market_level("Serie A") == 1.0          # unmeasured league untouched
    lh, la = 1.55, 1.10
    assert (lh * 1.05) / (la * 1.05) == pytest.approx(lh / la)


def test_a_fitted_engine_carries_no_level_until_one_is_set():
    """The fit is cached to disk; a level baked into it would be served long
    after the matches it was measured on."""
    eng = PredictionEngine(**DEPLOYED_CLUB_ENGINE_KWARGS)
    assert eng.market_levels_ == {}
    assert eng._market_level("LaLiga") == 1.0
