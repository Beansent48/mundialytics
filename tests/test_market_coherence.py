"""The page's markets come from one distribution that agrees with its 1X2.

The deployed 1X2 is sharpened; the scoreline matrix used to stay raw, so with
lambdas 2.4 / 0.7 the page said 84% home win while its own grid summed to 75%,
and the HT/FT paths implied a third full-time 1X2. The fix was validated on
six walk-forward seasons (scripts/experiment_market_coherence.py) before it
was switched on.

Run:  .venv/Scripts/python.exe -m pytest tests/test_market_coherence.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.props.half_time import HalfTimeModel  # noqa: E402
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS, deployed_markets_from_lambdas, markets_from_lambdas)

CASES = [(2.4, 0.7), (1.2, 1.2), (0.8, 1.9), (1.6, 1.0)]


@pytest.mark.parametrize("lh,la", CASES)
def test_deployed_matrix_sums_to_the_published_1x2(lh, la):
    probs, dist = deployed_markets_from_lambdas(lh, la)
    m = dist.matrix.to_numpy()
    assert np.tril(m, -1).sum() == pytest.approx(probs["p_home_win"], abs=1e-9)
    assert np.trace(m) == pytest.approx(probs["p_draw"], abs=1e-9)
    assert np.triu(m, 1).sum() == pytest.approx(probs["p_away_win"], abs=1e-9)
    assert m.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("lh,la", CASES)
def test_goal_markets_are_read_from_that_matrix(lh, la):
    probs, dist = deployed_markets_from_lambdas(lh, la)
    assert probs["p_over_25"] == pytest.approx(dist.total_goals_probability(2.5, "over"))
    assert probs["p_btts"] == pytest.approx(dist.p_btts)
    assert probs["p_over_25"] + probs["p_under_25"] == pytest.approx(1.0)


def test_coherence_does_not_move_the_1x2():
    k = DEPLOYED_CLUB_ENGINE_KWARGS
    args = dict(rho=k["outcome_rho"], temper=k["goal_temper"],
                sharpen_gamma=k["sharpen_gamma_1x2"])
    raw, _ = markets_from_lambdas(2.4, 0.7, coherent=False, **args)
    coh, _ = markets_from_lambdas(2.4, 0.7, coherent=True, **args)
    for key in ("p_home_win", "p_draw", "p_away_win"):
        assert coh[key] == raw[key]


def test_ht_ft_full_time_margin_matches_the_given_1x2():
    trio = (0.5607, 0.2454, 0.1939)
    paths = HalfTimeModel().predict_ht_ft(1.6, 1.0, ft_trio=trio)
    for i, b in enumerate("1X2"):
        ft = sum(v for k, v in paths.items() if k.endswith("/" + b))
        assert ft == pytest.approx(trio[i], abs=1e-9)
    assert sum(paths.values()) == pytest.approx(1.0)
