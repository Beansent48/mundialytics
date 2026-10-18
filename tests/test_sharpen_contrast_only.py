"""The 1X2 sharpening goes on the home-vs-away contrast, not on the draw axis.

`p_i^gamma` renormalized is exactly (u, v) -> (gamma*u, gamma*v) on the two
log-odds against the draw, so the one exponent that fixes the model's
under-confidence on the favourite also drained the draw -- which did not need
it. The published trio priced draws at 0.2318 where 0.2540 happen: z = +5.4,
short in 6 of 6 walk-forward seasons, and since coherent markets that deficit
reached every exact score, BTTS and HT/FT number too. Applying the same
exponent to the contrast alone fixed it in 6 of 6 seasons on 1X2 log loss,
exact score, HT/FT and 1X2 RPS alike (scripts/draw_deficit/README.md).

These tests pin the two halves of that: the map does what its name says, and
the OLD map is still exactly what you get when the flag is off -- because the
flag defaulting to off is what keeps every engine pickled or scored before
2026-10-18 reproducible.

Run:  .venv/Scripts/python.exe -m pytest tests/test_sharpen_contrast_only.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS, PredictionEngine, deployed_markets_from_lambdas,
    markets_from_lambdas, sharpen_trio)

TRIOS = np.array([[0.80, 0.13, 0.07], [0.37, 0.27, 0.36], [0.19, 0.24, 0.57],
                  [0.50, 0.26, 0.24], [0.44, 0.26, 0.30]])
LAMBDAS = [(2.4, 0.7), (1.2, 1.2), (0.8, 1.9), (1.6, 1.0)]
GAMMAS = [0.8, 1.2, 1.3, 1.6]


# ── the old map is still the default ─────────────────────────────────────────

@pytest.mark.parametrize("gamma", GAMMAS)
def test_default_is_still_p_to_the_gamma(gamma):
    """Every number scored before 2026-10-18 has to stay reproducible."""
    want = np.clip(TRIOS, 1e-9, 1.0) ** gamma
    want = want / want.sum(axis=1, keepdims=True)
    assert np.abs(sharpen_trio(TRIOS, gamma) - want).max() < 1e-12


@pytest.mark.parametrize("lh,la", LAMBDAS)
def test_markets_from_lambdas_defaults_to_the_old_map(lh, la):
    k = DEPLOYED_CLUB_ENGINE_KWARGS
    args = dict(rho=k["outcome_rho"], temper=k["goal_temper"],
                sharpen_gamma=k["sharpen_gamma_1x2"], coherent=True)
    old, _ = markets_from_lambdas(lh, la, **args)
    raw, _ = markets_from_lambdas(lh, la, sharpen_gamma=1.0, rho=k["outcome_rho"],
                                  temper=k["goal_temper"])
    trio = np.array([raw["p_home_win"], raw["p_draw"], raw["p_away_win"]]) ** args["sharpen_gamma"]
    trio /= trio.sum()
    assert old["p_draw"] == pytest.approx(trio[1], abs=1e-12)


@pytest.mark.parametrize("contrast_only", [False, True])
def test_gamma_one_is_a_no_op_either_way(contrast_only):
    assert np.abs(sharpen_trio(TRIOS, 1.0, contrast_only) - TRIOS).max() < 1e-12


# ── what the new map does ────────────────────────────────────────────────────

@pytest.mark.parametrize("gamma", GAMMAS)
def test_the_draw_axis_is_untouched(gamma):
    """sqrt(p_h*p_a)/p_d is exp(s), the draw-vs-rest half of the log-odds."""
    out = sharpen_trio(TRIOS, gamma, True)
    before = np.sqrt(TRIOS[:, 0] * TRIOS[:, 2]) / TRIOS[:, 1]
    after = np.sqrt(out[:, 0] * out[:, 2]) / out[:, 1]
    assert np.abs(np.log(after / before)).max() < 1e-12


@pytest.mark.parametrize("gamma", GAMMAS)
def test_the_contrast_is_scaled_by_exactly_gamma(gamma):
    out = sharpen_trio(TRIOS, gamma, True)
    assert np.abs(np.log(out[:, 0] / out[:, 2])
                  - gamma * np.log(TRIOS[:, 0] / TRIOS[:, 2])).max() < 1e-12


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("contrast_only", [False, True])
def test_the_result_is_a_distribution_and_keeps_the_order(gamma, contrast_only):
    out = sharpen_trio(TRIOS, gamma, contrast_only)
    assert (out > 0).all()
    assert np.abs(out.sum(axis=1) - 1.0).max() < 1e-12
    # sharpening must not change WHICH outcome is the favourite
    assert (out.argmax(axis=1) == TRIOS.argmax(axis=1)).all()


def test_one_trio_and_many_agree():
    """The (N, 3) form is what the vectorized backtests call instead of copying."""
    many = sharpen_trio(TRIOS, 1.3, True)
    for i, t in enumerate(TRIOS):
        assert np.abs(sharpen_trio(t, 1.3, True) - many[i]).max() < 1e-12
        assert sharpen_trio(t, 1.3, True).shape == (3,)


def test_it_gives_draw_mass_back_relative_to_the_old_map():
    """The deployed change, stated as an inequality rather than a number."""
    g = DEPLOYED_CLUB_ENGINE_KWARGS["sharpen_gamma_1x2"]
    assert g > 1.0
    old, new = sharpen_trio(TRIOS, g), sharpen_trio(TRIOS, g, True)
    assert (old[:, 1] < TRIOS[:, 1]).all()        # one exponent always costs draw mass
    assert (new[:, 1] > old[:, 1]).all()          # the contrast-only map gives some back
    assert (new[:, 1] <= TRIOS[:, 1] + 1e-12).all()   # but not all: cosh(gamma*d) >= cosh(d)


# ── the deployed chain ───────────────────────────────────────────────────────

def test_the_deployed_config_has_it_on():
    assert DEPLOYED_CLUB_ENGINE_KWARGS["sharpen_contrast_only"] is True


@pytest.mark.parametrize("lh,la", LAMBDAS)
def test_the_deployed_chain_uses_it(lh, la):
    k = DEPLOYED_CLUB_ENGINE_KWARGS
    probs, _ = deployed_markets_from_lambdas(lh, la)
    want, _ = markets_from_lambdas(
        lh, la, rho=k["outcome_rho"], temper=k["goal_temper"],
        sharpen_gamma=k["sharpen_gamma_1x2"], sharpen_contrast_only=True, coherent=True)
    assert probs["p_draw"] == pytest.approx(want["p_draw"], abs=1e-12)


@pytest.mark.parametrize("lh,la", LAMBDAS)
def test_the_deployed_matrix_still_sums_to_the_published_1x2(lh, la):
    """Coherence does not need the invariant broken to get the draw back."""
    probs, dist = deployed_markets_from_lambdas(lh, la)
    m = dist.matrix.to_numpy()
    assert np.tril(m, -1).sum() == pytest.approx(probs["p_home_win"], abs=1e-12)
    assert np.trace(m) == pytest.approx(probs["p_draw"], abs=1e-12)
    assert np.triu(m, 1).sum() == pytest.approx(probs["p_away_win"], abs=1e-12)
    assert m.sum() == pytest.approx(1.0)


def test_the_engine_passes_the_flag_through():
    on = PredictionEngine(sharpen_gamma_1x2=1.3, outcome_rho=-0.06, goal_temper=1.05,
                          coherent_markets=True, sharpen_contrast_only=True)
    off = PredictionEngine(sharpen_gamma_1x2=1.3, outcome_rho=-0.06, goal_temper=1.05,
                           coherent_markets=True, sharpen_contrast_only=False)
    assert on.sharpen_contrast_only is True
    assert off.sharpen_contrast_only is False
    assert (on.markets_from_lambdas(1.6, 1.0)[0]["p_draw"]
            > off.markets_from_lambdas(1.6, 1.0)[0]["p_draw"])


def test_an_engine_pickled_before_the_flag_still_prices():
    """getattr fallback: a cached fit from before 2026-10-18 loads without it."""
    eng = PredictionEngine(sharpen_gamma_1x2=1.3, outcome_rho=-0.06, goal_temper=1.05,
                           coherent_markets=True, sharpen_contrast_only=True)
    del eng.sharpen_contrast_only
    probs, _ = eng.markets_from_lambdas(1.6, 1.0)
    old = sharpen_trio([0.5, 0.26, 0.24], 1.3)
    assert probs["p_draw"] > 0 and old[1] > 0  # falls back to the old map, no crash
