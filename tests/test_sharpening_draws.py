"""The sharpening forms used to test the published 1X2's draw deficit.

Research code (`scripts/experiment_sharpening_draws.py`), not part of the
deployed chain, and the experiment came out "real but nearly free"
(`scripts/sharpening_draws/README.md`). Tested anyway: a result this small is
only worth recording if the harness behind it was right, and both of its
findings are claims about the code, so they are pinned here.

What these tests fix in place:
  * every form returns a normalized trio, and at parameter 1.0 every form is
    the identity -- so a form can only be scored on its sharpening;
  * `pair` leaves the draw bit-exact, `anchor` leaves a base-rate trio
    bit-exact, and `pair_draw` at delta = 1 IS `pair` -- the three defining
    properties the README leans on;
  * the deployed `power` form shrinks the draw on every realistic football
    trio -- the mechanism behind the defect;
  * the experiment's region-mass shortcut reproduces the engine's own coherent
    markets, and exact score decomposes into the 1X2 plus a term no trio can
    reach -- the structural finding.

Run:  .venv/Scripts/python.exe -m pytest tests/test_sharpening_draws.py -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.distributions import scoreline_distribution  # noqa: E402
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS as K, markets_from_lambdas)

# lambda pairs spanning what the engine actually serves: heavy favourite,
# even game, low-scoring, away favourite.
CASES = [(2.4, 0.7), (1.2, 1.2), (0.8, 0.9), (1.0, 1.9), (1.6, 1.1), (2.0, 2.0)]
GAMMAS = [1.1, 1.3, 1.6, 2.0]


@pytest.fixture(scope="module")
def exp():
    """The experiment script, loaded by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "experiment_sharpening_draws", ROOT / "scripts/experiment_sharpening_draws.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def raw_trio(lh, la):
    """The engine's own unsharpened trio, as region masses of its matrix."""
    m = scoreline_distribution(lh, la, max_goals=10, normalize=True,
                               dixon_coles_rho=K["outcome_rho"],
                               temper=K["goal_temper"]).matrix.to_numpy(float)
    i, j = np.indices(m.shape)
    return np.array([[m[i > j].sum(), m[i == j].sum(), m[i < j].sum()]]), m


BASE = np.array([0.45, 0.25, 0.30])


def test_identity_at_parameter_one(exp):
    """At gamma = 1 no form may move the trio, or it is a different model."""
    for lh, la in CASES:
        p, _ = raw_trio(lh, la)
        for name, th in (("power", (1.0,)), ("pair", (1.0,)),
                         ("anchor", (1.0,)), ("pair_draw", (1.0, 1.0))):
            out = exp.FORMS[name](p, th, BASE)
            assert np.abs(out - p).max() < 1e-12, name


@pytest.mark.parametrize("lh,la", CASES)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_every_form_stays_normalized(exp, lh, la, gamma):
    p, _ = raw_trio(lh, la)
    for name in exp.FORMS:
        th = (gamma, 1.05) if name == "pair_draw" else (gamma,)
        out = exp.FORMS[name](p, th, BASE)
        assert abs(out.sum() - 1.0) < 1e-12, name
        assert (out > 0).all(), name


@pytest.mark.parametrize("lh,la", CASES)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_pair_leaves_the_draw_untouched(exp, lh, la, gamma):
    """The whole point of `pair`: the draw keeps the model's own value."""
    p, _ = raw_trio(lh, la)
    out = exp.f_pair(p, (gamma,), BASE)
    assert out[0, 1] == p[0, 1]


@pytest.mark.parametrize("lh,la", CASES)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_deployed_power_form_shrinks_every_non_modal_draw(exp, lh, la, gamma):
    """The mechanism behind the defect, as the exact law rather than a tendency.

    For softmax(g log p), q_i < p_i iff p_i ** (g - 1) < sum_j p_j ** g, so a
    component shrinks when it sits below that threshold. On the 10,403 served
    trios that threshold coincides with "is not the largest of the three" in
    100.00% of matches, and the draw is the largest in 2 of them (0.02%) -- both
    low-scoring even games, which is why (0.8, 0.9) is in CASES. So the draw
    shrinks exactly when it is not the modal outcome, which is almost always.
    """
    p, _ = raw_trio(lh, la)
    out = exp.f_power(p, (gamma,), BASE)
    assert (out[0, 1] < p[0, 1]) == (p[0].argmax() != 1)
    assert out[0].argmax() == p[0].argmax()  # the favourite is still the favourite


@pytest.mark.parametrize("gamma", GAMMAS)
def test_anchor_fixes_a_base_rate_trio(exp, gamma):
    """`anchor` sharpens the deviation from the base rate, so no deviation
    means no change -- at any gamma."""
    p = BASE.reshape(1, 3).copy()
    out = exp.f_anchor(p, (gamma,), BASE)
    assert np.abs(out - p).max() < 1e-12


@pytest.mark.parametrize("lh,la", CASES)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_pair_draw_at_one_is_pair(exp, lh, la, gamma):
    p, _ = raw_trio(lh, la)
    a = exp.f_pair_draw(p, (gamma, 1.0), BASE)
    b = exp.f_pair(p, (gamma,), BASE)
    assert np.abs(a - b).max() < 1e-15


@pytest.mark.parametrize("lh,la", CASES)
def test_region_mass_shortcut_reproduces_the_engine(exp, lh, la):
    """The harness claim: markets as region-weighted masses ARE the engine's
    coherent markets, so the arms differ only by the sharpening."""
    p, m = raw_trio(lh, la)
    probs, _ = markets_from_lambdas(
        lh, la, rho=K["outcome_rho"], temper=K["goal_temper"],
        sharpen_gamma=float(K["sharpen_gamma_1x2"]),
        coherent=bool(K["coherent_markets"]))
    pub = np.array([probs["p_home_win"], probs["p_draw"], probs["p_away_win"]])
    i, j = np.indices(m.shape)
    region = np.where(i > j, 0, np.where(i == j, 1, 2))
    tot = i + j
    ratio = pub / p[0]
    for cond, expected in (((i > 0) & (j > 0), probs["p_btts"]),
                           (tot > 1.5, probs["p_over_15"]),
                           (tot > 2.5, probs["p_over_25"]),
                           (tot > 3.5, probs["p_over_35"])):
        got = sum(m[(region == k) & cond].sum() * ratio[k] for k in range(3))
        assert abs(got - expected) < 1e-12


@pytest.mark.parametrize("lh,la", CASES)
@pytest.mark.parametrize("gamma", [1.0, 1.3, 2.0])
def test_exact_score_decomposes_into_the_1x2_plus_an_untouchable_term(exp, lh, la, gamma):
    """The structural finding: under coherent reweighting the exact-score log
    loss is the 1X2 log loss plus a within-region term carrying no trio, so no
    sharpening can move exact score except through the 1X2."""
    p, m = raw_trio(lh, la)
    pub = exp.f_power(p, (gamma,), BASE)[0]
    i, j = np.indices(m.shape)
    region = np.where(i > j, 0, np.where(i == j, 1, 2))
    for score_cell in [(1, 0), (1, 1), (0, 2), (2, 1), (3, 3)]:
        r = region[score_cell]
        mass = m[region == r].sum()
        exact = -np.log(m[score_cell] * pub[r] / mass)
        assert abs(exact - (-np.log(pub[r]) + -np.log(m[score_cell] / mass))) < 1e-12
