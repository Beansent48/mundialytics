"""The trio maps that were measured against the deployed 1X2 sharpening.

Research code (`scripts/draw_deficit/`), not part of the deployed chain. The
experiment recommends one of these maps, so the implementation is worth pinning
down: a recommendation built on a map that does not do what its name says would
be worse than no recommendation.

Four properties, and one structural claim:
  * every family is the identity at its null parameters, and `two_axis` with
    alpha = beta IS the deployed p^gamma family -- so the comparison is against
    a nested baseline, not a different model;
  * alpha = 1 leaves the draw-vs-rest log-odds exactly alone and beta scales the
    home-vs-away contrast exactly, which is the whole claim of the candidate;
  * the maps are probability distributions, and coherence survives all of them;
  * **the identity the recommendation rests on**: with the raw matrix fixed, the
    exact-score and HT/FT log losses equal the 1X2 log loss plus a constant no
    trio map can touch. If that ever stops holding, the experiment's objective
    is wrong and its conclusion goes with it.

Run:  .venv/Scripts/python.exe -m pytest tests/test_draw_deficit.py -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TRIOS = np.array([[0.80, 0.13, 0.07], [0.37, 0.27, 0.36],
                  [0.19, 0.24, 0.57], [0.50, 0.26, 0.24],
                  [0.44, 0.26, 0.30]])


@pytest.fixture(scope="module")
def S():
    """scripts/draw_deficit/score_trio_forms.py, loaded by path."""
    sys.path.insert(0, str(ROOT / "scripts/draw_deficit"))
    spec = importlib.util.spec_from_file_location(
        "score_trio_forms", ROOT / "scripts/draw_deficit/score_trio_forms.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def X(S):
    return sys.modules["experiment_bivariate_goals"]


# ── the families ──────────────────────────────────────────────────────────────

def test_null_parameters_are_the_identity(S):
    for fn, p in ((S.f_gamma, [1.0]), (S.f_two_axis, [1.0, 1.0]),
                  (S.f_vector, [1.0, 0.0, 0.0])):
        assert np.abs(fn(TRIOS, p) - TRIOS).max() < 1e-12


@pytest.mark.parametrize("g", [0.8, 1.2, 1.3, 1.6])
def test_two_axis_nests_the_deployed_family(S, g):
    """alpha = beta = gamma must BE p^gamma, or the baseline is not nested."""
    assert np.abs(S.f_two_axis(TRIOS, [g, g]) - S.f_gamma(TRIOS, [g])).max() < 1e-12


@pytest.mark.parametrize("beta", [1.0, 1.3, 1.6])
def test_alpha_one_leaves_the_draw_axis_exactly_alone(S, beta):
    """The candidate's claim: sqrt(p_h p_a)/p_d is untouched when alpha = 1.

    That ratio is exp(s), the draw-vs-rest half of the log-odds, so this is the
    statement "the sharpening no longer touches the draw axis" made exact.
    """
    out = S.f_two_axis(TRIOS, [1.0, beta])
    before = np.sqrt(TRIOS[:, 0] * TRIOS[:, 2]) / TRIOS[:, 1]
    after = np.sqrt(out[:, 0] * out[:, 2]) / out[:, 1]
    assert np.abs(np.log(after) - np.log(before)).max() < 1e-12


@pytest.mark.parametrize("beta", [0.8, 1.3, 1.6])
def test_beta_scales_the_home_away_contrast_exactly(S, beta):
    out = S.f_two_axis(TRIOS, [1.0, beta])
    before = np.log(TRIOS[:, 0] / TRIOS[:, 2])
    after = np.log(out[:, 0] / out[:, 2])
    assert np.abs(after - beta * before).max() < 1e-12


def test_the_deployed_gamma_drains_the_draw_and_the_candidate_drains_less(S):
    """The mechanism the whole experiment is about, on fixed inputs."""
    g = S.K["sharpen_gamma_1x2"]
    assert g > 1.0
    deployed = S.f_gamma(TRIOS, [g])[:, 1]
    candidate = S.f_two_axis(TRIOS, [1.0, g])[:, 1]
    assert (deployed < TRIOS[:, 1]).all()          # gamma always costs draw mass
    assert (candidate > deployed).all()            # alpha = 1 gives some back
    assert (candidate <= TRIOS[:, 1] + 1e-12).all()  # but not all of it: cosh(beta d)


@pytest.mark.parametrize("form,p", [("gamma", [1.25]), ("two_axis", [0.9, 1.3]),
                                    ("vector", [1.2, 0.1, -0.2])])
def test_every_form_returns_a_distribution(S, form, p):
    out = S.FORMS[form][0](TRIOS, np.array(p))
    assert out.shape == TRIOS.shape
    assert (out >= 0).all()
    assert np.abs(out.sum(axis=1) - 1.0).max() < 1e-12


def test_fit_recovers_a_known_sharpening(S):
    """Synthetic outcomes drawn from a known map; the fit must find it back."""
    rng = np.random.default_rng(11)
    raw = rng.dirichlet([4.0, 3.0, 3.0], size=40000)
    truth = S.f_gamma(raw, [1.45])
    y = np.array([rng.choice(3, p=row) for row in truth])
    assert S.fit_form(raw, y, "gamma")[0] == pytest.approx(1.45, abs=0.05)
    a, b = S.fit_form(raw, y, "two_axis")
    assert a == pytest.approx(1.45, abs=0.12)
    assert b == pytest.approx(1.45, abs=0.12)


# ── the structural claim the recommendation rests on ──────────────────────────

@pytest.fixture(scope="module")
def synthetic(X):
    """A small synthetic walk-forward frame: lambdas, results, served trio."""
    rng = np.random.default_rng(5)
    n = 4000
    lh = rng.uniform(0.6, 2.8, n)
    la = rng.uniform(0.5, 2.2, n)
    hg = rng.poisson(lh)
    ag = rng.poisson(la)
    df = pd.DataFrame({"season": np.where(np.arange(n) < n // 2, "A", "B"),
                       "match_id": np.arange(n), "hg": hg, "ag": ag,
                       "lh": lh, "la": la,
                       "home_goals_ht": rng.binomial(hg, 0.44),
                       "away_goals_ht": rng.binomial(ag, 0.44)})
    df["outcome"] = np.where(df.hg > df.ag, 0, np.where(df.hg == df.ag, 1, 2))
    return df


def test_exact_score_and_htft_are_the_1x2_log_loss_plus_a_constant(S, X, synthetic):
    """The identity: only `log t[region]` depends on the trio map.

    This is what makes the trio's log loss the right objective, and what makes
    "it helps the 1X2" and "it helps exact score and HT/FT" the same sentence.
    """
    M = S.raw_matrix(synthetic)
    raw = X.trio_of(M)
    cs_gap, ht_gap = [], []
    for t in (S.f_gamma(raw, [1.0]), S.f_gamma(raw, [1.3]), S.f_gamma(raw, [1.7]),
              S.f_two_axis(raw, [1.0, 1.3]), S.f_two_axis(raw, [0.7, 1.5]),
              S.f_vector(raw, [1.2, 0.3, -0.3])):
        a = X.aggregate(synthetic, S.score_trio(synthetic, M, t))
        cs_gap.append(a["cs"] - a["1x2_ll"])
        ht_gap.append(a["htft"] - a["1x2_ll"])
    assert np.ptp(cs_gap) < 1e-9, "exact score is not the 1X2 log loss plus a constant"
    assert np.ptp(ht_gap) < 1e-9, "HT/FT is not the 1X2 log loss plus a constant"


@pytest.mark.parametrize("form,p", [("gamma", [1.3]), ("two_axis", [1.0, 1.3]),
                                    ("vector", [1.25, -0.1, -0.12])])
def test_coherence_survives_every_form(S, X, synthetic, form, p):
    M = S.raw_matrix(synthetic)
    t = S.FORMS[form][0](X.trio_of(M), np.array(p))
    sc = S.score_trio(synthetic, M, t)
    assert sc["disagree"].max() < 1e-9
    m = X.reweight_to_trio(M, t)
    assert np.abs(X.trio_of(m) - t).max() < 1e-12
    assert np.abs(m.sum(axis=(1, 2)) - 1.0).max() < 1e-12


def test_decouple_is_the_only_arm_that_breaks_coherence(S, X, synthetic):
    M = S.raw_matrix(synthetic)
    raw = X.trio_of(M)
    pub = S.f_gamma(raw, [1.3])
    sc = S.score_trio(synthetic, M, pub, t_matrix=S.f_gamma(raw, [1.0]))
    assert sc["disagree"].max() > 0.01, "decoupling should visibly break the invariant"
