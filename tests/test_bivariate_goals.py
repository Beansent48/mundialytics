"""The goals-dependence families used to test the Dixon-Coles patch.

These are research code (`scripts/bivariate_goals/dependence.py`), not part of
the deployed chain, and the experiment that uses them came out negative
(`scripts/bivariate_goals/README.md`). They are still tested, for two reasons:
a negative result is only worth recording if the harness that produced it was
correct, and the first thing to doubt about an effect this small is the
implementation.

Four properties pin them down:
  * at parameter 0 every family IS the product-Poisson the engine already uses;
  * the marginals are exactly the deployed ones, so a family can only be scored
    on its dependence;
  * the shared component's covariance is exactly lam3, as Karlis-Ntzoufras
    says, and the Gaussian copula's CDF agrees with scipy to machine precision;
  * the vectorised path in the experiment script equals this reference code.

Run:  .venv/Scripts/python.exe -m pytest tests/test_bivariate_goals.py -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/bivariate_goals"))

import dependence as D  # noqa: E402
from mundialytics.statistical_core.distributions import scoreline_distribution  # noqa: E402

CASES = [(2.4, 0.7), (1.2, 1.2), (0.8, 1.9), (1.6, 1.0)]
PARAMS = {"dixon_coles": -0.06, "bivariate_poisson": 0.03,
          "frank_copula": 0.4, "gaussian_copula": -0.07}


@pytest.fixture(scope="module")
def experiment():
    """The experiment script, loaded by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "experiment_bivariate_goals", ROOT / "scripts/experiment_bivariate_goals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("lh,la", CASES)
@pytest.mark.parametrize("family", D.FAMILIES[1:])
def test_zero_parameter_is_the_deployed_independent_matrix(family, lh, la):
    """No family may be a different model at independence."""
    ind = D.joint_matrix(lh, la, "independent", temper=1.05)
    assert np.abs(D.joint_matrix(lh, la, family, 0.0, temper=1.05) - ind).max() < 1e-12
    deployed = scoreline_distribution(lh, la, dixon_coles_rho=0.0, temper=1.05)
    assert np.abs(ind - deployed.matrix.to_numpy()).max() < 1e-12


@pytest.mark.parametrize("lh,la", CASES)
def test_dixon_coles_family_reproduces_the_deployed_patch(lh, la):
    """The baseline arm has to BE the deployed matrix, or nothing is comparable."""
    mine = D.joint_matrix(lh, la, "dixon_coles", -0.06, temper=1.05, fix_marginals=False)
    ref = scoreline_distribution(lh, la, dixon_coles_rho=-0.06, temper=1.05)
    assert np.abs(mine - ref.matrix.to_numpy()).max() < 1e-12


@pytest.mark.parametrize("lh,la", CASES)
@pytest.mark.parametrize("family,param", list(PARAMS.items()))
def test_marginals_are_held_fixed(family, param, lh, la):
    m = D.joint_matrix(lh, la, family, param, temper=1.05)
    assert m.sum() == pytest.approx(1.0)
    assert np.abs(m.sum(axis=1) - D.marginal_pmf(lh, temper=1.05)).max() < 1e-10
    assert np.abs(m.sum(axis=0) - D.marginal_pmf(la, temper=1.05)).max() < 1e-10
    assert (m >= 0).all()


@pytest.mark.parametrize("lam3", [0.0, 0.01, 0.05, 0.15])
def test_shared_component_covariance_is_exactly_lam3(lam3):
    """Karlis-Ntzoufras: H = X1 + X3, A = X2 + X3 => Cov(H, A) = lam3."""
    m = D.bivariate_poisson(1.65, 1.15, lam3)
    assert D.implied_covariance(m) == pytest.approx(lam3, abs=1e-4)
    assert (m.sum(axis=1) * np.arange(m.shape[0])).sum() == pytest.approx(1.65, abs=1e-4)


def test_shared_component_cannot_produce_negative_dependence():
    """The structural limit that matters for reading the result."""
    assert D.implied_covariance(D.bivariate_poisson(1.65, 1.15, -0.05)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("r", [0.05, 0.2, -0.3, 0.5])
def test_gaussian_copula_cdf_is_exact(r):
    """24-node Gauss-Legendre on the correlation path, not a latent grid."""
    from scipy.stats import multivariate_normal as mvn
    pts = np.array([[-1.2, 0.4], [0.0, 0.0], [2.0, -1.5], [1.0, 1.0]])
    mine = D._bvn_cdf(pts[:, 0], pts[:, 1], r)
    ref = np.array([mvn.cdf(p, mean=[0, 0], cov=[[1, r], [r, 1]]) for p in pts])
    assert np.abs(mine - ref).max() < 1e-10


@pytest.mark.parametrize("family,param", list(PARAMS.items()))
def test_dependence_is_monotone_in_the_parameter(family, param):
    """A family whose parameter did not move the covariance would prove nothing."""
    lo = D.implied_covariance(D.joint_matrix(1.65, 1.15, family, 0.0, temper=1.05))
    hi = D.implied_covariance(D.joint_matrix(1.65, 1.15, family, param, temper=1.05))
    assert abs(hi - lo) > 1e-4
    # Dixon-Coles is the only family whose sign convention is inverted.
    sign = -1.0 if family == "dixon_coles" else 1.0
    assert np.sign(hi - lo) == np.sign(sign * param)


def test_ipf_preserves_every_odds_ratio():
    """Row/column scaling is what makes the comparison controlled."""
    rng = np.random.default_rng(3)
    m = rng.uniform(0.01, 1.0, (6, 6))
    m /= m.sum()
    ph = D.marginal_pmf(1.7, n=5, temper=1.05)
    pa = D.marginal_pmf(1.1, n=5, temper=1.05)
    out = D.ipf(m, ph, pa)
    assert np.abs(out.sum(axis=1) - ph).max() < 1e-10
    assert np.abs(out.sum(axis=0) - pa).max() < 1e-10
    def log_odds_ratios(x):
        """log( x[i,j] x[k,l] / (x[i,l] x[k,j]) ) over axes (i, j, k, l)."""
        return np.log(x[:, :, None, None] * x[None, None, :, :]
                      / (x[:, None, None, :] * x.T[None, :, :, None]))

    assert np.abs(log_odds_ratios(m) - log_odds_ratios(out)).max() < 1e-9


# ── the experiment harness itself ─────────────────────────────────────────────

@pytest.mark.parametrize("family,param", [("independent", 0.0)] + list(PARAMS.items()))
def test_vectorised_path_equals_the_reference(experiment, family, param):
    lh = np.array([c[0] for c in CASES])
    la = np.array([c[1] for c in CASES])
    vec = experiment.joint(lh, la, family, param, temper=1.05)
    ref = np.stack([D.joint_matrix(l1, l2, family, param, temper=1.05)
                    for l1, l2 in CASES])
    assert np.abs(vec - ref).max() < 1e-11


def test_reweighting_keeps_the_published_1x2_exactly(experiment):
    """The invariant the repo already guarantees, for every candidate family."""
    lh = np.array([c[0] for c in CASES])
    la = np.array([c[1] for c in CASES])
    trio = np.array([[0.80, 0.13, 0.07], [0.37, 0.27, 0.36],
                     [0.19, 0.24, 0.57], [0.50, 0.26, 0.24]])
    for family, param in [("independent", 0.0)] + list(PARAMS.items()):
        m = experiment.reweight_to_trio(
            experiment.joint(lh, la, family, param, temper=1.05), trio)
        assert np.abs(experiment.trio_of(m) - trio).max() < 1e-12
        assert np.abs(m.sum(axis=(1, 2)) - 1.0).max() < 1e-12
        paths = experiment.htft_paths(lh, la, family, param, trio)
        ft = np.stack([paths[:, [experiment.PATHS.index(f"{x}/{y}") for x in "1X2"]].sum(1)
                       for y in "1X2"], axis=1)
        assert np.abs(ft - trio).max() < 1e-12


def test_htft_paths_reproduce_the_deployed_half_time_model(experiment):
    """At independence the vectorised HT/FT must equal props.half_time."""
    from mundialytics.props.half_time import HalfTimeModel
    htm = HalfTimeModel()
    lh = np.array([c[0] for c in CASES])
    la = np.array([c[1] for c in CASES])
    mine = experiment.htft_paths(lh, la, "independent", 0.0)
    for i, (l1, l2) in enumerate(CASES):
        ref = htm.predict_ht_ft(l1, l2)
        for key in experiment.PATHS:
            assert mine[i, experiment.PATHS.index(key)] == pytest.approx(ref[key], abs=1e-10)
