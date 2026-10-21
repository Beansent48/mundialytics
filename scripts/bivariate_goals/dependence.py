from __future__ import annotations

"""Joint goal distributions with a real dependence parameter, as library code.

Five ways to turn two marginal goal pmfs into a joint scoreline matrix:

    independent     the textbook product-Poisson
    dixon_coles     the deployed patch: four low-score cells rescaled
    bivariate_poisson   Karlis-Ntzoufras shared component H=X1+X3, A=X2+X3
    frank_copula    Frank copula on the two discrete marginals
    gaussian_copula Gaussian copula on the two discrete marginals

Every builder returns an (n+1, n+1) matrix that sums to 1. The three real
dependence families take one parameter that is zero at independence, so the
whole file degenerates to `np.outer(ph, pa)` when the parameter is 0 -- which is
asserted in tests/test_bivariate_goals.py.

NOTHING HERE IS IMPORTED BY THE ENGINE. The deployed scoreline matrix is
`statistical_core.distributions.scoreline_distribution` and is untouched; this
module exists so scripts/experiment_bivariate_goals.py can score alternatives
against it (user rule: experiments live in new files, see memory
feedback_protect_deployed_baseline).

Two implementation notes that matter for reading the numbers:

* **Marginals are held fixed across families.** The copulas preserve their
  marginals exactly by construction; the shared component and Dixon-Coles do
  not, so both are iteratively proportional-fitted (`ipf`) back onto the same
  target marginals. Without that step a family would be scored partly on a
  marginal change, and the deployed chain already has a validated marginal
  correction (`goal_temper`).
* **The Gaussian copula CDF is exact**, not quadrature over a latent grid:
  Phi2(x,y;r) = Phi(x)Phi(y) + the standard integral of the bivariate density
  over the correlation path, evaluated with 24-node Gauss-Legendre in r. Error
  is ~1e-12 for |r| <= 0.5, so the measured deltas are not integration noise.
"""

import numpy as np
from scipy.special import comb, factorial
from scipy.stats import norm, poisson

__all__ = [
    "marginal_pmf", "independent", "dixon_coles", "bivariate_poisson",
    "frank_copula", "gaussian_copula", "ipf", "joint_matrix",
    "FAMILIES", "implied_covariance",
]


# ── marginals ─────────────────────────────────────────────────────────────────

def marginal_pmf(lam: float, n: int = 10, temper: float = 1.0) -> np.ndarray:
    """Goal pmf over 0..n, normalized -- identical to the deployed marginal.

    `scoreline_distribution` tempers each side's Poisson pmf (p_k ∝ pois_k^t,
    renormalized) and normalizes the finite grid at the end; doing the same here
    means the independent family reproduces the deployed matrix exactly at
    rho=0.
    """
    p = poisson.pmf(np.arange(n + 1), max(float(lam), 1e-9))
    t = float(temper or 1.0)
    if abs(t - 1.0) > 1e-9:
        p = p ** t
    s = p.sum()
    return p / s if s > 0 else np.full(n + 1, 1.0 / (n + 1))


# ── families ──────────────────────────────────────────────────────────────────

def independent(ph: np.ndarray, pa: np.ndarray, _param: float = 0.0) -> np.ndarray:
    return np.outer(ph, pa)


def dixon_coles(ph: np.ndarray, pa: np.ndarray, rho: float,
                lh: float = 0.0, la: float = 0.0) -> np.ndarray:
    """The deployed patch, re-expressed as a family (tau on four cells)."""
    m = np.outer(ph, pa)
    r = float(rho or 0.0)
    if abs(r) < 1e-12:
        return m
    tau = np.array([[1.0 - r * lh * la, 1.0 + r * lh],
                    [1.0 + r * la, 1.0 - r]])
    m[:2, :2] *= np.clip(tau, 0.05, 3.0)
    s = m.sum()
    return m / s if s > 0 else m


def bivariate_poisson(lh: float, la: float, lam3: float, n: int = 10) -> np.ndarray:
    """Karlis-Ntzoufras bivariate Poisson: H = X1 + X3, A = X2 + X3.

    The shared component X3 ~ Poisson(lam3) is the goals-correlation device:
    Cov(H, A) = lam3 exactly, and the marginals stay Poisson(lh), Poisson(la)
    because lam1 = lh - lam3 and lam2 = la - lam3. Note the structural limit --
    **this family can only produce positive covariance**, which is the right
    shape only if the measured residual covariance is positive (it is, just
    barely: see the README).
    """
    lam3 = float(np.clip(lam3, 0.0, 0.95 * min(lh, la)))
    l1, l2 = max(lh - lam3, 1e-9), max(la - lam3, 1e-9)
    g = np.arange(n + 1)
    # l^k / k! without overflowing through l**k for large k
    with np.errstate(divide="ignore"):
        t1 = np.exp(g * np.log(l1) - np.log(factorial(g)))
        t2 = np.exp(g * np.log(l2) - np.log(factorial(g)))
    m = np.exp(-(l1 + l2 + lam3)) * np.outer(t1, t2)
    if lam3 > 0:
        r = lam3 / (l1 * l2)
        H, A = np.meshgrid(g, g, indexing="ij")
        s = np.ones_like(m)
        for k in range(1, n + 1):
            s += comb(H, k) * comb(A, k) * factorial(k) * r ** k
        m = m * s
    total = m.sum()
    return m / total if total > 0 else m


def _frank_cdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Frank copula C(u, v; theta); theta > 0 = positive dependence."""
    if abs(theta) < 1e-9:
        return u * v
    num = np.expm1(-theta * u) * np.expm1(-theta * v)
    return -np.log1p(num / np.expm1(-theta)) / theta


def _copula_pmf(ph: np.ndarray, pa: np.ndarray, cdf) -> np.ndarray:
    """Rectangle differences of a copula CDF over the marginals' CDF knots.

    P(H=h, A=a) = C(U_h, V_a) - C(U_{h-1}, V_a) - C(U_h, V_{a-1})
                  + C(U_{h-1}, V_{a-1}), with U_{-1} = 0 and U_n = 1. This
    reproduces the marginals exactly, so no refitting is needed afterwards.
    """
    u = np.concatenate(([0.0], np.clip(np.cumsum(ph), 0.0, 1.0)))
    v = np.concatenate(([0.0], np.clip(np.cumsum(pa), 0.0, 1.0)))
    u[-1] = v[-1] = 1.0
    C = cdf(u[:, None] * np.ones_like(v)[None, :], np.ones_like(u)[:, None] * v[None, :])
    m = C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1]
    m = np.clip(m, 0.0, None)
    total = m.sum()
    return m / total if total > 0 else m


def frank_copula(ph: np.ndarray, pa: np.ndarray, theta: float) -> np.ndarray:
    return _copula_pmf(ph, pa, lambda u, v: _frank_cdf(u, v, float(theta)))


# 24-node Gauss-Legendre on [-1, 1], reused for every Phi2 call.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)


def _bvn_cdf(x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    """Bivariate standard normal CDF, exact to ~1e-12 for |r| <= 0.5.

    Phi2(x, y; r) = Phi(x)Phi(y) + (1/2pi) * int_0^r
        exp(-(x^2 - 2 t x y + y^2) / (2(1 - t^2))) / sqrt(1 - t^2) dt
    """
    out = norm.cdf(x) * norm.cdf(y)
    r = float(r)
    if abs(r) < 1e-12:
        return out
    t = 0.5 * r * (_GL_X + 1.0)                      # nodes on [0, r]
    w = 0.5 * r * _GL_W
    s = 1.0 - t ** 2
    xb, yb = x[..., None], y[..., None]
    dens = np.exp(-(xb ** 2 - 2 * t * xb * yb + yb ** 2) / (2 * s)) / np.sqrt(s)
    return out + (dens * w).sum(axis=-1) / (2 * np.pi)


def gaussian_copula(ph: np.ndarray, pa: np.ndarray, rho: float) -> np.ndarray:
    """Gaussian copula: the latent-normal correlation is the one parameter."""
    r = float(np.clip(rho, -0.95, 0.95))
    if abs(r) < 1e-12:
        return np.outer(ph, pa)
    zu = np.clip(norm.ppf(np.clip(np.concatenate(([0.0], np.cumsum(ph))), 1e-12, 1 - 1e-12)), -8.0, 8.0)
    zv = np.clip(norm.ppf(np.clip(np.concatenate(([0.0], np.cumsum(pa))), 1e-12, 1 - 1e-12)), -8.0, 8.0)
    X = zu[:, None] * np.ones_like(zv)[None, :]
    Y = np.ones_like(zu)[:, None] * zv[None, :]
    C = _bvn_cdf(X, Y, r)
    m = C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1]
    m = np.clip(m, 0.0, None)
    total = m.sum()
    return m / total if total > 0 else m


# ── holding the marginals fixed ───────────────────────────────────────────────

def ipf(m: np.ndarray, ph: np.ndarray, pa: np.ndarray,
        iters: int = 60, tol: float = 1e-12) -> np.ndarray:
    """Iterative proportional fitting of `m` onto target marginals.

    Row/column scaling leaves every odds ratio m[i,j]m[k,l]/(m[i,l]m[k,j])
    unchanged, so this moves the marginals without touching the dependence
    structure -- exactly what a controlled comparison needs.
    """
    out = np.array(m, dtype=float)
    for _ in range(iters):
        rs = out.sum(axis=1)
        out *= np.where(rs > 0, ph / np.where(rs > 0, rs, 1.0), 1.0)[:, None]
        cs = out.sum(axis=0)
        out *= np.where(cs > 0, pa / np.where(cs > 0, cs, 1.0), 1.0)[None, :]
        if np.abs(out.sum(axis=1) - ph).max() < tol:
            break
    total = out.sum()
    return out / total if total > 0 else out


# ── one entry point ───────────────────────────────────────────────────────────

FAMILIES = ("independent", "dixon_coles", "bivariate_poisson",
            "frank_copula", "gaussian_copula")


def joint_matrix(lh: float, la: float, family: str, param: float = 0.0, *,
                 n: int = 10, temper: float = 1.0, fix_marginals: bool = True) -> np.ndarray:
    """Joint scoreline matrix for one family and one dependence parameter.

    `fix_marginals` IPFs the result back onto `marginal_pmf(lh)`/`(la)`, so
    every family is scored on the same marginals and only the dependence
    differs. The copulas are already exact there, so it is a no-op for them.
    """
    ph = marginal_pmf(lh, n=n, temper=temper)
    pa = marginal_pmf(la, n=n, temper=temper)
    if family == "independent":
        return np.outer(ph, pa)
    if family == "dixon_coles":
        m = dixon_coles(ph, pa, param, lh=lh, la=la)
    elif family == "bivariate_poisson":
        m = bivariate_poisson(lh, la, param, n=n)
    elif family == "frank_copula":
        m = frank_copula(ph, pa, param)
    elif family == "gaussian_copula":
        m = gaussian_copula(ph, pa, param)
    else:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    return ipf(m, ph, pa) if fix_marginals else m


def implied_covariance(m: np.ndarray) -> float:
    """Cov(H, A) under a scoreline matrix -- the quantity the families move."""
    g = np.arange(m.shape[0], dtype=float)
    eh = float((m.sum(axis=1) * g).sum())
    ea = float((m.sum(axis=0) * np.arange(m.shape[1], dtype=float)).sum())
    return float((m * np.outer(g, np.arange(m.shape[1], dtype=float))).sum() - eh * ea)
