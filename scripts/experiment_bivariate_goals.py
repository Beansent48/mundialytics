from __future__ import annotations

"""Does modelling the real goals correlation beat the Dixon-Coles patch?

THE OPEN ITEM. The README's own "Known limitations" says it: goals are two
independent Poissons with Dixon-Coles rescaling four low-score cells, "a patch,
not a correlation model; a true bivariate Poisson or a copula would be the next
step". This measures whether that next step is worth taking. The 1X2 is not
where it would pay -- dependence largely cancels when you aggregate to three
outcomes -- so the markets that matter here are the ones that read the matrix:
exact score, BTTS, over/under and half-time/full-time.

WHAT IS MEASURED, IN ORDER.
  corr       the empirical dependence, before any model. Marginal correlation,
             then the only quantity a model could use: the covariance of the
             goals CONDITIONAL on the lambdas actually served.
  fit        maximum-likelihood dependence parameter per family, per fitting
             window (expanding, leave-one-season-out, and the test season
             itself as an oracle ceiling).
  validate   out-of-sample by season, per market, for every family.
  sweep      every market as a function of the dependence parameter -- the
             decisive form of a negative result: not "our estimate did not
             help" but "no value of the parameter helps".
  coherence  the invariant the repo already guarantees: the reweighted matrix
             must still sum to exactly the published 1X2.

THE ENGINE IS UNTOUCHED. Nothing here is imported by src/; the families live in
scripts/bivariate_goals/dependence.py and are scored against
`deployed_markets_from_lambdas` as the baseline (user rule: experiments in new
files, deploy only after temporal out-of-sample validation -- memory
feedback_protect_deployed_baseline).

RESULT (2026-10-18): NEGATIVE, and the reason is interesting -- see
scripts/bivariate_goals/README.md. There is almost no conditional dependence to
model (+0.0138 covariance, t = 1.17), and the deployed chain already implies
+0.0152 of it, so the patch is not leaving anything on the table.

    python scripts/experiment_bivariate_goals.py all          # about 20 minutes
    python scripts/experiment_bivariate_goals.py corr --data-root PATH
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import comb, factorial, gammaln
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/bivariate_goals"))

import dependence as D  # noqa: E402
from mundialytics.props import half_time as HT  # noqa: E402
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS as K, markets_from_lambdas)

WF_REL = "data/processed/enriched/understat_xg/walkforward_preds_deployed.csv"
FOUND_REL = "data/processed/foundation_big5_multi_season.csv"
N_GOALS = 10          # the deployed grid is 0..10 per side
N_HALF = HT.MAX_GOALS  # the deployed half-time grid is 0..8
EPS = 1e-12

# Dependence-parameter grids. Deliberately wide: the fitted values all land
# near zero, so the sweep has to cover an order of magnitude more dependence
# than the data asks for before "no parameter helps" means anything.
GRIDS: dict[str, np.ndarray] = {
    # Karlis-Ntzoufras shared component. Cov(H, A) = lam3 exactly, so the grid
    # IS a covariance grid. Structurally one-sided: lam3 >= 0.
    "bivariate_poisson": np.array([0.0, 0.005, 0.0138, 0.025, 0.05, 0.10, 0.20]),
    # Frank copula; theta > 0 = positive dependence.
    "frank_copula": np.array([-1.0, -0.5, -0.2, -0.07, 0.0, 0.07, 0.2, 0.5, 1.0]),
    # Gaussian copula; the latent-normal correlation.
    "gaussian_copula": np.array([-0.15, -0.08, -0.03, -0.011, 0.0, 0.011, 0.03, 0.08, 0.15]),
    # The deployed patch itself, swept for reference. -0.06 is deployed.
    "dixon_coles": np.array([-0.20, -0.12, -0.06, -0.03, 0.0, 0.03, 0.06]),
}
CANDIDATES = ("bivariate_poisson", "frank_copula", "gaussian_copula")
BOUNDS = {"bivariate_poisson": (0.0, 0.40), "frank_copula": (-2.0, 2.0),
          "gaussian_copula": (-0.45, 0.45), "dixon_coles": (-0.30, 0.15)}


# ── vectorised families (the reference implementations are in dependence.py) ──

def v_marginal(lam: np.ndarray, n: int, temper: float = 1.0,
               fold_tail: bool = False) -> np.ndarray:
    """(N, n+1) goal pmfs. `fold_tail` reproduces props.half_time._pmf."""
    k = np.arange(n + 1, dtype=float)
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-9)[:, None]
    p = np.exp(-lam + k * np.log(lam) - gammaln(k + 1.0))
    if fold_tail:
        p[:, -1] += np.maximum(0.0, 1.0 - p.sum(axis=1))
    t = float(temper or 1.0)
    if abs(t - 1.0) > 1e-9:
        p = p ** t
    return p / p.sum(axis=1, keepdims=True)


def _norm(m: np.ndarray) -> np.ndarray:
    return m / np.clip(m.sum(axis=(1, 2), keepdims=True), EPS, None)


def v_independent(ph: np.ndarray, pa: np.ndarray, _p: float = 0.0) -> np.ndarray:
    return ph[:, :, None] * pa[:, None, :]


def v_dixon_coles(ph, pa, rho, lh, la) -> np.ndarray:
    m = v_independent(ph, pa)
    r = float(rho or 0.0)
    if abs(r) < 1e-12:
        return m
    m = m.copy()
    tau = np.empty((len(ph), 2, 2))
    tau[:, 0, 0] = 1.0 - r * lh * la
    tau[:, 0, 1] = 1.0 + r * lh
    tau[:, 1, 0] = 1.0 + r * la
    tau[:, 1, 1] = 1.0 - r
    m[:, :2, :2] *= np.clip(tau, 0.05, 3.0)
    return _norm(m)


_BP_COEF: dict[int, list[np.ndarray]] = {}


def _bp_coef(n: int) -> list[np.ndarray]:
    """comb(h,k)*comb(a,k)*k! on the (n+1, n+1) grid, for k = 1..n."""
    if n not in _BP_COEF:
        h, a = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        _BP_COEF[n] = [comb(h, k) * comb(a, k) * factorial(k) for k in range(1, n + 1)]
    return _BP_COEF[n]


def v_bivariate_poisson(lh: np.ndarray, la: np.ndarray, lam3: float, n: int) -> np.ndarray:
    l3 = np.clip(float(lam3), 0.0, None) * np.ones(len(lh))
    l3 = np.minimum(l3, 0.95 * np.minimum(lh, la))
    l1, l2 = np.maximum(lh - l3, 1e-9), np.maximum(la - l3, 1e-9)
    k = np.arange(n + 1, dtype=float)
    t1 = np.exp(k * np.log(l1)[:, None] - gammaln(k + 1.0))
    t2 = np.exp(k * np.log(l2)[:, None] - gammaln(k + 1.0))
    m = np.exp(-(l1 + l2 + l3))[:, None, None] * t1[:, :, None] * t2[:, None, :]
    if np.any(l3 > 0):
        r = (l3 / (l1 * l2))[:, None, None]
        s = np.ones_like(m)
        for k_i, c in enumerate(_bp_coef(n), start=1):
            s = s + c[None] * r ** k_i
        m = m * s
    return _norm(m)


def _cdf_knots(p: np.ndarray) -> np.ndarray:
    u = np.concatenate([np.zeros((len(p), 1)), np.cumsum(p, axis=1)], axis=1)
    u[:, -1] = 1.0
    return np.clip(u, 0.0, 1.0)


def _rect(C: np.ndarray) -> np.ndarray:
    m = C[:, 1:, 1:] - C[:, :-1, 1:] - C[:, 1:, :-1] + C[:, :-1, :-1]
    return _norm(np.clip(m, 0.0, None))


def v_frank_copula(ph: np.ndarray, pa: np.ndarray, theta: float) -> np.ndarray:
    th = float(theta)
    if abs(th) < 1e-9:
        return v_independent(ph, pa)
    u, v = _cdf_knots(ph)[:, :, None], _cdf_knots(pa)[:, None, :]
    C = -np.log1p(np.expm1(-th * u) * np.expm1(-th * v) / np.expm1(-th)) / th
    return _rect(C)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)


def _v_bvn_cdf(x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    """Exact bivariate normal CDF, broadcast over leading axes."""
    out = norm.cdf(x) * norm.cdf(y)
    t = 0.5 * r * (_GL_X + 1.0)
    w = 0.5 * r * _GL_W
    s = 1.0 - t ** 2
    xb, yb = x[..., None], y[..., None]
    dens = np.exp(-(xb ** 2 - 2 * t * xb * yb + yb ** 2) / (2 * s)) / np.sqrt(s)
    return out + (dens * w).sum(axis=-1) / (2 * np.pi)


def v_gaussian_copula(ph: np.ndarray, pa: np.ndarray, rho: float,
                      chunk: int = 2000) -> np.ndarray:
    r = float(np.clip(rho, -0.95, 0.95))
    if abs(r) < 1e-12:
        return v_independent(ph, pa)
    zu = np.clip(norm.ppf(np.clip(_cdf_knots(ph), 1e-12, 1 - 1e-12)), -8.0, 8.0)
    zv = np.clip(norm.ppf(np.clip(_cdf_knots(pa), 1e-12, 1 - 1e-12)), -8.0, 8.0)
    out = np.empty((len(ph), ph.shape[1], pa.shape[1]))
    for s in range(0, len(ph), chunk):
        e = min(s + chunk, len(ph))
        C = _v_bvn_cdf(zu[s:e, :, None], zv[s:e, None, :], r)
        out[s:e] = _rect(C)
    return out


def v_ipf(m: np.ndarray, ph: np.ndarray, pa: np.ndarray, iters: int = 40) -> np.ndarray:
    """Row/column scaling onto target marginals; odds ratios are preserved."""
    out = m.copy()
    for _ in range(iters):
        rs = out.sum(axis=2)
        out *= np.where(rs > EPS, ph / np.where(rs > EPS, rs, 1.0), 1.0)[:, :, None]
        cs = out.sum(axis=1)
        out *= np.where(cs > EPS, pa / np.where(cs > EPS, cs, 1.0), 1.0)[:, None, :]
        if np.abs(out.sum(axis=2) - ph).max() < 1e-13:
            break
    return _norm(out)


def joint(lh: np.ndarray, la: np.ndarray, family: str, param: float, *,
          n: int = N_GOALS, temper: float = 1.0, fix_marginals: bool = True,
          fold_tail: bool = False) -> np.ndarray:
    """(N, n+1, n+1) joint scoreline matrices, one family, one parameter."""
    ph = v_marginal(lh, n, temper, fold_tail)
    pa = v_marginal(la, n, temper, fold_tail)
    if family == "independent":
        return v_independent(ph, pa)
    if family == "dixon_coles":
        m = v_dixon_coles(ph, pa, param, lh, la)
    elif family == "bivariate_poisson":
        m = v_bivariate_poisson(lh, la, param, n)
    elif family == "frank_copula":
        m = v_frank_copula(ph, pa, param)
    elif family == "gaussian_copula":
        m = v_gaussian_copula(ph, pa, param)
    else:
        raise ValueError(f"unknown family {family!r}")
    return v_ipf(m, ph, pa) if fix_marginals else m


# ── markets off a batch of matrices ───────────────────────────────────────────

def _masks(n: int):
    i, j = np.indices((n + 1, n + 1))
    return {"home": i > j, "draw": i == j, "away": i < j,
            "o25": (i + j) >= 3, "btts": (i > 0) & (j > 0)}


def trio_of(m: np.ndarray) -> np.ndarray:
    mk = _masks(m.shape[1] - 1)
    return np.stack([m[:, mk["home"]].sum(1), m[:, mk["draw"]].sum(1),
                     m[:, mk["away"]].sum(1)], axis=1)


def sharpen(trio: np.ndarray, gamma: float) -> np.ndarray:
    t = np.clip(trio, 1e-9, 1.0) ** float(gamma)
    return t / t.sum(axis=1, keepdims=True)


def reweight_to_trio(m: np.ndarray, trio: np.ndarray) -> np.ndarray:
    """Scale each 1X2 region so the matrix sums to the published trio.

    This is `markets_from_lambdas(coherent=True)`, applied to any matrix. It is
    what makes the comparison fair: the published 1X2 is held fixed, so a
    family can only win or lose on the goal markets.
    """
    out = m.copy()
    mk = _masks(m.shape[1] - 1)
    for col, key in enumerate(("home", "draw", "away")):
        mass = out[:, mk[key]].sum(1)
        scale = np.where(mass > EPS, trio[:, col] / np.where(mass > EPS, mass, 1.0), 0.0)
        out[:, mk[key]] *= scale[:, None]
    return _norm(out)


def goal_markets(m: np.ndarray, hg: np.ndarray, ag: np.ndarray) -> dict[str, np.ndarray]:
    n = m.shape[1] - 1
    mk = _masks(n)
    h, a = np.minimum(hg, n), np.minimum(ag, n)
    return {"o25": m[:, mk["o25"]].sum(1), "btts": m[:, mk["btts"]].sum(1),
            "cs": m[np.arange(len(m)), h, a]}


# ── half-time / full-time, dependence in both halves ──────────────────────────

_ADJ = (("1", HT.H2_ADJ_HOME_LEAD), ("X", HT.H2_ADJ_LEVEL), ("2", HT.H2_ADJ_AWAY_LEAD))
PATHS = [f"{x}/{y}" for x in "1X2" for y in "1X2"]


def _margin_pmf(m: np.ndarray) -> np.ndarray:
    """(N, 2n+1) pmf of (home - away) goals, index 0 = -n."""
    n = m.shape[1] - 1
    i, j = np.indices((n + 1, n + 1))
    d = (i - j + n).ravel()
    flat = m.reshape(len(m), -1)
    out = np.zeros((len(m), 2 * n + 1))
    np.add.at(out.T, d, flat.T)
    return out


def _half_param(family: str, param: float, scale: float) -> float:
    """The dependence parameter for a half.

    A copula parameter is a property of the dependence structure, not of the
    scale, so it carries over unchanged. The shared component is a goal rate,
    so it scales with the half's goal volume -- geometric mean of the two
    sides' scalings, which keeps Cov(H, A) proportional to the goals on offer.
    """
    return param * scale if family == "bivariate_poisson" else param


def htft_paths(lh: np.ndarray, la: np.ndarray, family: str, param: float,
               trio: np.ndarray | None = None) -> np.ndarray:
    """(N, 9) HT/FT path probabilities, column order = PATHS.

    Reproduces props.half_time.predict_ht_ft exactly at family="independent"
    (same shares, same second-half game-state adjustment, same optional
    rescaling to a full-time trio), and injects the dependence into both the
    half-time matrix and each second-half matrix.
    """
    sh, sa = HT.HT_SHARE_HOME, HT.HT_SHARE_AWAY
    n = N_HALF
    m1 = joint(lh * sh, la * sa, family, _half_param(family, param, np.sqrt(sh * sa)),
               n=n, fold_tail=True)
    p1 = _margin_pmf(m1)                      # (N, 2n+1)
    l2h, l2a = lh * (1.0 - sh), la * (1.0 - sa)
    scale2 = np.sqrt((1.0 - sh) * (1.0 - sa))
    d = np.arange(-n, n + 1)
    out = np.zeros((len(lh), 9))
    for ht_res, adj in _ADJ:
        sel = {"1": d > 0, "X": d == 0, "2": d < 0}[ht_res]
        m2 = joint(l2h * adj, l2a * adj, family,
                   _half_param(family, param, scale2 * adj), n=n, fold_tail=True)
        p2 = _margin_pmf(m2)
        c2 = np.cumsum(p2, axis=1)
        for d1 in d[sel]:
            k = n - d1                        # index of d2 == -d1
            p_eq = p2[:, k]
            p_lt = c2[:, k] - p_eq
            p_gt = 1.0 - c2[:, k]
            w = p1[:, d1 + n]
            for ft_res, pr in (("1", p_gt), ("X", p_eq), ("2", p_lt)):
                out[:, PATHS.index(f"{ht_res}/{ft_res}")] += w * pr
    if trio is not None:
        ft = np.stack([out[:, [PATHS.index(f"{x}/{y}") for x in "1X2"]].sum(1)
                       for y in "1X2"], axis=1)
        for c, y in enumerate("1X2"):
            cols = [PATHS.index(f"{x}/{y}") for x in "1X2"]
            out[:, cols] *= np.where(ft[:, c] > EPS,
                                     trio[:, c] / np.where(ft[:, c] > EPS, ft[:, c], 1.0),
                                     0.0)[:, None]
    return out / np.clip(out.sum(axis=1, keepdims=True), EPS, None)


# ── scores ────────────────────────────────────────────────────────────────────

def rps3(y: np.ndarray, P: np.ndarray) -> float:
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return float(((cp - cy) ** 2)[:, :2].sum(axis=1).mean() / 2)


def mll(y: np.ndarray, P: np.ndarray) -> float:
    """Multiclass log loss from the probability of the realised class."""
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, None)).mean())


def bll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def nll_cell(p: np.ndarray) -> float:
    return float(-np.log(np.clip(p, 1e-9, None)).mean())


# ── data ──────────────────────────────────────────────────────────────────────

def load(data_root: Path) -> pd.DataFrame:
    wf = pd.read_csv(data_root / WF_REL)
    need = {"season", "match_id", "hg", "ag", "ph", "pd", "pa", "lh", "la"}
    missing = need - set(wf.columns)
    if missing:
        raise SystemExit(f"{WF_REL} is missing {sorted(missing)}; regenerate it with "
                         "scripts/generate_deployed_walkforward.py")
    ht = pd.read_csv(data_root / FOUND_REL, low_memory=False,
                     usecols=["match_id", "home_goals_ht", "away_goals_ht", "competition"])
    wf = wf.merge(ht, on="match_id", how="left")
    wf["outcome"] = np.where(wf.hg > wf.ag, 0, np.where(wf.hg == wf.ag, 1, 2))
    # a plain 0..N-1 index, so `group.index.to_numpy()` is a positional selector
    return wf.reset_index(drop=True)


def load_foundation(data_root: Path) -> pd.DataFrame:
    f = pd.read_csv(data_root / FOUND_REL, low_memory=False,
                    usecols=["season", "competition", "home_goals", "away_goals",
                             "home_goals_ht", "away_goals_ht"])
    return f.dropna(subset=["home_goals", "away_goals"])


# ── 1. the empirical dependence ───────────────────────────────────────────────

def cells_oe(df: pd.DataFrame, which: str, param: float, top: int = 4) -> None:
    """Observed / expected count per low scoreline, over the whole set.

    This is the diagnostic that decides between the two stories a +0.014
    covariance could be telling. A genuine goals correlation inflates the whole
    diagonal in proportion to the goals on offer; a draw/low-score effect sits
    in the 0-0 / 1-1 corner, where Dixon-Coles already works.
    """
    lh, la = df.lh.to_numpy(float), df.la.to_numpy(float)
    if which == "deployed":
        m = reweight_to_trio(
            joint(lh, la, "dixon_coles", param, temper=K["goal_temper"], fix_marginals=False),
            df[["ph", "pd", "pa"]].to_numpy(float))
    else:
        m = joint(lh, la, which, param, temper=K["goal_temper"])
    exp = m.sum(axis=0)
    obs = np.zeros_like(exp)
    h = np.minimum(df.hg.to_numpy(int), N_GOALS)
    a = np.minimum(df.ag.to_numpy(int), N_GOALS)
    np.add.at(obs, (h, a), 1.0)
    print("        " + "".join(f"{j:>8d}" for j in range(top)) + "    away goals")
    for i in range(top):
        row = "".join(f"{obs[i, j] / exp[i, j]:>8.3f}" if exp[i, j] > 0 else f"{'-':>8s}"
                      for j in range(top))
        print(f"    {i}   {row}")
    i, j = np.indices(exp.shape)
    for label, mask in (("all draws", i == j), ("0-0 and 1-1", (i == j) & (i <= 1)),
                        ("draws 2-2 and up", (i == j) & (i >= 2))):
        print(f"    {label:18s} observed {obs[mask].sum():6.0f}  expected "
              f"{exp[mask].sum():8.1f}   O/E {obs[mask].sum() / exp[mask].sum():.4f}")


def cmd_corr(df: pd.DataFrame, data_root: Path) -> None:
    print("=" * 78)
    print("1. HOW MUCH DEPENDENCE IS THERE, BEFORE ANY MODEL")
    print("=" * 78)
    f = load_foundation(data_root)
    h0, a0 = f.home_goals.to_numpy(float), f.away_goals.to_numpy(float)
    print(f"\nfoundation, {len(f):,} matches, {f.season.nunique()} seasons")
    print(f"  marginal Pearson corr(home, away) = {np.corrcoef(h0, a0)[0, 1]:+.4f}")
    print(f"  marginal Spearman                 = {spearmanr(h0, a0).statistic:+.4f}")
    print(f"  marginal Cov(home, away)          = {np.cov(h0, a0)[0, 1]:+.4f}")

    h, a = df.hg.to_numpy(float), df.ag.to_numpy(float)
    lh, la = df.lh.to_numpy(float), df.la.to_numpy(float)
    rh, ra = h - lh, a - la
    cov_lam = np.cov(lh, la)[0, 1]
    cov_res = np.cov(rh, ra)[0, 1]
    print(f"\nwalk-forward set, {len(df):,} matches, {df.season.nunique()} seasons "
          "(lambdas as served, out of sample)")
    print(f"  marginal Cov(home, away)   {np.cov(h, a)[0, 1]:+.4f}")
    print(f"  = Cov(lam_h, lam_a)        {cov_lam:+.4f}   <- mismatch: strong side faces a weak one")
    print(f"  + conditional Cov          {cov_res:+.4f}   <- THE ONLY PART A MODEL CAN USE")
    print(f"  corr(lam_h, lam_a) across matches = {np.corrcoef(lh, la)[0, 1]:+.4f}")

    rng = np.random.default_rng(0)
    bs = np.array([np.cov(rh[i], ra[i])[0, 1]
                   for i in rng.integers(0, len(h), (2000, len(h)))])
    se = float(bs.std())
    print(f"\n  conditional Cov = {cov_res:+.4f}  (bootstrap se {se:.4f}, t = {cov_res / se:+.2f}, "
          f"2000 resamples)")
    print(f"  as a correlation of the residuals: {cov_res / np.sqrt(rh.var() * ra.var()):+.4f}")
    print("\n  by season (the sign does not hold):")
    for s, g in df.groupby("season"):
        c = np.cov(g.hg - g.lh, g.ag - g.la)[0, 1]
        print(f"    {s}  n={len(g):5d}  conditional Cov {c:+.4f}")

    df = df.assign(lt=lh + la, gap=np.log(lh / la), rr=(rh - rh.mean()) * (ra - ra.mean()))
    for col, label in (("lt", "lambda_total"), ("gap", "log(lam_h / lam_a)")):
        print(f"\n  by {label} quintile (no monotone pattern = nothing to condition on):")
        q = pd.qcut(df[col], 5)
        for _, g in df.groupby(q, observed=True):
            print(f"    {label} mean {g[col].mean():+7.3f}  n={len(g):5d}  "
                  f"conditional Cov {g.rr.mean():+.4f}")

    ok = df.home_goals_ht.notna()
    g = df[ok]
    h1, a1 = g.home_goals_ht.to_numpy(float), g.away_goals_ht.to_numpy(float)
    l1h = g.lh.to_numpy(float) * HT.HT_SHARE_HOME
    l1a = g.la.to_numpy(float) * HT.HT_SHARE_AWAY
    print(f"\n  half-time, {len(g):,} matches: conditional Cov "
          f"{np.cov(h1 - l1h, a1 - l1a)[0, 1]:+.4f}")

    print("\n  observed / expected per scoreline, under independence with the deployed")
    print("  marginals -- a global correlation would tilt the whole table along the")
    print("  diagonal; a low-score effect would sit in one corner:")
    cells_oe(df, "independent", 0.0)
    print("\n  and the same table under the deployed matrix (Dixon-Coles + coherent):")
    cells_oe(df, "deployed", K["outcome_rho"])

    print("\n  what the DEPLOYED chain already implies (sampled 1 in 13 matches):")
    idx = np.arange(0, len(df), 13)
    for label, kw in (("power-map matrix (rho -0.06, temper 1.05, coherent) -- what was "
                       "published when this ran; v0.63.0 serves the contrast map",
                       dict(rho=K["outcome_rho"], temper=K["goal_temper"],
                            sharpen_gamma=K["sharpen_gamma_1x2"],
                            sharpen_contrast_only=False, coherent=True)),
                      ("Dixon-Coles + temper only, no 1X2 sharpening",
                       dict(rho=K["outcome_rho"], temper=K["goal_temper"],
                            sharpen_gamma=1.0, coherent=False)),
                      ("pure independent Poisson", dict(rho=0.0, temper=1.0,
                                                        sharpen_gamma=1.0, coherent=False))):
        covs = [D.implied_covariance(markets_from_lambdas(lh[i], la[i], **kw)[1]
                                     .matrix.to_numpy()) for i in idx]
        print(f"    implied Cov {np.mean(covs):+.4f}   {label}")
    print("\n  The gap between what the data shows and what the deployed matrix already")
    print("  implies is the entire prize. See scripts/bivariate_goals/README.md.")


# ── 2. maximum-likelihood dependence ──────────────────────────────────────────

def fit_param(sub: pd.DataFrame, family: str) -> tuple[float, float]:
    """(argmax param, log-likelihood gain per match vs independence)."""
    lh, la = sub.lh.to_numpy(float), sub.la.to_numpy(float)
    hg = np.minimum(sub.hg.to_numpy(int), N_GOALS)
    ag = np.minimum(sub.ag.to_numpy(int), N_GOALS)
    rows = np.arange(len(sub))

    def nll(p: float) -> float:
        m = joint(lh, la, family, float(p), temper=K["goal_temper"])
        return nll_cell(m[rows, hg, ag])

    base = nll(0.0)
    lo, hi = BOUNDS[family]
    r = minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-4})
    return (float(r.x), base - float(r.fun)) if r.fun < base else (0.0, 0.0)


def fit_windows(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """{family: {season: param}} for each fitting regime, plus diagnostics."""
    seasons = sorted(df.season.unique())
    out: dict[str, dict[str, float]] = {}
    for fam in CANDIDATES + ("dixon_coles",):
        out[fam] = {}
        for i, s in enumerate(seasons):
            prior = df[df.season.isin(seasons[:i])]
            out[fam][f"expanding::{s}"] = fit_param(prior, fam)[0] if i else np.nan
            out[fam][f"loso::{s}"] = fit_param(df[df.season != s], fam)[0]
            p, gain = fit_param(df[df.season == s], fam)
            out[fam][f"oracle::{s}"] = p
            out[fam][f"gain::{s}"] = gain
        out[fam]["full"], out[fam]["full_gain"] = fit_param(df, fam)
    return out


def cmd_fit(df: pd.DataFrame, fits: dict | None = None) -> dict:
    print("\n" + "=" * 78)
    print("2. MAXIMUM-LIKELIHOOD DEPENDENCE PARAMETER")
    print("=" * 78)
    fits = fits or fit_windows(df)
    seasons = sorted(df.season.unique())
    print("\nfitted on everything (all six seasons), exact-score likelihood:")
    for fam in CANDIDATES + ("dixon_coles",):
        p = fits[fam]["full"]
        cov = np.mean([D.implied_covariance(joint(
            df.lh.to_numpy(float)[i:i + 1], df.la.to_numpy(float)[i:i + 1], fam, p,
            temper=K["goal_temper"])[0]) for i in range(0, len(df), 97)])
        print(f"  {fam:19s} param {p:+.4f}   implied Cov {cov:+.4f}   "
              f"log-lik gain {fits[fam]['full_gain']:+.6f} / match")
    print("\nthe RAW trio, before the 1X2 sharpening and before any reweighting --")
    print("the one place the two devices compete head to head, because a draw-rate")
    print("deficit is what both of them exist to fix:")
    y = df.outcome.to_numpy(int)
    obs_draw = float((y == 1).mean())
    print(f"  observed draw rate {obs_draw:.4f} over {len(df):,} matches")
    print(f"  {'family':19s} {'param':>8s} {'P(draw)':>9s} {'O/E draws':>10s} "
          f"{'1X2 RPS':>9s} {'1X2 ll':>9s} {'o25 ll':>9s} {'btts ll':>9s}")
    lh, la = df.lh.to_numpy(float), df.la.to_numpy(float)
    hg, ag = df.hg.to_numpy(int), df.ag.to_numpy(int)
    tot, both = (hg + ag >= 3).astype(float), ((hg > 0) & (ag > 0)).astype(float)
    for fam, p in [("independent", 0.0), ("dixon_coles", K["outcome_rho"]),
                   ("dixon_coles", fits["dixon_coles"]["full"])] + \
                  [(f, fits[f]["full"]) for f in CANDIDATES]:
        m = joint(lh, la, fam, p, temper=K["goal_temper"],
                  fix_marginals=(fam != "dixon_coles"))
        trio = trio_of(m)
        gm = goal_markets(m, hg, ag)
        pdraw = float(trio[:, 1].mean())
        print(f"  {fam:19s} {p:+8.4f} {pdraw:9.4f} {obs_draw / pdraw:10.4f} "
              f"{rps3(y, trio):9.5f} {mll(y, trio):9.5f} "
              f"{bll(tot, gm['o25']):9.5f} {bll(both, gm['btts']):9.5f}")

    print("\nper fitting window (the parameter is not stable, which is what an")
    print("effect of this size looks like):")
    hdr = "  ".join(f"{s[-7:]:>9s}" for s in seasons)
    print(f"  {'family':19s} {'regime':10s} {hdr}")
    for fam in CANDIDATES:
        for regime in ("expanding", "loso", "oracle"):
            vals = "  ".join(
                ("      n/a" if np.isnan(fits[fam][f'{regime}::{s}'])
                 else f"{fits[fam][f'{regime}::{s}']:+9.4f}") for s in seasons)
            print(f"  {fam:19s} {regime:10s} {vals}")
    return fits


# ── 3. out-of-sample validation, per market, per season ───────────────────────

MARKETS = ("1x2_rps", "1x2_ll", "o25", "btts", "cs", "htft")


def score_block(df: pd.DataFrame, m: np.ndarray, paths: np.ndarray | None,
                trio: np.ndarray) -> dict[str, np.ndarray]:
    """Per-match scores for one arm: the pieces, aggregated later by season.

    `trio` is the 1X2 this arm publishes -- the deployed one when it is held
    fixed, the arm's own sharpened one otherwise.
    """
    hg, ag = df.hg.to_numpy(int), df.ag.to_numpy(int)
    gm = goal_markets(m, hg, ag)
    out = {"o25_p": gm["o25"], "btts_p": gm["btts"], "cs_p": gm["cs"],
           "trio": trio}
    if paths is not None:
        ok = df.home_goals_ht.notna().to_numpy()
        h1 = np.where(ok, df.home_goals_ht.fillna(0).to_numpy(int), 0)
        a1 = np.where(ok, df.away_goals_ht.fillna(0).to_numpy(int), 0)
        ht_res = np.where(h1 > a1, "1", np.where(h1 == a1, "X", "2"))
        ft_res = np.where(hg > ag, "1", np.where(hg == ag, "X", "2"))
        col = np.array([PATHS.index(f"{x}/{y}") for x, y in zip(ht_res, ft_res)])
        out["htft_p"] = np.where(ok, paths[np.arange(len(df)), col], np.nan)
    return out


def take(sc: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """Positional subset of a score block (works for the (N,) and (N,3) entries)."""
    return {k: v[idx] for k, v in sc.items()}


def aggregate(df: pd.DataFrame, sc: dict[str, np.ndarray]) -> dict[str, float]:
    hg, ag = df.hg.to_numpy(int), df.ag.to_numpy(int)
    tot = hg + ag
    y = df.outcome.to_numpy(int)
    res = {"1x2_rps": rps3(y, sc["trio"]), "1x2_ll": mll(y, sc["trio"]),
           "o25": bll((tot >= 3).astype(float), sc["o25_p"]),
           "btts": bll(((hg > 0) & (ag > 0)).astype(float), sc["btts_p"]),
           "cs": nll_cell(sc["cs_p"])}
    if "htft_p" in sc:
        p = sc["htft_p"]
        res["htft"] = nll_cell(p[~np.isnan(p)])
    return res


def arm_scores(df: pd.DataFrame, family: str, params: np.ndarray | float,
               *, own_1x2: bool, with_htft: bool = True) -> dict[str, np.ndarray]:
    """Score one arm over `df`. `params` may be per-match (fold-specific fits).

    `own_1x2=False` holds the published 1X2 fixed and reweights the matrix onto
    it, so the goal markets are the only thing that can move -- and the
    coherence invariant holds by construction. `own_1x2=True` instead lets the
    family publish its own sharpened trio, which is how the 1X2 question is
    answered.
    """
    lh, la = df.lh.to_numpy(float), df.la.to_numpy(float)
    trio_pub = df[["ph", "pd", "pa"]].to_numpy(float)
    p_arr = np.atleast_1d(np.asarray(params, dtype=float))
    uniq = np.unique(p_arr)
    m = np.empty((len(df), N_GOALS + 1, N_GOALS + 1))
    paths = np.empty((len(df), 9)) if with_htft else None
    for p in uniq:
        sel = np.ones(len(df), bool) if len(p_arr) == 1 else (p_arr == p)
        m[sel] = joint(lh[sel], la[sel], family, float(p), temper=K["goal_temper"])
        if with_htft:
            trio_h = None if own_1x2 else trio_pub[sel]
            paths[sel] = htft_paths(lh[sel], la[sel], family, float(p), trio_h)
    if own_1x2:
        trio = sharpen(trio_of(m), K["sharpen_gamma_1x2"])
        m = reweight_to_trio(m, trio)
        if with_htft:
            for p in uniq:
                sel = np.ones(len(df), bool) if len(p_arr) == 1 else (p_arr == p)
                paths[sel] = htft_paths(lh[sel], la[sel], family, float(p), trio[sel])
    else:
        trio = trio_pub
        m = reweight_to_trio(m, trio)
    return score_block(df, m, paths, trio)


def deployed_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """The baseline: the matrix and the HT/FT paths exactly as served."""
    lh, la = df.lh.to_numpy(float), df.la.to_numpy(float)
    trio = df[["ph", "pd", "pa"]].to_numpy(float)
    m = joint(lh, la, "dixon_coles", K["outcome_rho"], temper=K["goal_temper"],
              fix_marginals=False)
    m = reweight_to_trio(m, trio)
    paths = htft_paths(lh, la, "independent", 0.0, trio)
    return score_block(df, m, paths, trio)


def verify_against_engine(df: pd.DataFrame, n: int = 200) -> tuple[float, float]:
    """The vectorised baseline must equal the engine's POWER-map output.

    `sharpen_contrast_only=False` is pinned on purpose below. This experiment's
    baseline is the chain as it was served while it ran -- Dixon-Coles plus
    `p^gamma` -- and v0.63.0 has since moved the engine to the contrast map, so
    reading the flag off K here would silently re-point the baseline and change
    what every delta in this file means. A deliberate snapshot, spelled out
    rather than left to a default, because the default is what let the same hole
    open in two other harnesses.
    """
    rng = np.random.default_rng(7)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    sub = df.iloc[idx]
    lh, la = sub.lh.to_numpy(float), sub.la.to_numpy(float)
    mine = reweight_to_trio(
        joint(lh, la, "dixon_coles", K["outcome_rho"], temper=K["goal_temper"],
              fix_marginals=False), sub[["ph", "pd", "pa"]].to_numpy(float))
    err_m = 0.0
    for k, (h, a) in enumerate(zip(lh, la)):
        _, dist = markets_from_lambdas(h, a, rho=K["outcome_rho"], temper=K["goal_temper"],
                                       sharpen_gamma=K["sharpen_gamma_1x2"],
                                       sharpen_contrast_only=False, coherent=True)
        err_m = max(err_m, float(np.abs(mine[k] - dist.matrix.to_numpy()).max()))
    htm = HT.HalfTimeModel()
    mine_p = htft_paths(lh, la, "independent", 0.0, sub[["ph", "pd", "pa"]].to_numpy(float))
    err_p = 0.0
    for k, (h, a) in enumerate(zip(lh, la)):
        ref = htm.predict_ht_ft(h, a, ft_trio=tuple(sub[["ph", "pd", "pa"]].to_numpy(float)[k]))
        err_p = max(err_p, max(abs(mine_p[k, PATHS.index(key)] - ref[key]) for key in PATHS))
    return err_m, err_p


def cmd_validate(df: pd.DataFrame, fits: dict | None = None) -> dict:
    print("\n" + "=" * 78)
    print("3. OUT-OF-SAMPLE BY SEASON, PER MARKET")
    print("=" * 78)
    err_m, err_p = verify_against_engine(df)
    print(f"\nharness check: the baseline arm reproduces the engine's own matrix to "
          f"{err_m:.2e}\n               and its HT/FT paths to {err_p:.2e}")
    fits = fits or fit_windows(df)
    seasons = sorted(df.season.unique())

    base_sc = deployed_scores(df)
    base = {s: aggregate(g, take(base_sc, g.index.to_numpy()))
            for s, g in df.groupby("season")}
    print(f"\nbaseline (deployed: Dixon-Coles -0.06, temper 1.05, coherent), {len(df):,} matches")
    allb = aggregate(df, base_sc)
    print("  " + "  ".join(f"{k} {allb[k]:.5f}" for k in MARKETS))

    results: dict = {"baseline": allb, "arms": {}}
    print(f"\n{'arm':34s} " + "  ".join(f"{k:>10s}" for k in MARKETS))
    print(f"{'':34s} " + "  ".join(f"{'(folds)':>10s}" for _ in MARKETS))
    for fam in ("independent",) + CANDIDATES + ("dixon_coles",):
        regimes = ("fixed",) if fam == "independent" else ("expanding", "loso", "oracle")
        for regime in regimes:
            if regime == "fixed":
                sub, params = df, 0.0
            else:
                pmap = {s: fits[fam][f"{regime}::{s}"] for s in seasons}
                keep = [s for s in seasons if not np.isnan(pmap[s])]
                sub = df[df.season.isin(keep)].reset_index(drop=True)
                params = sub.season.map(pmap).to_numpy(float)
            sc = arm_scores(sub, fam, params, own_1x2=False)
            tot = aggregate(sub, sc)
            # per-season deltas against the baseline on the same matches
            wins = {k: 0 for k in MARKETS}
            n_folds = 0
            for s, g in sub.groupby("season"):
                n_folds += 1
                a = aggregate(g, take(sc, g.index.to_numpy()))
                b = base[s]
                for k in MARKETS:
                    if k in a and k in b and a[k] < b[k]:
                        wins[k] += 1
            mask = df.season.isin(sub.season.unique()).to_numpy()
            bt = aggregate(df[mask], take(base_sc, np.flatnonzero(mask)))
            label = fam if regime == "fixed" else f"{fam} [{regime}]"
            cells = "  ".join(f"{tot[k] - bt[k]:+10.5f}" for k in MARKETS)
            print(f"  {label:32s} {cells}")
            print(f"  {'':32s} " + "  ".join(f"{str(wins[k]) + '/' + str(n_folds):>10s}"
                                             for k in MARKETS))
            results["arms"][label] = {"delta": {k: tot[k] - bt[k] for k in MARKETS},
                                      "wins": wins, "folds": n_folds}
    print("\n(all deltas vs the deployed baseline on the same matches; negative = better.")
    print(" 1x2_rps/1x2_ll are identical by construction here -- the published 1X2 is held")
    print(" fixed so the goal markets are the only thing that can move. The 1X2 question")
    print(" is answered separately below.)")

    print("\n--- and if each family publishes its OWN sharpened 1X2 instead ---")
    print(f"{'arm':34s} " + "  ".join(f"{k:>10s}" for k in MARKETS))
    for fam in ("independent",) + CANDIDATES:
        p = 0.0 if fam == "independent" else fits[fam]["full"]
        sc = arm_scores(df, fam, p, own_1x2=True)
        tot = aggregate(df, sc)
        wins = {k: 0 for k in MARKETS}
        for s, g in df.groupby("season"):
            a = aggregate(g, take(sc, g.index.to_numpy()))
            for k in MARKETS:
                if a[k] < base[s][k]:
                    wins[k] += 1
        lbl = f"{fam} (own 1X2, param {p:+.4f})"
        print(f"  {lbl:32s} " + "  ".join(f"{tot[k] - allb[k]:+10.5f}" for k in MARKETS))
        print(f"  {'':32s} " + "  ".join(f"{str(wins[k]) + '/' + str(len(seasons)):>10s}"
                                         for k in MARKETS))
        results["arms"][lbl] = {"delta": {k: tot[k] - allb[k] for k in MARKETS},
                                "wins": wins, "folds": len(seasons)}
    return results


# ── 4. is there ANY parameter that helps? ─────────────────────────────────────

def cmd_sweep(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("4. EVERY MARKET AS A FUNCTION OF THE DEPENDENCE PARAMETER")
    print("=" * 78)
    print("\nIn sample, on all six seasons at once -- deliberately the most generous")
    print("possible test. If no parameter helps HERE, no fitting rule can rescue it.")
    base_sc = deployed_scores(df)
    allb = aggregate(df, base_sc)
    for fam in CANDIDATES + ("dixon_coles",):
        print(f"\n  {fam}  (published 1X2 held fixed)")
        print(f"    {'param':>8s} {'impliedCov':>11s} " +
              "  ".join(f"{k:>9s}" for k in MARKETS if k not in ("1x2_rps", "1x2_ll")))
        for p in GRIDS[fam]:
            sc = arm_scores(df, fam, float(p), own_1x2=False)
            tot = aggregate(df, sc)
            cov = np.mean([D.implied_covariance(joint(
                df.lh.to_numpy(float)[i:i + 1], df.la.to_numpy(float)[i:i + 1],
                fam, float(p), temper=K["goal_temper"])[0])
                for i in range(0, len(df), 97)])
            cells = "  ".join(f"{tot[k] - allb[k]:+9.5f}"
                              for k in MARKETS if k not in ("1x2_rps", "1x2_ll"))
            star = " *" if p == 0.0 else ""
            print(f"    {p:+8.4f} {cov:+11.4f}  {cells}{star}")
    print("\n  (* = independence. Deltas vs the deployed baseline; negative = better.)")

    print("\n  And the same sweep on the RAW trio -- the 1X2 before the sharpening,")
    print("  which is the only route by which a dependence family could change the")
    print("  published 1X2 at all:")
    y = df.outcome.to_numpy(int)
    obs_draw = float((y == 1).mean())
    lh, la = df.lh.to_numpy(float), df.la.to_numpy(float)
    for fam in CANDIDATES + ("dixon_coles",):
        print(f"\n    {fam}:  {'param':>8s} {'P(draw)':>9s} {'O/E draws':>10s} "
              f"{'RPS':>9s} {'log loss':>10s}")
        for p in GRIDS[fam]:
            trio = trio_of(joint(lh, la, fam, float(p), temper=K["goal_temper"],
                                 fix_marginals=(fam != "dixon_coles")))
            pdw = float(trio[:, 1].mean())
            print(f"    {'':>{len(fam) + 2}s}{p:+8.4f} {pdw:9.4f} {obs_draw / pdw:10.4f} "
                  f"{rps3(y, trio):9.5f} {mll(y, trio):10.5f}")


# ── 5. the coherence invariant ────────────────────────────────────────────────

def cmd_coherence(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("5. COHERENCE: THE MATRIX MUST STILL SUM TO THE PUBLISHED 1X2")
    print("=" * 78)
    sub = df.iloc[::7].reset_index(drop=True)
    trio = sub[["ph", "pd", "pa"]].to_numpy(float)
    lh, la = sub.lh.to_numpy(float), sub.la.to_numpy(float)
    print(f"\nchecked on {len(sub):,} matches (every 7th)")
    worst_overall = 0.0
    for fam, p in [("independent", 0.0), ("dixon_coles", K["outcome_rho"]),
                   ("bivariate_poisson", 0.05), ("frank_copula", 0.5),
                   ("gaussian_copula", 0.08)]:
        m = reweight_to_trio(joint(lh, la, fam, p, temper=K["goal_temper"]), trio)
        err_t = float(np.abs(trio_of(m) - trio).max())
        err_s = float(np.abs(m.sum(axis=(1, 2)) - 1.0).max())
        paths = htft_paths(lh, la, fam, p, trio)
        ft = np.stack([paths[:, [PATHS.index(f"{x}/{y}") for x in "1X2"]].sum(1)
                       for y in "1X2"], axis=1)
        err_h = float(np.abs(ft - trio).max())
        worst_overall = max(worst_overall, err_t, err_s, err_h)
        print(f"  {fam:19s} param {p:+.3f}   max|matrix 1X2 - published| {err_t:.2e}   "
              f"max|HT/FT full-time 1X2 - published| {err_h:.2e}   sum-1 {err_s:.2e}")
    ok = worst_overall < 1e-9
    print(f"\n  {'OK' if ok else 'FAIL'}: worst disagreement {worst_overall:.2e} "
          "(< 1e-9 required). No page can show a 1X2 its own grid contradicts.")
    if not ok:
        raise SystemExit("coherence invariant broken")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["corr", "fit", "validate", "sweep", "coherence", "all"])
    ap.add_argument("--data-root", default=str(ROOT),
                    help="where data/ lives (default: the repo root; useful from a worktree, "
                         "whose data/ is gitignored and therefore empty)")
    a = ap.parse_args()
    root = Path(a.data_root)
    df = load(root)
    fits = None
    if a.cmd in ("corr", "all"):
        cmd_corr(df, root)
    if a.cmd in ("fit", "all"):
        fits = cmd_fit(df, fits)
    if a.cmd in ("validate", "all"):
        cmd_validate(df, fits)
    if a.cmd in ("sweep", "all"):
        cmd_sweep(df)
    if a.cmd in ("coherence", "all"):
        cmd_coherence(df)


if __name__ == "__main__":
    main()
