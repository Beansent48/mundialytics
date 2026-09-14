from __future__ import annotations

"""B) Can the EXACT-SCORE distribution be sharpened? (the "always 1-1" question)

The user reads the modal exact score — near-always a low draw — as the model
having no opinion. That modal cell is a true property of a product-Poisson
matrix, but it is fair to ask whether the *distribution over scores* is itself
well-calibrated, or whether a different low-cell correction (Dixon-Coles rho)
and/or sub-Poisson tempering (theta) would assign more probability to the scores
that actually happen.

This measures exactly that, walk-forward, on the deployed blended lambdas
(the cache's lh/la already include the xG-rate blend + goals rescale, i.e. the
same numbers the served matrix is built from). For each held-out season the
(rho, theta) pair is chosen by scoreline log-likelihood on PAST seasons only,
then scored on the held-out one against:

  * scoreline log-loss  -- mean -log P(true exact score); the direct answer
  * 1X2 RPS             -- guardrail: the served 1X2 trio must not regress

Baseline is the deployed matrix config: rho=-0.17, theta=1.0. Nothing here
touches a default; it only reports whether a change is worth proposing, per the
protect-the-baseline rule. Run:

    .venv/Scripts/python.exe scripts/experiment_scoreline_calibration.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/processed/enriched/understat_xg/walkforward_preds.csv"
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
KMAX = 12
RHOS = np.round(np.arange(-0.25, 0.051, 0.02), 3)
THETAS = np.round(np.arange(1.0, 1.41, 0.05), 3)
BASE_RHO, BASE_THETA = -0.17, 1.0   # deployed matrix config


def tempered_pmf(lam: np.ndarray, theta: float) -> np.ndarray:
    """(n, KMAX+1) tempered-Poisson marginals: p_k ∝ pois(k|lam)^theta."""
    k = np.arange(KMAX + 1)
    base = poisson.pmf(k[None, :], lam[:, None]) ** theta
    return base / base.sum(axis=1, keepdims=True)


def _tau_lowcells(lh: np.ndarray, la: np.ndarray, rho: float):
    """Dixon-Coles tau for the four low-score cells, clipped as in production."""
    clip = lambda x: np.clip(x, 0.05, 3.0)
    return {
        (0, 0): clip(1.0 - rho * lh * la),
        (0, 1): clip(1.0 + rho * lh),
        (1, 0): clip(1.0 + rho * la),
        (1, 1): clip(1.0 - rho * np.ones_like(lh)),
    }


def _normaliser(ph, pa, tau) -> np.ndarray:
    """Z = total mass after the tau reweighting of the 4 low cells."""
    z = np.ones(ph.shape[0])
    for (i, j), t in tau.items():
        z = z + ph[:, i] * pa[:, j] * (t - 1.0)
    return z


def scoreline_logloss(hg, ag, lh, la, rho, theta) -> float:
    ph = tempered_pmf(lh, theta)
    pa = tempered_pmf(la, theta)
    tau = _tau_lowcells(lh, la, rho)
    z = _normaliser(ph, pa, tau)
    idx = np.arange(len(hg))
    p_true = ph[idx, hg] * pa[idx, ag]
    for (i, j), t in tau.items():
        m = (hg == i) & (ag == j)
        p_true = np.where(m, p_true * t, p_true)
    p_true = p_true / z
    return float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean())


def outcome_probs(lh, la, rho, theta):
    """(mh, md, ma) 1X2 marginals from the same (rho, theta) matrix."""
    ph = tempered_pmf(lh, theta)
    pa = tempered_pmf(la, theta)
    n = len(lh)
    mh = np.zeros(n)
    md = np.zeros(n)
    for i in range(KMAX + 1):
        md += ph[:, i] * pa[:, i]
        for j in range(i):
            mh += ph[:, i] * pa[:, j]
    ma = 1.0 - mh - md
    tau = _tau_lowcells(lh, la, rho)
    z = _normaliser(ph, pa, tau)
    md = md + ph[:, 0] * pa[:, 0] * (tau[(0, 0)] - 1.0) + ph[:, 1] * pa[:, 1] * (tau[(1, 1)] - 1.0)
    mh = mh + ph[:, 1] * pa[:, 0] * (tau[(1, 0)] - 1.0)
    ma = ma + ph[:, 0] * pa[:, 1] * (tau[(0, 1)] - 1.0)
    return mh / z, md / z, ma / z


def rps_1x2(hg, ag, lh, la, rho, theta) -> float:
    mh, md, ma = outcome_probs(lh, la, rho, theta)
    y = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    P = np.stack([mh, md, ma], axis=1)
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return float(((cp - cy) ** 2)[:, :2].sum(axis=1).mean() / 2)


def main() -> None:
    w = pd.read_csv(CACHE)
    w["hg"] = w["hg"].clip(upper=KMAX).astype(int)
    w["ag"] = w["ag"].clip(upper=KMAX).astype(int)

    base_ll, tuned_ll, base_rps, tuned_rps = [], [], [], []
    picks = []
    for s in TEST_SEASONS:
        tr = w[w.season < s]
        te = w[w.season == s]
        if len(tr) < 1000 or len(te) == 0:
            continue
        htr, atr = tr.hg.to_numpy(), tr.ag.to_numpy()
        ltr_h, ltr_a = tr.lh.to_numpy(), tr.la.to_numpy()
        grid = {(r, th): scoreline_logloss(htr, atr, ltr_h, ltr_a, r, th)
                for r in RHOS for th in THETAS}
        (r_opt, th_opt) = min(grid, key=grid.get)
        picks.append((s, r_opt, th_opt))

        hte, ate = te.hg.to_numpy(), te.ag.to_numpy()
        lte_h, lte_a = te.lh.to_numpy(), te.la.to_numpy()
        n = len(te)
        base_ll.append((scoreline_logloss(hte, ate, lte_h, lte_a, BASE_RHO, BASE_THETA), n, s))
        tuned_ll.append((scoreline_logloss(hte, ate, lte_h, lte_a, r_opt, th_opt), n, s))
        base_rps.append((rps_1x2(hte, ate, lte_h, lte_a, BASE_RHO, BASE_THETA), n, s))
        tuned_rps.append((rps_1x2(hte, ate, lte_h, lte_a, r_opt, th_opt), n, s))

    pool = lambda a: sum(x[0] * x[1] for x in a) / sum(x[1] for x in a)
    signs = lambda a, b: " ".join(f"{x[2][-2:]}{'+' if y[0] < x[0] else '-'}"
                                  for x, y in zip(a, b))

    print("chosen (rho, theta) per fold (train-only):")
    for s, r, th in picks:
        print(f"  {s}: rho={r:+.3f}  theta={th:.2f}")
    b, t = pool(base_ll), pool(tuned_ll)
    print(f"\nscoreline log-loss: base(rho={BASE_RHO},th=1.0) {b:.4f} -> tuned {t:.4f} "
          f"(d {t-b:+.4f})  [{signs(base_ll, tuned_ll)}]")
    b, t = pool(base_rps), pool(tuned_rps)
    print(f"1X2 RPS guardrail : base {b:.4f} -> tuned {t:.4f} "
          f"(d {t-b:+.4f})  [{signs(base_rps, tuned_rps)}]")
    print("\n(+ = tuned better that fold. Deploy only if scoreline log-loss "
          "improves on 4-5/5 folds AND 1X2 RPS does not regress.)")


if __name__ == "__main__":
    main()
