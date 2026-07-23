from __future__ import annotations

"""C) Sub-Poisson goal counts (Boshnakov et al. 2017 motivation).

Measured on 10,403 walk-forward matches: Pearson dispersion 0.91 home / 0.92
away (goals are UNDER-dispersed vs Poisson; reality has more 2-3 goal games
and thinner 4+ tails than our matrices assume; DC-rho only fixes the low
cells). Pragmatic one-parameter version of the paper's idea: temper the
Poisson pmf, p_k ∝ pois(k|lam)^theta renormalized (theta>1 concentrates),
theta fitted by max likelihood on PAST seasons only (walk-forward), shared
across home/away. Evaluated on O/U totals and indicative 1X2 from the
independent-product matrix (both arms identical machinery, only theta differs).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/processed/enriched/understat_xg/walkforward_preds.csv"
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
KMAX = 12
THETAS = np.arange(1.0, 1.51, 0.05)


def tempered_pmf(lam: np.ndarray, theta: float) -> np.ndarray:
    """(n, KMAX+1) matrix of tempered-Poisson pmfs."""
    k = np.arange(KMAX + 1)
    base = poisson.pmf(k[None, :], lam[:, None]) ** theta
    return base / base.sum(axis=1, keepdims=True)


def goal_ll(hg, ag, lh, la, theta) -> float:
    ph = tempered_pmf(lh, theta)
    pa = tempered_pmf(la, theta)
    idx = np.arange(len(hg))
    return float(-(np.log(np.clip(ph[idx, hg], 1e-12, 1))
                   + np.log(np.clip(pa[idx, ag], 1e-12, 1))).mean())


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    w = pd.read_csv(CACHE)
    w["hg"] = w["hg"].clip(upper=KMAX).astype(int)
    w["ag"] = w["ag"].clip(upper=KMAX).astype(int)

    res = {ln: {"pois": [], "temp": []} for ln in [1.5, 2.5, 3.5]}
    res1x2 = {"pois": [], "temp": []}
    for s in TEST_SEASONS:
        te = w[w.season == s]
        tr = w[w.season < s]
        if len(tr) < 1000 or len(te) == 0:
            continue
        lls = {th: goal_ll(tr.hg.to_numpy(), tr.ag.to_numpy(),
                           tr.lh.to_numpy(), tr.la.to_numpy(), th) for th in THETAS}
        theta = min(lls, key=lls.get)
        lh, la = te.lh.to_numpy(), te.la.to_numpy()
        for tag, th in [("pois", 1.0), ("temp", theta)]:
            ph = tempered_pmf(lh, th)
            pa = tempered_pmf(la, th)
            # total-goals distribution by convolution
            tot = np.zeros((len(te), 2 * KMAX + 1))
            for i in range(KMAX + 1):
                for j in range(KMAX + 1):
                    tot[:, i + j] += ph[:, i] * pa[:, j]
            y_tot = (te.hg + te.ag).to_numpy()
            for ln in [1.5, 2.5, 3.5]:
                p_over = tot[:, int(ln) + 1:].sum(axis=1)
                res[ln][tag].append((bll((y_tot > ln).astype(float), p_over), len(te), s, theta))
            # indicative 1X2 from the same matrices
            mh = np.zeros(len(te))
            md = np.zeros(len(te))
            for i in range(KMAX + 1):
                md += ph[:, i] * pa[:, i]
                for j in range(i):
                    mh += ph[:, i] * pa[:, j]
            ma = 1 - mh - md
            y = np.where(te.hg > te.ag, 0, np.where(te.hg == te.ag, 1, 2))
            P = np.stack([mh, md, ma], axis=1)
            Y = np.zeros_like(P)
            Y[np.arange(len(y)), y] = 1
            cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
            res1x2[tag].append((float(((cp - cy) ** 2)[:, :2].sum(axis=1).mean() / 2), len(te), s, theta))

    pool = lambda a: sum(x[0] * x[1] for x in a) / sum(x[1] for x in a)
    thetas = [f"{x[3]:.2f}" for x in res[2.5]["temp"]]
    print(f"theta elegido por fold (train-only): {thetas}")
    for ln in [1.5, 2.5, 3.5]:
        p0, p1 = pool(res[ln]["pois"]), pool(res[ln]["temp"])
        folds = " ".join(f"{a[2][-2:]}{'+' if b[0] < a[0] else '-'}"
                         for a, b in zip(res[ln]["pois"], res[ln]["temp"]))
        print(f"O/U {ln}: poisson {p0:.4f} -> tempered {p1:.4f} (d {p1-p0:+.4f})  [{folds}]")
    p0, p1 = pool(res1x2["pois"]), pool(res1x2["temp"])
    folds = " ".join(f"{a[2][-2:]}{'+' if b[0] < a[0] else '-'}"
                     for a, b in zip(res1x2["pois"], res1x2["temp"]))
    print(f"1X2 (indicativo, matriz indep): {p0:.4f} -> {p1:.4f} (d {p1-p0:+.4f})  [{folds}]")


if __name__ == "__main__":
    main()
