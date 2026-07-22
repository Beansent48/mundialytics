from __future__ import annotations

"""Fit and OOS-validate a sharpening calibration for the deployed model's 1X2
probabilities (and a Platt recal for Over2.5), using the cached walk-forward
predictions. Leave-one-fold-out: the calibration parameter is learned on the
other seasons and applied to the held-out one. Reports ECE + RPS + log-loss
before/after. Isolated: no engine changes.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/processed/enriched/understat_xg/walkforward_preds.csv"


def sharpen(P: np.ndarray, gamma: float) -> np.ndarray:
    Q = np.clip(P, 1e-9, 1.0) ** gamma
    return Q / Q.sum(axis=1, keepdims=True)


def metrics_1x2(P: np.ndarray, o: np.ndarray) -> dict:
    y = np.c_[(o == "home"), (o == "draw"), (o == "away")].astype(float)
    rps = (0.5 * ((P[:, 0] - y[:, 0]) ** 2 + (P[:, 0] + P[:, 1] - y[:, 0] - y[:, 1]) ** 2)).mean()
    ll = -np.log(np.clip((P * y).sum(axis=1), 1e-9, 1)).mean()
    ys = y.T.ravel(); ps = P.T.ravel()
    edges = np.linspace(0, 1, 11); idx = np.clip(np.digitize(ps, edges) - 1, 0, 9)
    ece = sum((idx == b).sum() / len(ps) * abs(ps[idx == b].mean() - ys[idx == b].mean())
              for b in range(10) if (idx == b).sum())
    return {"rps": float(rps), "ll": float(ll), "ece": float(ece)}


def main() -> None:
    d = pd.read_csv(CACHE)
    o = np.where(d.hg > d.ag, "home", np.where(d.hg < d.ag, "away", "draw"))
    P = d[["ph", "pd", "pa"]].to_numpy(float)
    seasons = d["season"].to_numpy()
    uniq = sorted(set(seasons))

    print("===== 1X2 sharpening (leave-one-fold-out) =====")
    grid = np.round(np.arange(0.80, 1.61, 0.05), 2)
    raw_pool, cal_pool, o_pool, chosen = [], [], [], {}
    for s in uniq:
        tr = seasons != s; te = seasons == s
        best_g, best_ll = 1.0, np.inf
        for g in grid:
            ll = metrics_1x2(sharpen(P[tr], g), o[tr])["ll"]
            if ll < best_ll:
                best_ll, best_g = ll, g
        chosen[s] = best_g
        raw_pool.append(P[te]); cal_pool.append(sharpen(P[te], best_g)); o_pool.append(o[te])
    Praw = np.vstack(raw_pool); Pcal = np.vstack(cal_pool); oo = np.concatenate(o_pool)
    m0, m1 = metrics_1x2(Praw, oo), metrics_1x2(Pcal, oo)
    print(f"  learned gamma per held-out fold: { {s: g for s, g in chosen.items()} }")
    print(f"  RAW : RPS {m0['rps']:.4f}  LL {m0['ll']:.4f}  ECE {m0['ece']:.4f}")
    print(f"  CAL : RPS {m1['rps']:.4f}  LL {m1['ll']:.4f}  ECE {m1['ece']:.4f}")
    print(f"  delta RPS {m1['rps']-m0['rps']:+.4f}  LL {m1['ll']-m0['ll']:+.4f}  ECE {m1['ece']-m0['ece']:+.4f}")
    per = []
    for s in uniq:
        te = seasons == s
        d0 = metrics_1x2(P[te], o[te])["rps"]; d1 = metrics_1x2(sharpen(P[te], chosen[s]), o[te])["rps"]
        per.append(d1 - d0)
        print(f"    {s}: dRPS {d1-d0:+.4f} (gamma {chosen[s]})")

    print("\n===== Over2.5 Platt (leave-one-fold-out) =====")
    over = ((d.hg + d.ag) > 2.5).astype(float).to_numpy()
    po = np.clip(d.po25.to_numpy(float), 1e-6, 1 - 1e-6)
    z = np.log(po / (1 - po))
    from sklearn.linear_model import LogisticRegression
    raws, cals, ys = [], [], []
    for s in uniq:
        tr = seasons != s; te = seasons == s
        lr = LogisticRegression(C=1e6).fit(z[tr].reshape(-1, 1), over[tr])
        cals.append(lr.predict_proba(z[te].reshape(-1, 1))[:, 1]); raws.append(po[te]); ys.append(over[te])
    pr, pc, yy = np.concatenate(raws), np.concatenate(cals), np.concatenate(ys)
    def bll(p): p = np.clip(p, 1e-9, 1 - 1e-9); return float(-(yy*np.log(p)+(1-yy)*np.log(1-p)).mean())
    def bece(p):
        edges = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, edges)-1, 0, 9)
        return float(sum((idx==b).sum()/len(p)*abs(p[idx==b].mean()-yy[idx==b].mean()) for b in range(10) if (idx==b).sum()))
    print(f"  RAW : LL {bll(pr):.4f}  ECE {bece(pr):.4f}")
    print(f"  CAL : LL {bll(pc):.4f}  ECE {bece(pc):.4f}   delta LL {bll(pc)-bll(pr):+.4f}")


if __name__ == "__main__":
    main()
