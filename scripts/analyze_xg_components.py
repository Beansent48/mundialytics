from __future__ import annotations

"""Fast vectorized analysis of cached xG lambda components (step B).

Reads data/processed/enriched/understat_xg/xg_components.csv (produced by
backtest_xg_phases.py) and evaluates blend arms with a fully vectorized
Dixon-Coles Poisson scoreline computation (all matches at once), so grid-search
over blend weights is cheap.
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
K = 11          # goals grid 0..10
RHO = -0.07     # Dixon-Coles low-score correction (engine default)

_ks = np.arange(K)
_home_mask = (_ks[:, None] > _ks[None, :])
_draw_mask = (_ks[:, None] == _ks[None, :])
_away_mask = (_ks[:, None] < _ks[None, :])
_over_mask = (_ks[:, None] + _ks[None, :] >= 3)
_btts_mask = (_ks[:, None] >= 1) & (_ks[None, :] >= 1)


def arm_probs(lh: np.ndarray, la: np.ndarray, rho: float = RHO) -> dict:
    lh = np.clip(lh, 0.05, 6.0); la = np.clip(la, 0.05, 6.0)
    ph = poisson.pmf(_ks[:, None], lh[None, :])   # (K, N)
    pa = poisson.pmf(_ks[:, None], la[None, :])   # (K, N)
    joint = ph[:, None, :] * pa[None, :, :]       # (K, K, N)
    joint[0, 0, :] *= (1 - rho * lh * la)
    joint[0, 1, :] *= (1 + rho * lh)
    joint[1, 0, :] *= (1 + rho * la)
    joint[1, 1, :] *= (1 - rho)
    joint = np.clip(joint, 0, None)
    joint /= joint.sum(axis=(0, 1), keepdims=True)
    m = lambda mask: (joint * mask[:, :, None]).sum(axis=(0, 1))
    return {"p_home_win": m(_home_mask), "p_draw": m(_draw_mask), "p_away_win": m(_away_mask),
            "p_over_25": m(_over_mask), "p_btts": m(_btts_mask)}


def arm_lambdas(comp: pd.DataFrame, gl_col: str, w_gl: float, w_ad: float, w_adx: float):
    lh = w_gl * comp[f"{gl_col}_h"] + w_ad * comp["ad_h"] + w_adx * comp["adxg_h"]
    la = w_gl * comp[f"{gl_col}_a"] + w_ad * comp["ad_a"] + w_adx * comp["adxg_a"]
    return lh.to_numpy(float), la.to_numpy(float)


def score(comp: pd.DataFrame, p: dict) -> dict:
    outcome = comp["actual_outcome"].to_numpy()
    ph, pdw, pa = p["p_home_win"], p["p_draw"], p["p_away_win"]
    yh = (outcome == "home").astype(float); yd = (outcome == "draw").astype(float); ya = (outcome == "away").astype(float)
    rps = 0.5 * ((ph - yh) ** 2 + (ph + pdw - yh - yd) ** 2)
    actual = np.where(outcome == "home", ph, np.where(outcome == "draw", pdw, pa))
    ll = -np.log(np.clip(actual, 1e-9, 1.0))
    brier = (ph - yh) ** 2 + (pdw - yd) ** 2 + (pa - ya) ** 2
    over = comp["actual_over_25"].to_numpy(float); btts = comp["actual_btts"].to_numpy(float)
    po = np.clip(p["p_over_25"], 1e-9, 1 - 1e-9); pb = np.clip(p["p_btts"], 1e-9, 1 - 1e-9)
    return {"n": int(len(comp)), "rps": float(rps.mean()), "logloss_1x2": float(ll.mean()),
            "brier_1x2": float(brier.mean()),
            "logloss_over25": float(-(over * np.log(po) + (1 - over) * np.log(1 - po)).mean()),
            "logloss_btts": float(-(btts * np.log(pb) + (1 - btts) * np.log(1 - pb)).mean()),
            "acc_1x2": float((np.argmax(np.c_[ph, pdw, pa], axis=1)
                              == np.where(outcome == "home", 0, np.where(outcome == "draw", 1, 2))).mean()),
            "bias_home": float(ph.mean() - yh.mean()), "bias_draw": float(pdw.mean() - yd.mean()),
            "bias_away": float(pa.mean() - ya.mean())}


def score_arm(comp, gl_col, w_gl, w_ad, w_adx):
    return score(comp, arm_probs(*arm_lambdas(comp, gl_col, w_gl, w_ad, w_adx)))


def learn_weights(comp, gl_col, step=0.05):
    grid = np.round(np.arange(0, 1 + 1e-9, step), 3)
    best, best_rps = (0.6, 0.4, 0.0), 1e9
    for w_gl, w_adx in product(grid, grid):
        w_ad = 1.0 - w_gl - w_adx
        if w_ad < -1e-9:
            continue
        r = score_arm(comp, gl_col, w_gl, w_ad, w_adx)["rps"]
        if r < best_rps:
            best_rps, best = r, (float(w_gl), float(max(w_ad, 0.0)), float(w_adx))
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--components", default="data/processed/enriched/understat_xg/xg_components.csv")
    ap.add_argument("--out", default="data/processed/enriched/understat_xg/xg_phases_result.json")
    args = ap.parse_args()
    comp = pd.read_csv(ROOT / args.components)
    seasons = list(dict.fromkeys(comp["season"].tolist()))
    print(f"components: {len(comp)} rows, seasons={seasons}", flush=True)

    arms = {"baseline": ("gl_noxg", 0.6, 0.4, 0.0), "feature": ("gl_xg", 0.6, 0.4, 0.0),
            "target": ("gl_noxg", 0.6, 0.0, 0.4), "both": ("gl_xg", 0.6, 0.0, 0.4)}
    res = {name: score_arm(comp, *cfg) for name, cfg in arms.items()}

    # Out-of-sample learned weight: for each season, learn on earlier seasons only.
    chosen = {}
    parts_c, parts_p = [], []
    for i, s in enumerate(seasons):
        past = comp[comp["season"].isin(seasons[:i])]
        w = ("gl_noxg", 0.6, 0.4, 0.0) if len(past) < 300 else ("gl_noxg", *learn_weights(past, "gl_noxg"))
        chosen[s] = w
        cur = comp[comp["season"] == s].reset_index(drop=True)
        parts_c.append(cur); parts_p.append(arm_probs(*arm_lambdas(cur, *w)))
    lc = pd.concat(parts_c, ignore_index=True)
    lp = {k: np.concatenate([p[k] for p in parts_p]) for k in parts_p[0]}
    res["learned_oos"] = score(lc, lp)
    oracle = learn_weights(comp, "gl_noxg"); oracle_xg = learn_weights(comp, "gl_xg")
    res["oracle_insample"] = score_arm(comp, "gl_noxg", *oracle)

    keys = ["rps", "logloss_1x2", "brier_1x2", "logloss_over25", "logloss_btts", "acc_1x2", "bias_home", "bias_away"]
    print("\n===== POOLED ARMS =====", flush=True)
    print(f"{'arm':16s}" + "".join(f"{k:>14s}" for k in keys), flush=True)
    for name in ["baseline", "feature", "target", "both", "learned_oos", "oracle_insample"]:
        s = res[name]
        print(f"{name:16s}" + "".join(f"{s[k]:+14.5f}" for k in keys), flush=True)
    base = res["baseline"]
    print("\nDeltas vs baseline (neg=better on scores):", flush=True)
    for name in ["feature", "target", "both", "learned_oos", "oracle_insample"]:
        s = res[name]
        print(f"  {name:16s} dRPS={s['rps']-base['rps']:+.5f}  dLL={s['logloss_1x2']-base['logloss_1x2']:+.5f}"
              f"  dBias_away={s['bias_away']-base['bias_away']:+.5f}", flush=True)
    print(f"\noracle weights (gl_noxg): gl={oracle[0]:.2f} ad={oracle[1]:.2f} adxg={oracle[2]:.2f}", flush=True)
    print(f"oracle weights (gl_xg):   gl={oracle_xg[0]:.2f} ad={oracle_xg[1]:.2f} adxg={oracle_xg[2]:.2f}", flush=True)
    print("learned OOS weights per season:", flush=True)
    for s, w in chosen.items():
        print(f"  {s}: gl={w[1]:.2f} ad={w[2]:.2f} adxg={w[3]:.2f}", flush=True)

    (ROOT / args.out).write_text(json.dumps(
        {"arms": res, "oracle_goals": list(oracle), "oracle_xgfeat": list(oracle_xg),
         "learned_oos_weights": {s: list(w[1:]) for s, w in chosen.items()}}, indent=2), encoding="utf-8")
    print(f"\nWROTE {ROOT / args.out}", flush=True)


if __name__ == "__main__":
    main()
