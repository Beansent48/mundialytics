from __future__ import annotations

"""Global evaluation from the components cache: config comparison (deployed vs an
xG-rate-augmented model, with OUT-OF-SAMPLE learned blend to avoid overfitting),
full per-market metrics, pure-statistical-value skill scores, and reference
benchmarks vs top football models.
"""

import importlib.util
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_az = importlib.util.spec_from_file_location("az", ROOT / "scripts" / "analyze_xg_components.py")
az = importlib.util.module_from_spec(_az); _az.loader.exec_module(az)

# Approximate published RPS reference points for club-football 1X2 (no-odds unless noted).
BENCHMARKS = [
    ("Coin/uniform (max ignorance)", 0.2222),
    ("Base-rate 'climatology'", None),          # filled from data
    ("Academic Poisson / Dixon-Coles (no odds)", 0.213),
    ("FiveThirtyEight SPI (club)", 0.2065),
    ("Bookmaker closing odds (ceiling)", 0.1900),
]


def lam(c: pd.DataFrame, w: dict) -> tuple[np.ndarray, np.ndarray]:
    lh = sum(w[k] * c[f"{k}_h"] for k in w); la = sum(w[k] * c[f"{k}_a"] for k in w)
    return np.clip(lh.to_numpy(float), 0.05, 6.0), np.clip(la.to_numpy(float), 0.05, 6.0)


def config_score(c: pd.DataFrame, w: dict) -> dict:
    lh, la = lam(c, w)
    return az.score(c, az.arm_probs(lh, la))


def learn_w(c: pd.DataFrame, keys: list[str], step: float = 0.1) -> dict:
    grid = np.round(np.arange(0, 1 + 1e-9, step), 3)
    best, brps = {k: (1.0 if k == keys[0] else 0.0) for k in keys}, 1e9
    if len(keys) == 2:
        combos = [(a, round(1 - a, 3)) for a in grid]
    else:  # 3 keys
        combos = [(a, b, round(1 - a - b, 3)) for a, b in product(grid, grid) if a + b <= 1 + 1e-9]
    for combo in combos:
        w = dict(zip(keys, combo))
        r = config_score(c, w)["rps"]
        if r < brps:
            brps, best = r, w
    return best


def skill_block(c: pd.DataFrame, w: dict, base: dict) -> None:
    lh, la = lam(c, w); p = az.arm_probs(lh, la); s = az.score(c, p)
    o = c["actual_outcome"].to_numpy()
    over = c["actual_over_25"].to_numpy(float); btts = c["actual_btts"].to_numpy(float)
    # base rates
    br = {"home": (o == "home").mean(), "draw": (o == "draw").mean(), "away": (o == "away").mean()}
    orate, brate = over.mean(), btts.mean()
    base_rps = base["rps"]; base_ll = base["ll1x2"]
    ln2 = math.log(2)
    # info gain bits/match over base rate for each market
    def bll(y, pr): pr = np.clip(pr, 1e-9, 1 - 1e-9); return -(y*np.log(pr)+(1-y)*np.log(1-pr)).mean()
    ll_o = bll(over, p["p_over_25"]); base_ll_o = bll(over, np.full_like(over, orate))
    ll_b = bll(btts, p["p_btts"]); base_ll_b = bll(btts, np.full_like(btts, brate))
    print(f"    RPS {s['rps']:.4f}   RPSS(vs base) {1 - s['rps']/base_rps:+.3f}   acc {s['acc_1x2']:.3f} (base {max(br.values()):.3f})")
    print(f"    1X2 info gain    {(base_ll - s['logloss_1x2'])/ln2:+.4f} bits/match")
    print(f"    O/U2.5 Brier {((p['p_over_25']-over)**2).mean():.4f}  skill {1-((p['p_over_25']-over)**2).mean()/((orate-over)**2).mean():+.3f}  info {(base_ll_o-ll_o)/ln2:+.4f} bits")
    print(f"    BTTS   Brier {((p['p_btts']-btts)**2).mean():.4f}  skill {1-((p['p_btts']-btts)**2).mean()/((brate-btts)**2).mean():+.3f}  info {(base_ll_b-ll_b)/ln2:+.4f} bits")
    # 1X2 calibration ECE (pool the 3 selections)
    ys = np.concatenate([(o=="home").astype(float),(o=="draw").astype(float),(o=="away").astype(float)])
    ps = np.concatenate([p["p_home_win"],p["p_draw"],p["p_away_win"]])
    edges=np.linspace(0,1,11); idx=np.clip(np.digitize(ps,edges)-1,0,9); ece=0.0
    for b in range(10):
        m=idx==b
        if m.sum(): ece += m.sum()/len(ps)*abs(ps[m].mean()-ys[m].mean())
    print(f"    1X2 calibration ECE {ece:.4f}  (lower=better; <0.02 = well calibrated)")


def main() -> None:
    c = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/global_components.csv")
    c["actual_outcome"] = np.where(c.hg > c.ag, "home", np.where(c.hg < c.ag, "away", "draw"))
    c["actual_over_25"] = ((c.hg + c.ag) > 2.5).astype(int)
    c["actual_btts"] = ((c.hg > 0) & (c.ag > 0)).astype(int)
    c = c.dropna(subset=["gl_h", "ad_h", "xr_h", "adx_h"])
    seasons = sorted(c["season"].unique())
    print(f"Global backtest: {len(c)} held-out matches, {len(seasons)} folds {seasons}\n")

    deployed = {"gl": 0.30, "ad": 0.70}
    dep = config_score(c, deployed)
    o = c["actual_outcome"].to_numpy()
    br = {"home": (o == "home").mean(), "draw": (o == "draw").mean(), "away": (o == "away").mean()}
    base_rps = float((0.5*((br["home"]-(o=="home"))**2 + (br["home"]+br["draw"]-(o=="home")-(o=="draw"))**2)).mean())
    base_ll = float(-np.log(np.where(o=="home",br["home"],np.where(o=="draw",br["draw"],br["away"]))).mean())
    base = {"rps": base_rps, "ll1x2": base_ll}

    # Out-of-sample learned blend over {ad, xr} and {gl, ad, xr}: learn on other folds.
    def lofo(keys):
        preds=[]
        for s in seasons:
            wl = learn_w(c[c["season"] != s], keys)
            preds.append((c[c["season"] == s], wl))
        pooled = pd.concat([d for d,_ in preds], ignore_index=True)
        # apply per-fold weights
        lh=[];la=[]
        for d,wl in preds:
            l1,l2=lam(d,wl); lh.append(l1); la.append(l2)
        lh=np.concatenate(lh); la=np.concatenate(la)
        return az.score(pooled, az.arm_probs(lh,la)), preds
    oos2, p2 = lofo(["ad","xr"]); oos3, p3 = lofo(["gl","ad","xr"])
    oracle3 = config_score(c, learn_w(c, ["gl","ad","xr"]))

    print("===== 1X2 RPS by config =====")
    print(f"  base-rate climatology            {base_rps:.4f}")
    print(f"  DEPLOYED (0.30 gl + 0.70 ad)      {dep['rps']:.4f}")
    print(f"  OOS-learned  ad+xr               {oos2['rps']:.4f}")
    print(f"  OOS-learned  gl+ad+xr            {oos3['rps']:.4f}")
    print(f"  in-sample oracle gl+ad+xr        {config_score(c, learn_w(c,['gl','ad','xr']))['rps']:.4f}  (upper bound)")
    print(f"  learned weights per fold (gl+ad+xr): " + "; ".join(f"{s}:{tuple(round(v,2) for v in w.values())}" for (d,w),s in zip(p3,seasons)))

    print("\n===== DEPLOYED — full statistical value =====")
    skill_block(c, deployed, base)
    print("\n===== IMPROVED (OOS-learned gl+ad+xr) — full statistical value =====")
    # use the pooled OOS predictions: rebuild a single weight? report via per-fold. Approx with mean weights.
    meanw = {k: float(np.mean([w[k] for _, w in p3])) for k in ["gl","ad","xr"]}
    print(f"    (mean learned weights: gl {meanw['gl']:.2f}  ad {meanw['ad']:.2f}  xr {meanw['xr']:.2f})")
    skill_block(c, meanw, base)

    print("\n===== per-fold RPS (improved, mean weights) — stability / no-overfit =====")
    per=[]
    for s in seasons:
        r = config_score(c[c["season"]==s], meanw)["rps"]; per.append(r)
        print(f"  {s}: {r:.4f}")
    print(f"  spread std {np.std(per):.4f}")

    print("\n===== BENCHMARK vs reference models (1X2 RPS; lower=better) =====")
    improved_rps = oos3["rps"]
    rows = [(n, v) for n, v in BENCHMARKS]
    rows = [(n, base_rps if v is None else v) for n, v in rows]
    rows.append(("*** OURS deployed ***", dep["rps"]))
    rows.append(("*** OURS improved (xG-rate) ***", improved_rps))
    for n, v in sorted(rows, key=lambda x: -x[1]):
        bar = "#" * int((0.2222 - v) / 0.0015)
        print(f"  {n:40s} {v:.4f}  {bar}")


if __name__ == "__main__":
    main()
