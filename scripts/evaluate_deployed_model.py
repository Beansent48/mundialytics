from __future__ import annotations

"""Full metric evaluation of the DEPLOYED club config (blend 0.30, no ELO) plus an
overfitting check. Uses the cached 8-fold components (mode='none') for the goal
markets, and does a fresh train-vs-test fit to measure the generalization gap.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("az", ROOT / "scripts" / "analyze_xg_components.py")
az = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(az)

W_GL, W_AD = 0.30, 0.70  # deployed blend


def reliability(y: np.ndarray, p: np.ndarray, bins=10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                     "pred": float(p[m].mean()), "emp": float(y[m].mean())})
    return pd.DataFrame(rows)


def main() -> None:
    comp = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/elo_components_8fold.csv")
    comp = comp[comp["mode"] == "none"].copy()
    print(f"Deployed-config eval: {len(comp)} held-out matches, blend {W_GL}/{W_AD}, no ELO\n")

    lh, la = az.arm_lambdas(comp, "gl_noxg", W_GL, W_AD, 0.0)
    p = az.arm_probs(lh, la)
    s = az.score(comp, p)

    # naive baselines (pooled base rates)
    o = comp["actual_outcome"].to_numpy()
    base = {"home": (o == "home").mean(), "draw": (o == "draw").mean(), "away": (o == "away").mean()}
    over_rate = comp["actual_over_25"].mean(); btts_rate = comp["actual_btts"].mean()
    # base-rate RPS (always predict base rates)
    ph0 = np.full(len(o), base["home"]); pd0 = np.full(len(o), base["draw"])
    base_rps = float((0.5 * ((ph0 - (o == "home")) ** 2 + (ph0 + pd0 - (o == "home") - (o == "draw")) ** 2)).mean())

    print("========== 1X2 ==========")
    print(f"  RPS         {s['rps']:.4f}   (base-rate {base_rps:.4f})")
    print(f"  log-loss    {s['logloss_1x2']:.4f}")
    print(f"  Brier       {s['brier_1x2']:.4f}")
    print(f"  accuracy    {s['acc_1x2']:.4f}   (always-home {base['home']:.4f})")
    print(f"  bias  H {s['bias_home']:+.4f}  D {s['bias_draw']:+.4f}  A {s['bias_away']:+.4f}")

    print("\n========== Over/Under 2.5 ==========")
    over = comp["actual_over_25"].to_numpy(float); po = p["p_over_25"]
    ll_o = -(over*np.log(np.clip(po,1e-9,1))+(1-over)*np.log(np.clip(1-po,1e-9,1))).mean()
    br_o = ((po-over)**2).mean()
    base_ll_o = -(over*np.log(over_rate)+(1-over)*np.log(1-over_rate)).mean()
    print(f"  log-loss    {ll_o:.4f}   (base-rate {base_ll_o:.4f})")
    print(f"  Brier       {br_o:.4f}")
    print(f"  pred mean {po.mean():.3f}  actual {over_rate:.3f}")
    print(reliability(over, po).to_string(index=False))

    print("\n========== BTTS ==========")
    btts = comp["actual_btts"].to_numpy(float); pb = p["p_btts"]
    ll_b = -(btts*np.log(np.clip(pb,1e-9,1))+(1-btts)*np.log(np.clip(1-pb,1e-9,1))).mean()
    base_ll_b = -(btts*np.log(btts_rate)+(1-btts)*np.log(1-btts_rate)).mean()
    print(f"  log-loss    {ll_b:.4f}   (base-rate {base_ll_b:.4f})")
    print(f"  Brier       {((pb-btts)**2).mean():.4f}")
    print(f"  pred mean {pb.mean():.3f}  actual {btts_rate:.3f}")
    print(reliability(btts, pb).to_string(index=False))

    print("\n========== goal-lambda calibration ==========")
    print(f"  mean predicted total goals (lh+la) = {(lh+la).mean():.3f}  (Big5 typical ~2.7)")

    print("\n========== per-fold RPS (stability / time-generalization) ==========")
    for ssn in sorted(comp["season"].unique()):
        c = comp[comp["season"] == ssn]
        lhs, las = az.arm_lambdas(c, "gl_noxg", W_GL, W_AD, 0.0)
        rp = az.score(c, az.arm_probs(lhs, las))["rps"]
        print(f"  {ssn}: RPS {rp:.4f}  (n={len(c)})")
    per = [az.score(comp[comp['season']==x], az.arm_probs(*az.arm_lambdas(comp[comp['season']==x],'gl_noxg',W_GL,W_AD,0.0)))['rps'] for x in comp['season'].unique()]
    print(f"  spread: min {min(per):.4f}  max {max(per):.4f}  std {np.std(per):.4f}")


if __name__ == "__main__":
    main()
