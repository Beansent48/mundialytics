from __future__ import annotations

"""Player-props micro A/Bs:
  1) attack-factor exponent grid {0.5, 0.7, 0.9, 1.0} per attacking prop
     (0.7 was picked once by intuition, never optimized);
  2) assists: add a key-passes volume signal to the xA/assist blend
     (key passes are ~5x more frequent than assists -> stabler small samples).
Walk-forward, usual folds, deployed v5 recipe as the base.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def p_ge(mu, k, disp=1.0):
    mu = np.clip(np.asarray(mu, float), 1e-6, 10)
    if disp > 1.05:
        r = mu / (disp - 1.0)
        return 1 - nbinom.cdf(k - 1, r, 1.0 / disp)
    return 1 - poisson.cdf(k - 1, mu)


def main() -> None:
    t0 = time.time()
    import backtest_player_props as bp
    pm = bp.build_panel()
    pm = bp.add_context(pm)
    g = pm.sort_values(["player_id", "date", "game_id"]).groupby("player_id", group_keys=False)
    pm["key_passes"] = pd.to_numeric(pm["key_passes"], errors="coerce").fillna(0.0)
    pm["c_key_passes"] = g["key_passes"].apply(lambda s: s.shift(1).cumsum()).fillna(0.0)
    pm["rr_key_passes"] = g["key_passes"].apply(lambda s: s.shift(1).rolling(15, min_periods=1).sum()).fillna(0.0)

    played = pm["minutes"] > 0
    train = pm[(~pm["season"].isin(bp.TEST_SEASONS)) & played]
    stats = ["xg", "goals", "shots", "xa", "assists", "key_passes"]
    pri = train.groupby("pgroup").apply(
        lambda gr: pd.Series({c: gr[c].sum() / max(gr["minutes"].sum(), 1) * 90.0 for c in stats}),
        include_groups=False)
    glob = {c: train[c].sum() / max(train["minutes"].sum(), 1) * 90.0 for c in stats}
    RW = {"xg": 0.5, "goals": 0.5, "shots": 0.5, "xa": 0.0, "assists": 0.0, "key_passes": 0.0}
    for c in stats:
        prior = pm["pgroup"].map(pri[c]).fillna(glob[c])
        r_car = bp.shrunk_rate(pm[f"c_{c}"], pm["cmin"], prior)
        if RW[c] > 0:
            r_rec = bp.shrunk_rate(pm[f"rr_{c}"], pm["rmin15"], r_car, k=450.0)
            pm[f"r_{c}"] = RW[c] * r_rec + (1 - RW[c]) * r_car
        else:
            pm[f"r_{c}"] = r_car

    pos_min = train.groupby("pgroup")["minutes"].mean()
    prior_min = pm["pgroup"].map(pos_min).fillna(float(train["minutes"].mean()))
    cred_m = pm["nplayed10"].fillna(0) / (pm["nplayed10"].fillna(0) + 3.0)
    pm["exp_min"] = cred_m * pm["avg_minp10"].fillna(prior_min) + (1 - cred_m) * prior_min
    emins = pm["exp_min"].clip(20, 95) / 90.0
    af_raw = pm["atk_factor"].fillna(1.0)

    test_mask = pm["season"].isin(bp.TEST_SEASONS) & pm["team_lam"].notna() & played
    t = pm[test_mask]
    seas = t["season"].to_numpy()
    print(f"panel ready ({time.time()-t0:.0f}s), test {len(t)}", flush=True)

    # 1) af exponent grid
    props = {
        "ANYTIME": ((0.7 * pm["r_xg"] + 0.3 * pm["r_goals"]) * emins, 1, 1.0, (pm["goals"] >= 1)),
        "SHOTS2+": (pm["r_shots"] * emins, 2, 1.3, (pm["shots"] >= 2)),
        "ASSIST": ((0.7 * pm["r_xa"] + 0.3 * pm["r_assists"]) * emins, 1, 1.0, (pm["assists"] >= 1)),
    }
    print("\n===== af exponent grid (LL per prop; * = deployed 0.7) =====")
    for name, (mu_base, k, disp, y_all) in props.items():
        y = y_all[test_mask].astype(float).to_numpy()
        row = []
        for e in [0.5, 0.7, 0.9, 1.0]:
            mu = (mu_base * af_raw ** e)[test_mask]
            row.append((e, bll(y, p_ge(mu, k, disp))))
        best = min(row, key=lambda x: x[1])
        cells = " | ".join(f"{'*' if e == 0.7 else ''}{e}: {v:.4f}{' <-BEST' if (e, v) == best else ''}"
                           for e, v in row)
        print(f"  {name:8s}: {cells}", flush=True)

    # 2) assists + key passes
    conv = glob["assists"] / max(glob["key_passes"], 1e-9)
    y = (t["assists"] >= 1).astype(float).to_numpy()
    af7 = af_raw ** 0.7
    variants = {
        "deployed (.7xa+.3as)": (0.7 * pm["r_xa"] + 0.3 * pm["r_assists"]) * emins * af7,
        "+KP (.55/.20/.25)": (0.55 * pm["r_xa"] + 0.2 * pm["r_assists"] + 0.25 * pm["r_key_passes"] * conv) * emins * af7,
        "+KP (.5/.2/.3)": (0.5 * pm["r_xa"] + 0.2 * pm["r_assists"] + 0.3 * pm["r_key_passes"] * conv) * emins * af7,
    }
    print(f"\n===== ASSIST + key passes (conv={conv:.3f}) =====")
    base_ll = None
    for name, mu in variants.items():
        p = p_ge(mu[test_mask], 1)
        ll = bll(y, p)
        if base_ll is None:
            base_ll = ll
        folds = " ".join(f"{s[-2:]}{'+' if bll(y[seas==s], p[seas==s]) < base_ll else ''}"
                         for s in bp.TEST_SEASONS) if name != "deployed (.7xa+.3as)" else ""
        print(f"  {name:22s}: LL {ll:.4f} (d {ll-base_ll:+.4f})", flush=True)


if __name__ == "__main__":
    main()
