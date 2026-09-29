from __future__ import annotations

"""Player props once the XI is out: does knowing who STARTS sharpen them?

The props are priced at 06:30 with E[min | plays] — one number that averages a
player's starts and his cameos (~80' vs ~20'). About an hour before kick-off
ESPN publishes both XIs (scripts/log_lineup_pass.py already reads them for the
1X2), so the lineup pass could price each player on the minutes of the role he
actually has tonight.

An older test (2026-07-22) found "bimodal minutes" identical to the average —
but that one weighted the two modes by the player's HISTORICAL start share. Here
the start/bench status is the confirmed one (Understat's position == 'Sub' marks
the substitutes, which is exactly what the published lineup tells us).

  DEP     deployed: last-10-played minutes shrunk to the position mean
  LINEUP  starters: last-10 minutes AS A STARTER shrunk to the position's
          starter mean; subs: the same over sub appearances

Scored on appearances (props are conditioned on playing), folds 2021/22-2025/26,
plus the ranking the app publishes: in how many team-matches with a goal was a
real scorer in our top 3 / top 1.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import backtest_player_props as bp  # noqa: E402


def p_ge(mu, k, disp=1.0):
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, 10)
    if disp > 1.05:
        r = mu / (disp - 1.0)
        return 1 - nbinom.cdf(k - 1, r, 1.0 / disp)
    return 1 - poisson.cdf(k - 1, mu)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    t0 = time.time()
    pm = bp.add_context(bp.build_panel())
    print(f"panel ready ({time.time() - t0:.0f}s)", flush=True)
    played = pm["minutes"] > 0
    train = pm[(~pm["season"].isin(bp.TEST_SEASONS)) & played]
    test_mask = pm["season"].isin(bp.TEST_SEASONS) & pm["team_lam"].notna() & played

    cols = ["xg", "goals", "shots", "xa", "assists", "yellow_cards"]
    pri = train.groupby("pgroup").apply(
        lambda gr: pd.Series({c: gr[c].sum() / max(gr["minutes"].sum(), 1) * 90.0 for c in cols}),
        include_groups=False)
    glob = {c: train[c].sum() / max(train["minutes"].sum(), 1) * 90.0 for c in cols}
    recent_w = {"xg": 0.5, "goals": 0.5, "shots": 0.5, "xa": 0.0, "assists": 0.0, "yellow_cards": 0.0}
    for c in cols:
        prior = pm["pgroup"].map(pri[c]).fillna(glob[c])
        r_car = bp.shrunk_rate(pm[f"c_{c}"], pm["cmin"], prior)
        if recent_w[c] > 0:
            r_rec = bp.shrunk_rate(pm[f"rr_{c}"], pm["rmin15"], r_car, k=450.0)
            pm[f"r_{c}"] = 0.5 * r_rec + 0.5 * r_car
        else:
            pm[f"r_{c}"] = r_car

    pos_min = train.groupby("pgroup")["minutes"].mean()
    prior_min = pm["pgroup"].map(pos_min).fillna(float(train["minutes"].mean()))
    cred = pm["nplayed10"].fillna(0) / (pm["nplayed10"].fillna(0) + 3.0)
    dep_min = cred * pm["avg_minp10"].fillna(prior_min) + (1 - cred) * prior_min

    # role-conditional minutes, each shrunk to its own position-role mean
    st = train[train.started == 1].groupby("pgroup")["minutes"].mean()
    sb = train[(train.started == 0) & (train.minutes > 0)].groupby("pgroup")["minutes"].mean()
    g2 = pm.sort_values(["player_id", "team", "tgn"]).groupby(["player_id", "team"])
    n_st = g2["min_start"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).count()).fillna(0)
    n_sb = g2["min_sub"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).count()).fillna(0)
    pr_st = pm["pgroup"].map(st).fillna(83.0)
    pr_sb = pm["pgroup"].map(sb).fillna(24.0)
    c_st, c_sb = n_st / (n_st + 3.0), n_sb / (n_sb + 3.0)
    min_if_start = c_st * pm["avg_min_start10"].fillna(pr_st) + (1 - c_st) * pr_st
    min_if_sub = c_sb * pm["avg_min_sub10"].fillna(pr_sb) + (1 - c_sb) * pr_sb
    lin_min = np.where(pm["started"] == 1, min_if_start, min_if_sub)

    af = pm["atk_factor"].fillna(1.0) ** 0.7
    arms = {}
    for arm, mins in [("DEP", dep_min), ("LINEUP", pd.Series(lin_min, index=pm.index))]:
        em = mins.clip(5, 95) / 90.0 if arm == "LINEUP" else mins.clip(20, 95) / 90.0
        arms[arm] = {
            "goal": (0.7 * pm["r_xg"] + 0.3 * pm["r_goals"]) * em * af,
            "shots": pm["r_shots"] * em * af,
            "ass": (0.7 * pm["r_xa"] + 0.3 * pm["r_assists"]) * em * af,
            "yc": pm["r_yellow_cards"] * em ** 0.7,
        }
    # LINEUP_CAL: per-90 career rates mix starts and cameos, so rate x minutes
    # overshoots starters and undershoots subs. mu = a * r * (min/90)^g, with g
    # and a fitted per stat on TRAIN seasons only (a matches the train total).
    train_mask = (~pm["season"].isin(bp.TEST_SEASONS)) & played
    lin_em = pd.Series(lin_min, index=pm.index).clip(5, 95) / 90.0
    base_rate = {"goal": (0.7 * pm["r_xg"] + 0.3 * pm["r_goals"]) * af,
                 "shots": pm["r_shots"] * af,
                 "ass": (0.7 * pm["r_xa"] + 0.3 * pm["r_assists"]) * af,
                 "yc": pm["r_yellow_cards"]}
    actual = {"goal": pm["goals"], "shots": pm["shots"], "ass": pm["assists"], "yc": pm["yellow_cards"]}
    fit_k = {"goal": (1, 1.0), "shots": (2, 1.3), "ass": (1, 1.0), "yc": (1, 1.0)}
    arms["LINEUP_CAL"] = {}
    for key, rate in base_rate.items():
        k_, d_ = fit_k[key]
        y_tr = (actual[key][train_mask] >= k_).astype(float).to_numpy()
        best = None
        for g in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            raw = rate * lin_em ** g
            a = float(actual[key][train_mask].sum() / max(raw[train_mask].sum(), 1e-9))
            ll = bll(y_tr, p_ge((a * raw)[train_mask].to_numpy(), k_, d_))
            if best is None or ll < best[0]:
                best = (ll, g, a)
        _, g, a = best
        arms["LINEUP_CAL"][key] = a * rate * lin_em ** g
        print(f"  {key}: minutes exponent {g}, scale {a:.3f} (train-picked)", flush=True)

    test = pm[test_mask].copy()
    seas = test["season"].to_numpy()
    props = {
        "anytime":   ("goal", 1, 1.0, test["goals"] >= 1),
        "2+ goals":  ("goal", 2, 1.0, test["goals"] >= 2),
        "shots>1.5": ("shots", 2, 1.3, test["shots"] >= 2),
        "shots>2.5": ("shots", 3, 1.3, test["shots"] >= 3),
        "assist":    ("ass", 1, 1.0, test["assists"] >= 1),
        "yellow":    ("yc", 1, 1.0, test["yellow_cards"] >= 1),
    }
    print(f"\ntest appearances {len(test)}  (starters {test['started'].mean():.1%})")
    for name, (mu, k, d, yser) in props.items():
        y = yser.astype(float).to_numpy()
        p0 = p_ge(arms["DEP"][mu][test_mask].to_numpy(), k, d)
        out = [f"  {name:9s} DEP {bll(y, p0):.5f}"]
        roles = [("XI", test["started"].to_numpy() == 1), ("sub", test["started"].to_numpy() == 0)]
        for arm in ["LINEUP", "LINEUP_CAL"]:
            p1 = p_ge(arms[arm][mu][test_mask].to_numpy(), k, d)
            wins = sum(bll(y[seas == s], p1[seas == s]) < bll(y[seas == s], p0[seas == s])
                       for s in bp.TEST_SEASONS)
            cal = " ".join(f"{r} {p1[msk].mean():.3f}/{y[msk].mean():.3f}" for r, msk in roles)
            out.append(f"{arm} {bll(y, p1) - bll(y, p0):+.5f} ({wins}/5) [{cal}]")
        print(" | ".join(out), flush=True)

    # the published ranking: top-k per team-match among players who featured
    test["p_dep"] = p_ge(arms["DEP"]["goal"][test_mask].to_numpy(), 1)
    test["p_lin"] = p_ge(arms["LINEUP"]["goal"][test_mask].to_numpy(), 1)
    test["p_cal"] = p_ge(arms["LINEUP_CAL"]["goal"][test_mask].to_numpy(), 1)
    test["scored"] = test["goals"] >= 1
    grp = test.groupby(["game_id", "team"])
    scored_any = grp["scored"].transform("any")
    t = test[scored_any]
    for col in ["p_dep", "p_lin", "p_cal"]:
        rk = t.groupby(["game_id", "team"])[col].rank(ascending=False, method="first")
        top1 = t[rk <= 1].groupby(["game_id", "team"])["scored"].any().mean()
        top3 = t[rk <= 3].groupby(["game_id", "team"])["scored"].any().mean()
        hit1 = t[rk <= 1]["scored"].mean()
        print(f"  ranking {col}: top-1 contains a scorer {top1:.1%}, top-3 {top3:.1%}, "
              f"top-1 pick scores {hit1:.1%}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
