from __future__ import annotations

"""Player props on the confirmed lineup: redistribute the TEAM's expectation.

predict_lineup (v0.56.0) prices each player on his role tonight, but every
player is priced on his own: when the first-choice striker is out, nobody else
in the XI picks up his share. Here the confirmed matchday squad is summed —
the eleven STARTERS (Understat lists only subs who came on, so the bench is unknown) — and
compared with what the team is expected to produce:
  goals / assists  the engine's team lambda
  shots            the team's walk-forward shots per match (last 10, Understat)
Each player's mu is scaled by f = (expected / squad sum / typical ratio)^alpha.

The July attempt (renorm of player shots to the team-side lambda) failed: the
squad was guessed from apps/10, and its sum ran systematically low (median scale
1.26 at the clip), which broke calibrated mus. Two differences now: the squad
is the confirmed one, and the ratio is divided by its TRAIN median, so only a
match that is unusual for this team moves anything.

Base = LINEUP_CAL (the deployed lineup price, player_props.LINEUP_MIN_FIT).
Folds 2021/22-2025/26, scored on players who played.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import backtest_player_props as bp  # noqa: E402
from mundialytics.props.player_props import LINEUP_MIN_FIT  # noqa: E402

ALPHAS = [0.25, 0.5, 0.75, 1.0]


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
    train_rows = ~pm["season"].isin(bp.TEST_SEASONS)
    train = pm[train_rows & played]
    cols = ["xg", "goals", "shots", "xa", "assists", "yellow_cards"]
    pri = train.groupby("pgroup").apply(
        lambda gr: pd.Series({c: gr[c].sum() / max(gr["minutes"].sum(), 1) * 90.0 for c in cols}),
        include_groups=False)
    glob = {c: train[c].sum() / max(train["minutes"].sum(), 1) * 90.0 for c in cols}
    r = {}
    for c in cols:
        prior = pm["pgroup"].map(pri[c]).fillna(glob[c])
        r_car = bp.shrunk_rate(pm[f"c_{c}"], pm["cmin"], prior)
        if c in ("xg", "goals", "shots"):
            r[c] = 0.5 * bp.shrunk_rate(pm[f"rr_{c}"], pm["rmin15"], r_car, k=450.0) + 0.5 * r_car
        else:
            r[c] = r_car

    # role minutes (as predict_lineup), all squad rows incl. unused bench
    st = train[train.started == 1].groupby("pgroup")["minutes"].mean()
    sb = train[(train.started == 0) & (train.minutes > 0)].groupby("pgroup")["minutes"].mean()
    g2 = pm.sort_values(["player_id", "team", "tgn"]).groupby(["player_id", "team"])
    n_st = g2["min_start"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).count()).fillna(0)
    n_sb = g2["min_sub"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).count()).fillna(0)
    pr_st, pr_sb = pm["pgroup"].map(st).fillna(83.0), pm["pgroup"].map(sb).fillna(24.0)
    m_st = (n_st / (n_st + 3)) * pm["avg_min_start10"].fillna(pr_st) + (3 / (n_st + 3)) * pr_st
    m_sb = (n_sb / (n_sb + 3)) * pm["avg_min_sub10"].fillna(pr_sb) + (3 / (n_sb + 3)) * pr_sb
    em = pd.Series(np.where(pm["started"] == 1, m_st, m_sb), index=pm.index).clip(5, 95) / 90.0
    af = pm["atk_factor"].fillna(1.0) ** 0.7
    (gg, ag), (gs, as_), (ga, aa) = LINEUP_MIN_FIT["goal"], LINEUP_MIN_FIT["shots"], LINEUP_MIN_FIT["ass"]
    pm["mu_goal"] = ag * (0.7 * r["xg"] + 0.3 * r["goals"]) * em ** gg * af
    pm["mu_shots"] = as_ * r["shots"] * em ** gs * af
    pm["mu_ass"] = aa * (0.7 * r["xa"] + 0.3 * r["assists"]) * em ** ga * af

    # squad sums: starters + q x bench (q from train: share of bench who come on)
    bench = train_rows & (pm["started"] == 0)
    q = float((pm.loc[bench, "minutes"] > 0).mean())
    # Understat lists only the subs who CAME ON (q comes out 1.0), so summing
    # the bench would use who enters -- unknown before kick-off. Starters only:
    # the XI is what the lineup tells us an hour before.
    w = np.where(pm["started"] == 1, 1.0, 0.0)
    key = ["game_id", "team"]
    for mu in ("mu_goal", "mu_shots", "mu_ass"):
        pm[f"sum_{mu}"] = (pm[mu] * w).groupby([pm[k] for k in key]).transform("sum")
    # team expectations: engine lambda (goals, assists ~ goals) and the team's
    # walk-forward shots per match over its last 10 games
    tsh = (pm.groupby(key + ["date"], as_index=False)["shots"].sum().sort_values(["team", "date"]))
    tsh["team_shots_exp"] = (tsh.groupby("team")["shots"]
                             .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean()))
    pm = pm.merge(tsh[key + ["team_shots_exp"]], on=key, how="left")
    exp = {"mu_goal": pm["team_lam"], "mu_ass": pm["team_lam"], "mu_shots": pm["team_shots_exp"]}
    ratio = {mu: exp[mu] / pm[f"sum_{mu}"] for mu in exp}
    med = {mu: float(ratio[mu][train_rows.to_numpy() & ratio[mu].notna().to_numpy()].median()) for mu in exp}
    print(f"q (bench who come on) {q:.3f}; train median ratio "
          + ", ".join(f"{k} {v:.3f}" for k, v in med.items()), flush=True)

    test = pm["season"].isin(bp.TEST_SEASONS) & (pm["minutes"] > 0) & pm["team_lam"].notna()
    for mu in exp:
        test &= ratio[mu].notna()
    t = pm[test]
    seas = t["season"].to_numpy()
    props = {"anytime": ("mu_goal", 1, 1.0, t["goals"] >= 1), "2+ goals": ("mu_goal", 2, 1.0, t["goals"] >= 2),
             "shots>1.5": ("mu_shots", 2, 1.3, t["shots"] >= 2), "shots>2.5": ("mu_shots", 3, 1.3, t["shots"] >= 3),
             "assist": ("mu_ass", 1, 1.0, t["assists"] >= 1)}
    print(f"\ntest appearances {len(t)}; deltas vs LINEUP_CAL (negative = better)")
    for name, (mu, k, d, yser) in props.items():
        y = yser.astype(float).to_numpy()
        base = t[mu].to_numpy()
        p0 = p_ge(base, k, d)
        line = [f"{name:9s} base {bll(y, p0):.5f}"]
        rel = (ratio[mu][test] / med[mu]).clip(0.5, 2.0).to_numpy()
        for a in ALPHAS:
            p1 = p_ge(base * rel ** a, k, d)
            wins = sum(bll(y[seas == s], p1[seas == s]) < bll(y[seas == s], p0[seas == s]) for s in bp.TEST_SEASONS)
            line.append(f"a{a:g} {bll(y, p1) - bll(y, p0):+.5f} ({wins}/5)")
        print("  " + " | ".join(line), flush=True)
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
