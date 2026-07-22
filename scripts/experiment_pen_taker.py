from __future__ import annotations

"""Penalty-taker upgrade for anytime scorer / 2+ goals.

Data discovery 2026-07-23: penalties ARE in understat_shots.csv — soccerdata's
SHOT_SITUATIONS mapping omits "Penalty" so they carry situation=NaN (6,677 rows,
mean xG 0.756, conversion 78%). No re-download needed.

Model change: split the goal mu into non-pen + pen components:
  mu_np  = (0.7*r_npxg + 0.3*r_npgoals) * emins * af      (rates now penalty-free)
  mu_pen = taker_share * team_pen_rate * 0.78 * emins * af
  taker_share = player pens / team pens over the last ~60 squad appearances,
                credibility-shrunk (team pens / (team pens + 4)) toward 0.
Walk-forward everywhere. A/B vs the deployed v5 mu on the usual folds.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SHOTS = ROOT / "data/external/advanced/understat/understat_shots.csv"


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def p_ge(mu, k):
    return 1 - poisson.cdf(k - 1, np.clip(np.asarray(mu, float), 1e-6, 10))


def main() -> None:
    t0 = time.time()
    import backtest_player_props as bp
    pm = bp.build_panel()
    pm = bp.add_context(pm)

    # penalty attempts/goals per (player, game) from the shots file (NaN situation = Penalty)
    sh = pd.read_csv(SHOTS, usecols=["game_id", "player_id", "situation", "result"])
    pen = sh[sh["situation"].isna()].copy()
    pen["pen_goal"] = (pen["result"] == "Goal").astype(float)
    pg = pen.groupby(["game_id", "player_id"]).agg(pen_att=("result", "size"),
                                                   pen_goal=("pen_goal", "sum")).reset_index()
    pm = pm.merge(pg, on=["game_id", "player_id"], how="left")
    pm[["pen_att", "pen_goal"]] = pm[["pen_att", "pen_goal"]].fillna(0.0)
    # team pens in each game (for share denominator + team pen rate)
    tp = pm.groupby(["team", "game_id"])["pen_att"].sum().rename("team_pen_att").reset_index()
    pm = pm.merge(tp, on=["team", "game_id"], how="left")
    print(f"panel + pens ready ({time.time()-t0:.0f}s); pen rows joined: {int((pm.pen_att>0).sum())}", flush=True)

    pm = pm.sort_values(["player_id", "date", "game_id"])
    g = pm.groupby("player_id", group_keys=False)
    pm["p_pen60"] = g["pen_att"].apply(lambda s: s.shift(1).rolling(60, min_periods=1).sum()).fillna(0)
    pm["t_pen60"] = g["team_pen_att"].apply(lambda s: s.shift(1).rolling(60, min_periods=1).sum()).fillna(0)
    cred = pm["t_pen60"] / (pm["t_pen60"] + 4.0)
    pm["taker_share"] = cred * (pm["p_pen60"] / pm["t_pen60"].clip(lower=1e-9))
    # penalty-free per-row stats -> walk-forward npxg/npgoal rates
    pm["npxg"] = (pm["xg"] - 0.76 * pm["pen_att"]).clip(lower=0)
    pm["npgoals"] = (pm["goals"] - pm["pen_goal"]).clip(lower=0)
    for c in ["npxg", "npgoals"]:
        pm[f"c_{c}"] = g[c].apply(lambda s: s.shift(1).cumsum()).fillna(0.0)
        pm[f"rr_{c}"] = g[c].apply(lambda s: s.shift(1).rolling(15, min_periods=1).sum()).fillna(0.0)
    # team pen rate per game (attacking pens won), walk-forward at team level
    tg = tp.merge(pm[["team", "game_id", "date"]].drop_duplicates(), on=["team", "game_id"]).sort_values(["team", "date"])
    tg["team_pen_rate"] = (tg.groupby("team", group_keys=False)["team_pen_att"]
                           .apply(lambda s: s.shift(1).rolling(38, min_periods=5).mean()))
    pm = pm.merge(tg[["team", "game_id", "team_pen_rate"]], on=["team", "game_id"], how="left")

    played = pm["minutes"] > 0
    train = pm[(~pm["season"].isin(bp.TEST_SEASONS)) & played]
    pri = train.groupby("pgroup").apply(
        lambda gr: pd.Series({c: gr[c].sum() / max(gr["minutes"].sum(), 1) * 90.0
                              for c in ["xg", "goals", "npxg", "npgoals"]}), include_groups=False)
    glob = {c: train[c].sum() / max(train["minutes"].sum(), 1) * 90.0
            for c in ["xg", "goals", "npxg", "npgoals"]}

    def rate(c):
        prior = pm["pgroup"].map(pri[c]).fillna(glob[c])
        r_car = bp.shrunk_rate(pm[f"c_{c}"], pm["cmin"], prior)
        r_rec = bp.shrunk_rate(pm[f"rr_{c}"], pm["rmin15"], r_car, k=450.0)
        return 0.5 * r_rec + 0.5 * r_car

    for c in ["xg", "goals", "npxg", "npgoals"]:
        pm[f"r_{c}"] = rate(c)

    pos_min = train.groupby("pgroup")["minutes"].mean()
    prior_min = pm["pgroup"].map(pos_min).fillna(float(train["minutes"].mean()))
    cred_m = pm["nplayed10"].fillna(0) / (pm["nplayed10"].fillna(0) + 3.0)
    pm["exp_min"] = cred_m * pm["avg_minp10"].fillna(prior_min) + (1 - cred_m) * prior_min
    emins = pm["exp_min"].clip(20, 95) / 90.0
    af = pm["atk_factor"].fillna(1.0) ** 0.7

    mu_v5 = (0.7 * pm["r_xg"] + 0.3 * pm["r_goals"]) * emins * af
    mu_np = (0.7 * pm["r_npxg"] + 0.3 * pm["r_npgoals"]) * emins * af
    mu_pen = pm["taker_share"].fillna(0) * pm["team_pen_rate"].fillna(0.22) * 0.78 * emins * af
    mu_v7 = mu_np + mu_pen

    test_mask = pm["season"].isin(bp.TEST_SEASONS) & pm["team_lam"].notna() & played
    t = pm[test_mask]
    print(f"test appearances: {len(t)} | mean mu_pen {mu_pen[test_mask].mean():.4f} | "
          f"takers (share>0.5): {(t.taker_share > 0.5).sum()}", flush=True)
    for k, name in [(1, "ANYTIME"), (2, "2+ GOALS")]:
        y = (t["goals"] >= k).astype(float).to_numpy()
        p5, p7 = p_ge(mu_v5[test_mask], k), p_ge(mu_v7[test_mask], k)
        seas = t["season"].to_numpy()
        folds = " ".join(f"{s[-2:]}{'+' if bll(y[seas==s], p7[seas==s]) < bll(y[seas==s], p5[seas==s]) else '-'}"
                         for s in bp.TEST_SEASONS)
        print(f"{name}: v5 {bll(y, p5):.4f} -> v7-pen {bll(y, p7):.4f} "
              f"(d {bll(y, p7)-bll(y, p5):+.4f})  [{folds}]", flush=True)
    # sanity: top takers by share (latest rows)
    latest = t.sort_values("date").groupby("player_id").tail(1)
    top = latest.nlargest(8, "taker_share")[["player", "team", "taker_share", "p_pen60", "t_pen60"]]
    print("\ntop taker_share (latest):")
    print(top.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
