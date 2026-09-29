from __future__ import annotations

"""Player props: is in-season ESPN data worth feeding back into the player state?

Understat, the player model's only training source, stopped on 2026-05-24. So in
2026/27 production prices every player on his state at the END of last season:
career rates, last-15 form, E[min | plays], and the "likely XI" shortlist all
freeze in May. ESPN (espn_player_match_current.csv) has every 2026/27 matchday
squad with goals, shots, assists, cards and starter/sub — but no minutes and
no xG.

Replayed on Understat history, per test season (state at season start = what
production has without fresh data):
  FROZEN  every player's state frozen at the season's first match
  ESPN    state updated walk-forward with ESPN-grade rows for the season:
          minutes estimated from the role (the player's pre-season mean as a
          starter / as a sub, else the position's), xG = shots x his
          pre-season xG per shot, xA = none (his pre-season rate carries on)
  IDEAL   the full Understat walk-forward (true minutes and xG), the ceiling

Scored (1) prop log-loss on appearances, recipe as deployed; (2) the morning
shortlist: of the 11 players per team we would publish, how many played.
Shortlist ranks: FROZEN by pre-season E[min]; ESPN by the share of the team's
last 5 matches the player STARTED (ties by E[min]) — what ESPN data can tell.
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

STATS = ["xg", "goals", "shots", "xa", "assists", "yellow_cards"]
RECENT_W = {"xg": 0.5, "goals": 0.5, "shots": 0.5, "xa": 0.0, "assists": 0.0, "yellow_cards": 0.0}


def p_ge(mu, k, disp=1.0):
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, 10)
    if disp > 1.05:
        r = mu / (disp - 1.0)
        return 1 - nbinom.cdf(k - 1, r, 1.0 / disp)
    return 1 - poisson.cdf(k - 1, mu)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def state_features(pm: pd.DataFrame, minutes: str, stat_cols: dict[str, str]) -> pd.DataFrame:
    """Walk-forward per-player state (strictly before each row), the harness's
    semantics: career sums, last-15 sums, last-10-played minutes."""
    pm = pm.sort_values(["player_id", "date", "game_id"])
    g = pm.groupby("player_id", sort=False)
    out = pd.DataFrame(index=pm.index)
    sh_min = g[minutes].shift(1)
    gm = sh_min.groupby(pm["player_id"], sort=False)
    out["cmin"] = gm.cumsum().fillna(0.0)
    out["rmin15"] = gm.rolling(15, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0.0)
    for c, src in stat_cols.items():
        sh = g[src].shift(1)
        gs = sh.groupby(pm["player_id"], sort=False)
        out[f"c_{c}"] = gs.cumsum().fillna(0.0)
        out[f"rr_{c}"] = gs.rolling(15, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0.0)
    played = pm[minutes].where(pm[minutes] > 0)
    shp = played.groupby(pm["player_id"], sort=False).shift(1)
    gp = shp.groupby(pm["player_id"], sort=False)
    out["avg_minp10"] = gp.rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    out["nplayed10"] = gp.rolling(10, min_periods=1).count().reset_index(level=0, drop=True)
    return out


def mus(pm: pd.DataFrame, st: pd.DataFrame, pri, glob, pos_min) -> dict:
    """Deployed recipe on a given state."""
    r = {}
    for c in STATS:
        prior = pm["pgroup"].map(pri[c]).fillna(glob[c])
        r_car = bp.shrunk_rate(st[f"c_{c}"], st["cmin"], prior)
        if RECENT_W[c] > 0:
            r_rec = bp.shrunk_rate(st[f"rr_{c}"], st["rmin15"], r_car, k=450.0)
            r[c] = 0.5 * r_rec + 0.5 * r_car
        else:
            r[c] = r_car
    prior_min = pm["pgroup"].map(pos_min).fillna(float(np.mean(list(pos_min.values))))
    cred = st["nplayed10"].fillna(0) / (st["nplayed10"].fillna(0) + 3.0)
    exp_min = cred * st["avg_minp10"].fillna(prior_min) + (1 - cred) * prior_min
    em = exp_min.clip(20, 95) / 90.0
    af = pm["atk_factor"].fillna(1.0) ** 0.7
    return {"goal": (0.7 * r["xg"] + 0.3 * r["goals"]) * em * af, "shots": r["shots"] * em * af,
            "ass": (0.7 * r["xa"] + 0.3 * r["assists"]) * em * af, "yc": r["yellow_cards"] * em ** 0.7,
            "exp_min": exp_min}


def main() -> None:
    t0 = time.time()
    pm = bp.add_context(bp.build_panel())
    pm = pm.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)
    print(f"panel ready ({time.time() - t0:.0f}s)", flush=True)
    played = pm["minutes"] > 0
    train = pm[(~pm["season"].isin(bp.TEST_SEASONS)) & played]
    pri = train.groupby("pgroup").apply(
        lambda gr: pd.Series({c: gr[c].sum() / max(gr["minutes"].sum(), 1) * 90.0 for c in STATS}),
        include_groups=False)
    glob = {c: train[c].sum() / max(train["minutes"].sum(), 1) * 90.0 for c in STATS}
    pos_min = train.groupby("pgroup")["minutes"].mean()
    pos_st = train[train.started == 1].groupby("pgroup")["minutes"].mean()
    pos_sb = train[(train.started == 0) & (train.minutes > 0)].groupby("pgroup")["minutes"].mean()
    xg_per_shot = float(train["xg"].sum() / max(train["shots"].sum(), 1))

    ideal = state_features(pm, "minutes", {c: c for c in STATS})
    props = {"anytime": ("goal", 1, 1.0, "goals", 1), "shots>1.5": ("shots", 2, 1.3, "shots", 2),
             "shots>2.5": ("shots", 3, 1.3, "shots", 3), "assist": ("ass", 1, 1.0, "assists", 1),
             "yellow": ("yc", 1, 1.0, "yellow_cards", 1)}
    rows_ll, rows_sl = [], []
    for s in bp.TEST_SEASONS:
        in_s = (pm["season"] == s).to_numpy()
        s0 = pm.loc[in_s, "date"].min()
        pre = pm["date"] < s0
        # pre-season per-player role minutes and xG per shot (ESPN arm's estimates)
        hist = pm[pre & (pm["minutes"] > 0)]
        st_min = hist[hist.started == 1].groupby("player_id")["minutes"].mean()
        sb_min = hist[hist.started == 0].groupby("player_id")["minutes"].mean()
        xps = (hist.groupby("player_id")["xg"].sum()
               / hist.groupby("player_id")["shots"].sum().replace(0, np.nan)).clip(0.02, 0.5)
        n_sh = hist.groupby("player_id")["shots"].sum()
        xps = ((xps * n_sh + xg_per_shot * 20) / (n_sh + 20)).fillna(xg_per_shot)  # shrunk
        # pre-season xA per 90: ESPN carries no chance creation, so it just carries on
        xa90 = (hist.groupby("player_id")["xa"].sum() / hist.groupby("player_id")["minutes"].sum() * 90)

        view = pm[pre | in_s].copy()
        cur = view["season"] == s
        est_st = view["player_id"].map(st_min).fillna(view["pgroup"].map(pos_st)).fillna(83.0)
        est_sb = view["player_id"].map(sb_min).fillna(view["pgroup"].map(pos_sb)).fillna(24.0)
        est = np.where(view["minutes"] > 0, np.where(view["started"] == 1, est_st, est_sb), 0.0)
        view["min_e"] = np.where(cur, est, view["minutes"])
        view["xg_e"] = np.where(cur, view["shots"] * view["player_id"].map(xps).fillna(xg_per_shot),
                                view["xg"])
        view["xa_e"] = np.where(cur, view["player_id"].map(xa90).fillna(view["pgroup"].map(pri["xa"]))
                                * view["min_e"] / 90.0, view["xa"])
        espn = state_features(view, "min_e", {"xg": "xg_e", "goals": "goals", "shots": "shots",
                                              "xa": "xa_e", "assists": "assists",
                                              "yellow_cards": "yellow_cards"})
        # ESPN_MIN: the same, but with the TRUE minutes (what parsing ESPN's
        # substitution commentary would give) -- measures what real minutes add
        view["xa_m"] = np.where(cur, view["player_id"].map(xa90).fillna(view["pgroup"].map(pri["xa"]))
                                * view["minutes"] / 90.0, view["xa"])
        espn_min = state_features(view, "minutes", {"xg": "xg_e", "goals": "goals", "shots": "shots",
                                                    "xa": "xa_m", "assists": "assists",
                                                    "yellow_cards": "yellow_cards"})
        # FROZEN: each player's state at his first row of the season
        first = view[cur].groupby("player_id").head(1)
        fz = ideal.loc[first.index].assign(player_id=first["player_id"].to_numpy()).set_index("player_id")
        rows_cur = view[cur]
        frozen = fz.reindex(rows_cur["player_id"]).set_axis(rows_cur.index)

        pmc = pm.loc[rows_cur.index]
        arms = {"FROZEN": mus(pmc, frozen, pri, glob, pos_min),
                "ESPN": mus(pmc, espn.loc[rows_cur.index], pri, glob, pos_min),
                "ESPN_MIN": mus(pmc, espn_min.loc[rows_cur.index], pri, glob, pos_min),
                "IDEAL": mus(pmc, ideal.loc[rows_cur.index], pri, glob, pos_min)}
        test = (pmc["minutes"] > 0) & pmc["team_lam"].notna()
        for name, (mu, k, d, col, thr) in props.items():
            y = (pmc.loc[test, col] >= thr).astype(float).to_numpy()
            for arm, m in arms.items():
                rows_ll.append((s, name, arm, bll(y, p_ge(m[mu][test].to_numpy(), k, d)), len(y)))

        # shortlist: per team-match, the 11 we would publish, among players who
        # featured for that team this season (the registered squad proxy)
        tm = rows_cur[["team", "game_id", "tgn", "date"]].drop_duplicates(["team", "game_id"])
        squad = rows_cur.groupby("team")["player_id"].unique()
        starts = rows_cur[["team", "tgn", "player_id", "started", "minutes"]]
        started_at = {(r.team, r.tgn, r.player_id): (r.started == 1 and r.minutes > 0, r.minutes > 0)
                      for r in starts.itertuples(index=False)}
        fz_min = arms["FROZEN"]["exp_min"].groupby(rows_cur["player_id"]).first()
        es_min = pd.Series(arms["ESPN"]["exp_min"].to_numpy(), index=rows_cur.index)
        team_tgns = rows_cur.groupby("team")["tgn"].unique()
        sample = tm.sample(min(len(tm), 1500), random_state=0)
        for r in sample.itertuples(index=False):
            cands = squad[r.team]
            prev = sorted(t for t in team_tgns[r.team] if t < r.tgn)[-5:]
            if len(prev) < 3:
                continue
            share = {p: np.mean([started_at.get((r.team, t, p), (False, False))[0] for t in prev])
                     for p in cands}
            # ESPN arm E[min]: latest estimate before this match for each candidate
            last_rows = rows_cur[(rows_cur.team == r.team) & (rows_cur.tgn < r.tgn)]
            es = es_min.loc[last_rows.index].groupby(last_rows["player_id"]).last()
            top_f = sorted(cands, key=lambda p: -fz_min.get(p, 0))[:11]
            top_e = sorted(cands, key=lambda p: (-share[p], -es.get(p, fz_min.get(p, 0))))[:11]
            for arm, top in (("FROZEN", top_f), ("ESPN", top_e)):
                got = [started_at.get((r.team, r.tgn, p), (False, False)) for p in top]
                rows_sl.append((s, arm, np.mean([g[1] for g in got]), np.mean([g[0] for g in got])))
        print(f"  {s} done ({time.time() - t0:.0f}s)", flush=True)

    ll = pd.DataFrame(rows_ll, columns=["season", "prop", "arm", "ll", "n"])
    print("\nlog-loss vs FROZEN (production today); negative = better")
    for prop, g in ll.groupby("prop", sort=False):
        piv = g.pivot(index="season", columns="arm", values="ll")
        n = g.groupby("season")["n"].first()
        pool = {a: float((piv[a] * n).sum() / n.sum()) for a in piv.columns}
        wins = int((piv["ESPN"] < piv["FROZEN"]).sum())
        wins_m = int((piv["ESPN_MIN"] < piv["ESPN"]).sum())
        print(f"  {prop:9s} FROZEN {pool['FROZEN']:.5f} | ESPN {pool['ESPN'] - pool['FROZEN']:+.5f} "
              f"({wins}/5) | ESPN_MIN {pool['ESPN_MIN'] - pool['FROZEN']:+.5f} "
              f"(vs ESPN {pool['ESPN_MIN'] - pool['ESPN']:+.5f}, {wins_m}/5) | IDEAL {pool['IDEAL'] - pool['FROZEN']:+.5f}")
    sl = pd.DataFrame(rows_sl, columns=["season", "arm", "played", "started"])
    print("\nshortlist of 11: share who played / started")
    print(sl.groupby("arm")[["played", "started"]].mean().round(3).to_string())
    print(sl.groupby(["season", "arm"])["played"].mean().unstack().round(3).to_string())
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
