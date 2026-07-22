from __future__ import annotations

"""Micro-test: renormalize player SHOTS mus so the roster's expected sum matches
the team-side shots lambda from the (validated) team model.

scale(match, team) = team_side_pred / sum_roster(mu_shots_j * p_play_j),
p_play_j = apps10_j/10. Player mu'_shots = mu_shots * clip(scale, 0.7, 1.4).
Evaluate shots O1.5/O2.5 log-loss on the usual test appearances, before/after.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
CANON = ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv"
W = (5, 10, 19)
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
SHOTS_DISP = 1.3


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def p_ge(mu, k, disp=SHOTS_DISP):
    mu = np.clip(np.asarray(mu, float), 1e-6, 10)
    r = mu / (disp - 1.0)
    return 1 - nbinom.cdf(k - 1, r, 1.0 / disp)


def team_side_preds() -> pd.DataFrame:
    """Walk-forward team-side SHOTS lambdas per (match_id, is_home) for test seasons."""
    df = pd.read_csv(FOUND, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["season"] >= "2014-2015"]
    m = df.dropna(subset=["home_shots", "away_shots", "home_goals", "away_goals", "date"]).copy()
    for c in ["home_shots", "away_shots", "home_goals", "away_goals"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["home_shots", "away_shots", "home_goals", "away_goals"])
    rows = []
    for r in m.itertuples(index=False):
        rows.append(dict(match_id=r.match_id, date=r.date, season=r.season, team=r.home_team,
                         opp=r.away_team, is_home=1, ev_for=r.home_shots, ev_against=r.away_shots,
                         gf=r.home_goals, ga=r.away_goals))
        rows.append(dict(match_id=r.match_id, date=r.date, season=r.season, team=r.away_team,
                         opp=r.home_team, is_home=0, ev_for=r.away_shots, ev_against=r.home_shots,
                         gf=r.away_goals, ga=r.home_goals))
    lr = pd.DataFrame(rows).sort_values(["team", "date", "match_id"])
    for col in ["ev_for", "ev_against", "gf", "ga"]:
        for w in W:
            lr[f"{col}_r{w}"] = (lr.groupby("team", group_keys=False)[col]
                                 .apply(lambda s: s.shift(1).rolling(w, min_periods=3).mean()))
        lr[f"{col}_ewm"] = (lr.groupby("team", group_keys=False)[col]
                            .apply(lambda s: s.shift(1).ewm(halflife=5, min_periods=3).mean()))
    opp_src = [f"ev_against_r{w}" for w in W] + ["ev_against_ewm", "gf_ewm", "ga_ewm"]
    opp = lr[["match_id", "team"] + opp_src].rename(
        columns={"team": "opp", **{c: f"opp_{c}" for c in opp_src}})
    lr = lr.merge(opp, on=["match_id", "opp"], how="left")
    lr["delta"] = (lr["gf_ewm"] + lr["opp_ga_ewm"]) / 2 - (lr["opp_gf_ewm"] + lr["ga_ewm"]) / 2
    lr["abs_delta"] = lr["delta"].abs()
    feats = ([f"ev_for_r{w}" for w in W] + ["ev_for_ewm"]
             + [f"opp_ev_against_r{w}" for w in W] + ["opp_ev_against_ewm"]
             + ["is_home", "delta", "abs_delta"])
    out = []
    for s in TEST_SEASONS:
        te_rows = lr[lr.season == s]
        if len(te_rows) == 0:
            continue
        tr = lr[lr.date < te_rows.date.min()].dropna(subset=feats + ["ev_for"])
        reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
        te = te_rows.dropna(subset=feats).copy()
        te["team_shots_pred"] = np.clip(reg.predict(te[feats]), 1, 30)
        out.append(te[["match_id", "team", "is_home", "team_shots_pred"]])
    return pd.concat(out, ignore_index=True)

def main() -> None:
    t0 = time.time()
    # 1) player panel with mus (reuse the harness pipeline)
    import backtest_player_props as bp
    pm = bp.build_panel()
    pm = bp.add_context(pm)
    played = pm["minutes"] > 0
    train_mask = (~pm["season"].isin(bp.TEST_SEASONS)) & played
    train = pm[train_mask]
    pri = train.groupby("pgroup").apply(
        lambda gr: pd.Series({c: gr[c].sum() / max(gr["minutes"].sum(), 1) * 90.0
                              for c in ["xg", "goals", "shots", "xa", "assists", "yellow_cards"]}),
        include_groups=False)
    glob = {c: train[c].sum() / max(train["minutes"].sum(), 1) * 90.0
            for c in ["xg", "goals", "shots", "xa", "assists", "yellow_cards"]}
    prior = pm["pgroup"].map(pri["shots"]).fillna(glob["shots"])
    r_car = bp.shrunk_rate(pm["c_shots"], pm["cmin"], prior)
    r_rec = bp.shrunk_rate(pm["rr_shots"], pm["rmin15"], r_car, k=450.0)
    pm["r_shots"] = 0.5 * r_rec + 0.5 * r_car
    pos_min = train.groupby("pgroup")["minutes"].mean()
    prior_min = pm["pgroup"].map(pos_min).fillna(float(train["minutes"].mean()))
    cred_m = pm["nplayed10"].fillna(0) / (pm["nplayed10"].fillna(0) + 3.0)
    pm["exp_min"] = cred_m * pm["avg_minp10"].fillna(prior_min) + (1 - cred_m) * prior_min
    emins = pm["exp_min"].clip(20, 95) / 90.0
    af = pm["atk_factor"].fillna(1.0) ** 0.7
    pm["mu_shots"] = pm["r_shots"] * emins * af
    pm["p_play"] = (pm["apps10"].clip(upper=10) / 10.0).clip(0.05, 1.0)
    print(f"panel ready ({time.time()-t0:.0f}s)", flush=True)

    # 2) team side preds -> map foundation match_id to understat game_id
    ts = team_side_preds()
    canon = pd.read_csv(CANON, low_memory=False)[["match_id", "provider_match_id", "home_team", "away_team"]]
    ts = ts.merge(canon, on="match_id", how="inner")
    ts["game_id"] = pd.to_numeric(ts["provider_match_id"], errors="coerce")
    ts["team_fd"] = np.where(ts["is_home"] == 1, ts["home_team"], ts["away_team"])
    ts = ts.dropna(subset=["game_id"])[["game_id", "team_fd", "team_shots_pred"]]
    print(f"team side preds: {len(ts)} ({time.time()-t0:.0f}s)", flush=True)

    # 3) roster expected sum per (game, team) and scale
    pm = pm.merge(ts, on=["game_id", "team_fd"], how="left")
    grp = pm.groupby(["game_id", "team_fd"])
    pm["roster_sum"] = grp["mu_shots"].transform(lambda s: np.nan)
    pm["_contrib"] = pm["mu_shots"] * pm["p_play"]
    pm["roster_sum"] = grp["_contrib"].transform("sum")
    pm["scale"] = (pm["team_shots_pred"] / pm["roster_sum"].clip(lower=0.5)).clip(0.7, 1.4)
    pm["mu_shots_r"] = pm["mu_shots"] * pm["scale"].fillna(1.0)

    test = pm[pm["season"].isin(bp.TEST_SEASONS) & pm["team_lam"].notna() & (pm["minutes"] > 0)]
    has = test["team_shots_pred"].notna()
    print(f"test appearances: {len(test)}, with team pred: {has.sum()} ({has.mean():.0%})")
    print(f"scale stats: {test['scale'].describe().round(3).to_dict()}")
    for k, name in [(2, "SHOTS 2+"), (3, "SHOTS 3+")]:
        y = (test["shots"] >= k).astype(float).to_numpy()
        p0 = p_ge(test["mu_shots"], k)
        p1 = p_ge(test["mu_shots_r"], k)
        seas = test["season"].to_numpy()
        folds = " ".join(f"{s[-2:]}{'+' if bll(y[seas==s], p1[seas==s]) < bll(y[seas==s], p0[seas==s]) else '-'}"
                         for s in bp.TEST_SEASONS)
        print(f"{name}: base LL {bll(y, p0):.4f} -> renorm {bll(y, p1):.4f} "
              f"(d {bll(y, p1)-bll(y, p0):+.4f})  [{folds}]", flush=True)


if __name__ == "__main__":
    main()
