from __future__ import annotations

"""TEAM-side prop lines (team corners O3.5/4.5/5.5, team cards O1.5/2.5,
team shots O11.5/13.5) — the per-side lambdas already computed by the deployed
recipe, now validated as their own market.

Honest baseline: league mean of the SIDE (home teams take more corners than
away teams — the baseline must know home/away + league). NB with side-level
dispersion. Usual folds. Deployed ASYM feature set.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
W = (5, 10, 19)
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
MARKETS = {
    "corners_side": ("home_corners", "away_corners", [3.5, 4.5, 5.5]),
    "yellows_side": ("home_yellow_cards", "away_yellow_cards", [1.5, 2.5]),
    "shots_side":   ("home_shots", "away_shots", [9.5, 11.5, 13.5]),
}


def prob_over(lam, line, disp):
    k = int(np.floor(line))
    lam = np.clip(lam, 0.05, 30.0)
    if disp > 1.05:
        r = lam / (disp - 1.0)
        return 1.0 - nbinom.cdf(k, r, 1.0 / disp)
    return 1.0 - poisson.cdf(k, lam)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ece(y, p):
    q = pd.qcut(p, 8, duplicates="drop")
    cal = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).agg(
        n=("y", "size"), pred=("p", "mean"), emp=("y", "mean"))
    return float((cal.n / cal.n.sum() * (cal.pred - cal.emp).abs()).sum())


def main() -> None:
    df = pd.read_csv(FOUND, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["season"] >= "2014-2015"]

    for market, (hc, ac, lines) in MARKETS.items():
        t0 = time.time()
        m = df.dropna(subset=[hc, ac, "home_goals", "away_goals", "date"]).copy()
        for c in [hc, ac, "home_goals", "away_goals"]:
            m[c] = pd.to_numeric(m[c], errors="coerce")
        m = m.dropna(subset=[hc, ac, "home_goals", "away_goals"])
        rows = []
        for r in m.itertuples(index=False):
            rows.append(dict(match_id=r.match_id, date=r.date, season=r.season, comp=r.competition,
                             team=r.home_team, opp=r.away_team, is_home=1,
                             ev_for=getattr(r, hc), ev_against=getattr(r, ac),
                             gf=r.home_goals, ga=r.away_goals))
            rows.append(dict(match_id=r.match_id, date=r.date, season=r.season, comp=r.competition,
                             team=r.away_team, opp=r.home_team, is_home=0,
                             ev_for=getattr(r, ac), ev_against=getattr(r, hc),
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

        res = {ln: {"m": [], "b": [], "yp": []} for ln in lines}
        for s in TEST_SEASONS:
            te_rows = lr[(lr.season == s)]
            if len(te_rows) == 0:
                continue
            s_start = te_rows.date.min()
            tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
            if len(tr) < 2000:
                continue
            reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
            te = te_rows.dropna(subset=feats).copy()
            te["pred"] = np.clip(reg.predict(te[feats]), 0.05, 30)
            # side-level dispersion + league x side baseline from train
            trs = lr[lr.date < s_start]
            disp = float(np.clip(trs["ev_for"].var() / max(trs["ev_for"].mean(), 1e-9), 0.9, 3.0))
            side_mean = trs.groupby(["comp", "is_home"])["ev_for"].mean()
            base = te.apply(lambda r: side_mean.get((r["comp"], r["is_home"]), trs["ev_for"].mean()), axis=1)
            y_act = te["ev_for"].astype(float)
            for ln in lines:
                y = (y_act > ln).astype(float).to_numpy()
                pmod = prob_over(te["pred"].to_numpy(), ln, disp)
                pbase = prob_over(base.to_numpy(), ln, disp)
                res[ln]["m"].append((bll(y, pmod), len(y), s))
                res[ln]["b"].append((bll(y, pbase), len(y), s))
                res[ln]["yp"].append((y, pmod))
        pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
        print(f"\n===== {market.upper()} ({time.time()-t0:.0f}s, disp {disp:.2f}) =====")
        for ln in lines:
            folds = " ".join(f"{s_[2][-2:]}{'+' if s_[0] < b_[0] else '-'}"
                             for s_, b_ in zip(res[ln]["m"], res[ln]["b"]))
            yy = np.concatenate([y for y, _ in res[ln]["yp"]])
            pp = np.concatenate([p for _, p in res[ln]["yp"]])
            print(f"  O {ln}: LL {pool(res[ln]['m']):.4f} vs side-base {pool(res[ln]['b']):.4f} "
                  f"delta {pool(res[ln]['m'])-pool(res[ln]['b']):+.4f} | ECE {ece(yy, pp):.4f}  [{folds}]",
                  flush=True)


if __name__ == "__main__":
    main()
