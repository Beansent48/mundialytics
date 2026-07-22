from __future__ import annotations

"""Micro-test: per-LEAGUE NB dispersion vs global dispersion for team-prop
totals. Same models (deployed ASYM recipe), only the dispersion used in the
O/U probability changes. If Serie A fouls disperse differently than EPL fouls,
per-league tails should calibrate better."""

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
    "corners": ("home_corners", "away_corners", [8.5, 9.5, 10.5]),
    "yellows": ("home_yellow_cards", "away_yellow_cards", [3.5, 4.5, 5.5]),
    "fouls":   ("home_fouls", "away_fouls", [21.5, 23.5]),
    "shots":   ("home_shots", "away_shots", [22.5, 24.5]),
    "sot":     ("home_sot", "away_sot", [7.5, 8.5]),
}


def prob_over(lam, line, disp):
    k = int(np.floor(line))
    lam = np.clip(lam, 0.2, 40.0)
    disp = np.maximum(np.asarray(disp, float), 1.001)
    r = lam / (disp - 1.0)
    return 1.0 - nbinom.cdf(k, r, 1.0 / disp)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


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
            rows.append(dict(match_id=r.match_id, date=r.date, team=r.home_team, opp=r.away_team,
                             is_home=1, ev_for=getattr(r, hc), ev_against=getattr(r, ac),
                             gf=r.home_goals, ga=r.away_goals))
            rows.append(dict(match_id=r.match_id, date=r.date, team=r.away_team, opp=r.home_team,
                             is_home=0, ev_for=getattr(r, ac), ev_against=getattr(r, hc),
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

        res = {ln: {"g": [], "l": []} for ln in lines}
        disp_report = None
        for s in TEST_SEASONS:
            te_m = m[m.season == s]
            if len(te_m) == 0:
                continue
            s_start = te_m.date.min()
            tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
            if len(tr) < 2000:
                continue
            reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
            te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
            te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 25)
            pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
            tot = pv[1] + pv[0]
            tr_tot = m[m.date < s_start]
            tt = (tr_tot[hc] + tr_tot[ac]).astype(float)
            disp_g = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 1.02, 3.0))
            lg_disp = (tr_tot.assign(t=tt).groupby("competition")["t"]
                       .apply(lambda s2: np.clip(s2.var() / max(s2.mean(), 1e-9), 1.02, 3.0)))
            disp_report = lg_disp.round(2).to_dict()
            tei = te_m.set_index("match_id").loc[tot.index]
            disp_l = tei["competition"].map(lg_disp).fillna(disp_g).to_numpy()
            act = tei[[hc, ac]].sum(axis=1).astype(float)
            for ln in lines:
                y = (act > ln).astype(float).to_numpy()
                res[ln]["g"].append((bll(y, prob_over(tot.to_numpy(), ln, disp_g)), len(y), s))
                res[ln]["l"].append((bll(y, prob_over(tot.to_numpy(), ln, disp_l)), len(y), s))
        pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
        print(f"\n===== {market.upper()} ({time.time()-t0:.0f}s)  per-league disp {disp_report} =====")
        for ln in lines:
            folds = " ".join(f"{g[2][-2:]}{'+' if l[0] < g[0] else '-'}"
                             for g, l in zip(res[ln]["g"], res[ln]["l"]))
            print(f"  O/U {ln}: global-disp LL {pool(res[ln]['g']):.4f} | per-league {pool(res[ln]['l']):.4f} "
                  f"delta {pool(res[ln]['l'])-pool(res[ln]['g']):+.4f}  [{folds}]", flush=True)


if __name__ == "__main__":
    main()
