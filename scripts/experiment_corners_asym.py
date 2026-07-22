from __future__ import annotations

"""Corners take 2: strength-ASYMMETRY features (game-state theory).

Previous negative (experiment_corners_features): shot/xG VOLUME rollings add
nothing. Different hypothesis here: corners are driven by expected SUPREMACY
(the dominant side camps in the opponent half; trailing favorites force
corners late). Features added to the base corners recipe, per side:
  delta  = (gf_ewm_team + ga_ewm_opp)/2 - (gf_ewm_opp + ga_ewm_team)/2
  |delta| (dominance magnitude regardless of sign)
built from goals rollings (available for the full panel, walk-forward safe).
A/B on the usual folds vs the deployed base recipe.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
W = (5, 10, 19)
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
LINES = [8.5, 9.5, 10.5]


def prob_over(lam, line, disp):
    k = int(np.floor(line))
    lam = np.clip(lam, 0.2, 40.0)
    r = lam / (disp - 1.0)
    return 1.0 - nbinom.cdf(k, r, 1.0 / disp)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


ALL_MARKETS = {
    "corners": ("home_corners", "away_corners", [8.5, 9.5, 10.5]),
    "yellows": ("home_yellow_cards", "away_yellow_cards", [3.5, 4.5, 5.5]),
    "fouls":   ("home_fouls", "away_fouls", [21.5, 23.5]),
    "shots":   ("home_shots", "away_shots", [22.5, 24.5]),
    "sot":     ("home_sot", "away_sot", [7.5, 8.5]),
}


def main() -> None:
    df = pd.read_csv(FOUND, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["season"] >= "2014-2015"]
    for mk, (hcol, acol, mlines) in ALL_MARKETS.items():
        print(f"\n########## {mk.upper()} ##########", flush=True)
        run_market(df, hcol, acol, mlines)


def run_market(df: pd.DataFrame, HC: str, AC: str, mlines: list[float]) -> None:
    global LINES
    LINES = mlines
    m = df.dropna(subset=[HC, AC, "home_goals", "away_goals", "date"]).copy()
    for c in [HC, AC, "home_goals", "away_goals"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=[HC, AC, "home_goals", "away_goals"])

    rows = []
    for r in m.itertuples(index=False):
        rows.append(dict(match_id=r.match_id, date=r.date, team=r.home_team, opp=r.away_team,
                         is_home=1, ev_for=getattr(r, HC), ev_against=getattr(r, AC),
                         gf=r.home_goals, ga=r.away_goals))
        rows.append(dict(match_id=r.match_id, date=r.date, team=r.away_team, opp=r.home_team,
                         is_home=0, ev_for=getattr(r, AC), ev_against=getattr(r, HC),
                         gf=r.away_goals, ga=r.home_goals))
    lr = pd.DataFrame(rows).sort_values(["team", "date", "match_id"])
    for col in ["ev_for", "ev_against", "gf", "ga"]:
        for w in W:
            lr[f"{col}_r{w}"] = (lr.groupby("team", group_keys=False)[col]
                                 .apply(lambda s: s.shift(1).rolling(w, min_periods=3).mean()))
        lr[f"{col}_ewm"] = (lr.groupby("team", group_keys=False)[col]
                            .apply(lambda s: s.shift(1).ewm(halflife=5, min_periods=3).mean()))
    opp_cols = [f"ev_against_r{w}" for w in W] + ["ev_against_ewm", "gf_ewm", "ga_ewm"]
    opp = lr[["match_id", "team"] + opp_cols].rename(
        columns={"team": "opp", **{c: f"opp_{c}" for c in opp_cols}})
    lr = lr.merge(opp, on=["match_id", "opp"], how="left")
    # expected supremacy of THIS side
    lr["delta"] = ((lr["gf_ewm"] + lr["opp_ga_ewm"]) / 2 - (lr["opp_gf_ewm"] + lr["ga_ewm"]) / 2)
    lr["abs_delta"] = lr["delta"].abs()

    base_feats = ([f"ev_for_r{w}" for w in W] + ["ev_for_ewm"]
                  + [f"opp_ev_against_r{w}" for w in W] + ["opp_ev_against_ewm"] + ["is_home"])
    aug_feats = base_feats + ["delta", "abs_delta"]

    for tag, feats in [("BASE", base_feats), ("ASYM", aug_feats)]:
        res = {ln: {"m": [], "b": []} for ln in LINES}
        t0 = time.time()
        for s in TEST_SEASONS:
            te_m = m[m.season == s]
            s_start = te_m.date.min()
            tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
            reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
            te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
            te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 25)
            pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
            tot = pv[1] + pv[0]
            tr_tot = m[m.date < s_start]
            tt = (tr_tot[HC] + tr_tot[AC]).astype(float)
            disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 1.11, 3.0))
            lg = tr_tot.assign(tot=tt).groupby("competition")["tot"].mean()
            tei = te_m.set_index("match_id").loc[tot.index]
            base_v = tei["competition"].map(lg).fillna(float(tt.mean())).to_numpy()
            act = tei[[HC, AC]].sum(axis=1).astype(float)
            for ln in LINES:
                y = (act > ln).astype(float).to_numpy()
                res[ln]["m"].append((bll(y, prob_over(tot.to_numpy(), ln, disp)), len(y), s))
                res[ln]["b"].append((bll(y, prob_over(base_v, ln, disp)), len(y), s))
        pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
        print(f"\n{tag} ({time.time()-t0:.0f}s):")
        for ln in LINES:
            folds = " ".join(f"{s_[2][-2:]}{'+' if s_[0] < b_[0] else '-'}"
                             for s_, b_ in zip(res[ln]["m"], res[ln]["b"]))
            print(f"  O/U {ln}: LL {pool(res[ln]['m']):.4f} vs lg-base {pool(res[ln]['b']):.4f} "
                  f"delta {pool(res[ln]['m'])-pool(res[ln]['b']):+.4f}  [{folds}]", flush=True)


if __name__ == "__main__":
    main()
