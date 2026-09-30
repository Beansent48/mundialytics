from __future__ import annotations

"""Shots relative to the league level: does it fix 2026/27 without costing the rest?

2026/27 moved: 28.2 shots per match against ~25 in every season before. The
deployed team-props recipe still predicts ~2 shots per match too few. The
running-residual correction (experiment_league_level.py, RES) fixed this season
but cost 2/5 normal seasons, so it stayed out (v0.56.0).

Here every count is expressed as a share of its league's level at the time:
  L        league-season side level on earlier dates, shrunk to the previous
           season with K matches (team_props._league_levels)
  REL      every match's for/against divided by its own L before the rollings;
           the Poisson model predicts that share; lambda = share x L now
In a stable season REL ~= DEP. When the whole league moves, the level passes
through with coefficient 1 by construction instead of the damped sum of
coefficients on noisy rollings.

Arms on the deployed per-market features (no Platt, no ADM on either arm):
DEP vs REL(K). Test folds 2021/22-2025/26 + the 2026/27 matches so far;
totals and team-side lines.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.props import team_props as tp  # noqa: E402

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
PREDS = [ROOT / "data/processed/enriched/understat_xg/walkforward_preds.csv",
         ROOT / "data/processed/enriched/understat_xg/walkforward_preds_hist.csv"]
TEST = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026", "2026-2027"]
KS = [int(k) for k in (sys.argv[sys.argv.index("--k") + 1].split(",") if "--k" in sys.argv else ["20", "40", "70"])]
ONLY = sys.argv[sys.argv.index("--market") + 1].split(",") if "--market" in sys.argv else ["shots", "sot", "corners"]


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def long_rows(m, hc, ac, market):
    lr = tp.TeamPropsModel._long_rows(m, hc, ac, hl=tp.EWM_HL.get(market, 5))
    lr["delta_lam"] = lr["delta_lam"].fillna(lr["delta"])
    lr["abs_delta_lam"] = lr["delta_lam"].abs()
    return lr


def main() -> None:
    lam = pd.concat([pd.read_csv(p)[["match_id", "lh", "la"]] for p in PREDS if p.exists()],
                    ignore_index=True).drop_duplicates("match_id")
    full = pd.read_csv(FOUND, low_memory=False)
    full["date"] = pd.to_datetime(full["date"], errors="coerce")
    full = full[full["season"] >= "2014-2015"].dropna(subset=["home_goals", "away_goals", "date"])
    full = full.merge(lam, on="match_id", how="left")
    full, _ = tp.TeamPropsModel._add_positions(full)
    model = tp.TeamPropsModel()
    model._use_lam = True

    for market in ONLY:
        hc, ac, lines = tp.MARKETS[market]
        s_lines = tp.SIDE_LINES.get(market, [])
        feats = model._feature_names(market)
        m = full.copy()
        for c in (hc, ac):
            m[c] = pd.to_numeric(m[c], errors="coerce")
        m = m.dropna(subset=[hc, ac])
        t0 = time.time()
        lr_dep = long_rows(m, hc, ac, market)
        arms = {"DEP": (lr_dep, None)}
        for k in KS:
            lvl, _ = tp._league_levels(m, hc, ac, k)
            mr = m.assign(L=m["match_id"].map(lvl))
            mr[hc], mr[ac] = mr[hc] / mr["L"], mr[ac] / mr["L"]
            lr = long_rows(mr, hc, ac, market)
            arms[f"REL{k}"] = (lr, lr["match_id"].map(lvl))
        res = {a: {} for a in arms}
        bias = {a: {} for a in arms}
        for s in TEST:
            te_m = m[m.season == s]
            if te_m.empty:
                continue
            s_start = te_m.date.min()
            tr_m = m[m.date < s_start]
            tt = (tr_m[hc] + tr_m[ac]).astype(float)
            disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
            sv = lr_dep.loc[lr_dep.date < s_start, "ev_for"].dropna().astype(float)
            disp_s = float(np.clip(sv.var() / max(sv.mean(), 1e-9), 0.9, 3.0))
            preds, common = {}, None
            for arm, (lr, L) in arms.items():
                tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
                reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
                te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
                pred = reg.predict(te[feats])
                if L is not None:
                    pred = pred * L.loc[te.index].to_numpy()
                te["pred"] = np.clip(pred, 0.1, 25)
                pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
                preds[arm] = pv
                common = pv.index if common is None else common.intersection(pv.index)
            tei = te_m.set_index("match_id").loc[common]
            act_h, act_a = tei[hc].to_numpy(float), tei[ac].to_numpy(float)
            for arm, pv in preds.items():
                lh, la = pv.loc[common, 1].to_numpy(), pv.loc[common, 0].to_numpy()
                res[arm][s] = {
                    **{f"O{ln}": bll((act_h + act_a > ln).astype(float), tp._prob_over(lh + la, ln, disp))
                       for ln in lines},
                    **{f"side{ln}": bll((np.r_[act_h, act_a] > ln).astype(float),
                                        tp._prob_over(np.r_[lh, la], ln, disp_s)) for ln in s_lines},
                    "n": len(common)}
                bias[arm][s] = float(((lh + la) - (act_h + act_a)).mean())

        hist = TEST[:-1]
        print(f"\n===== {market.upper()} ({time.time() - t0:.0f}s) =====")
        for arm in arms:
            if arm == "DEP":
                continue
            tot_keys = [f"O{ln}" for ln in lines]
            side_keys = [f"side{ln}" for ln in s_lines]

            def d(keys, seasons):
                num = sum((np.mean([res[arm][s][k] for k in keys]) - np.mean([res["DEP"][s][k] for k in keys]))
                          * res["DEP"][s]["n"] for s in seasons)
                return num / sum(res["DEP"][s]["n"] for s in seasons)
            wins = sum(np.mean([res[arm][s][k] - res["DEP"][s][k] for k in tot_keys]) < 0 for s in hist)
            side = f"{d(side_keys, hist):+.5f}" if side_keys else "  n/a  "
            live = d(tot_keys, ["2026-2027"])
            print(f"  {arm:6s} totals {d(tot_keys, hist):+.5f} ({wins}/5) | sides {side} | "
                  f"26/27 totals {live:+.4f} | 26/27 bias {bias['DEP']['2026-2027']:+.2f} -> "
                  f"{bias[arm]['2026-2027']:+.2f} per match")


if __name__ == "__main__":
    main()
