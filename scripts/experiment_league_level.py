from __future__ import annotations

"""Does a league-SEASON level term fix the event markets' in-season drift?

Why (2026-09-29): the live 2026/27 log under-predicts shots (says 63% over,
77% happen) and over-predicts cards, because the season itself moved — 28.2
shots per match vs ~25 in every season before, in all five leagues; yellows
3.60 vs 3.89. The deployed recipe has no league term at all: team rollings
carry the level only partially (Poisson coefficients on noisy rollings sum to
< 1, so a league-wide shift passes through damped). On 2014+ history the first
70 matches of a season predict the rest better when blended 50/50 with the
previous season — for events, not for goals.

Variants, each on the DEPLOYED per-market config (features, EWM halflife,
stakes, ADM blend — read off the module so nothing drifts), no Platt on any
arm (Platt is fit on past seasons and cannot move with the current one):
  DEP   deployed recipe
  LVL   + lg_lvl: league-season side mean before this date, shrunk to the
          previous season with K matches  (n*cur + K*prev)/(n + K)
  DRIFT + lg_prev and lg_drift = lg_lvl - lg_prev (identity and drift apart)
  RES   DEP x r, r = running actual/predicted ratio of this league-season
          before this date, shrunk to 1 with K matches (a bias correction)

Walk-forward by season (fit at season start, features updated within it);
test folds 2021/22-2025/26 plus the 2026/27 matches played so far.
Scored on totals AND team-side lines (the shots-ADM lesson: a totals gain can
hide a sides loss).
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
from mundialytics.statistical_core.attack_defense_model import AttackDefenseModel  # noqa: E402

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
PREDS = [ROOT / "data/processed/enriched/understat_xg/walkforward_preds.csv",
         ROOT / "data/processed/enriched/understat_xg/walkforward_preds_hist.csv"]
TEST = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026", "2026-2027"]
K_GRID = [int(k) for k in (sys.argv[sys.argv.index("--k") + 1].split(",")
                           if "--k" in sys.argv else ["70"])]
GATES = [0.04, 0.06, 0.08]
ONLY = sys.argv[sys.argv.index("--market") + 1].split(",") if "--market" in sys.argv else list(tp.MARKETS)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def league_levels(m: pd.DataFrame, hc: str, ac: str, k: int) -> pd.DataFrame:
    """Per match: league-season side level from matches on EARLIER dates only,
    shrunk to the previous season's mean (same league)."""
    m = m.sort_values("date").copy()
    m["tot"] = (m[hc] + m[ac]).astype(float)
    season_mean = m.groupby(["competition", "season"])["tot"].mean()
    out = []
    for (comp, season), g in m.groupby(["competition", "season"], sort=False):
        prevs = [s for (c, s) in season_mean.index if c == comp and s < season]
        prev = (season_mean[(comp, max(prevs))] if prevs
                else float(m.loc[m.date < g.date.min(), "tot"].mean()) if (m.date < g.date.min()).any()
                else float(g["tot"].mean()))
        by_day = g.groupby("date")["tot"].agg(["sum", "size"])
        cum_s = by_day["sum"].cumsum().shift(1, fill_value=0.0)
        cum_n = by_day["size"].cumsum().shift(1, fill_value=0)
        lvl_day = (cum_s + k * prev) / (cum_n + k)
        gg = g[["match_id", "date"]].copy()
        gg["lg_lvl"] = gg["date"].map(lvl_day) / 2.0
        gg["lg_prev"] = prev / 2.0
        gg["n_before"] = gg["date"].map(cum_n)
        out.append(gg)
    lv = pd.concat(out, ignore_index=True)
    lv["lg_drift"] = lv["lg_lvl"] - lv["lg_prev"]
    return lv[["match_id", "lg_lvl", "lg_prev", "lg_drift", "n_before"]]


def running_ratio(te: pd.DataFrame, k: int) -> pd.Series:
    """te: one row per match with competition, date, act_tot, pred_tot.
    r = (A + k*mu)/(P + k*mu) over EARLIER dates of the same league-season."""
    r = pd.Series(1.0, index=te.index)
    for _, g in te.groupby("competition"):
        by_day = g.groupby("date")[["act_tot", "pred_tot"]].sum()
        a = by_day["act_tot"].cumsum().shift(1, fill_value=0.0)
        p = by_day["pred_tot"].cumsum().shift(1, fill_value=0.0)
        mu = float(g["pred_tot"].mean())
        ratio = (a + k * mu) / (p + k * mu)
        r.loc[g.index] = g["date"].map(ratio).to_numpy()
    return r


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
        hl = tp.EWM_HL.get(market, 5)
        adm_w = tp.ADM_W.get(market, 0.0)
        t0 = time.time()
        m = full.copy()
        for c in (hc, ac):
            m[c] = pd.to_numeric(m[c], errors="coerce")
        m = m.dropna(subset=[hc, ac])
        lr = tp.TeamPropsModel._long_rows(m, hc, ac, hl=hl)
        lr = lr.merge(m[["match_id", "competition", "season"]], on="match_id", how="left")
        # lambdas missing for 2026/27 in the caches -> goals-delta proxy, as predict does
        lr["delta_lam"] = lr["delta_lam"].fillna(lr["delta"])
        lr["abs_delta_lam"] = lr["delta_lam"].abs()
        dep = model._feature_names(market)

        for k in K_GRID:
            lv = league_levels(m, hc, ac, k)
            L = lr.merge(lv, on="match_id", how="left")
            arms = {"DEP": dep, "LVL": dep + ["lg_lvl"], "DRIFT": dep + ["lg_prev", "lg_drift"]}
            extra = ["RES"] + [f"RESg{int(t * 100)}" for t in GATES]
            res = {a: {ln: [] for ln in lines} for a in list(arms) + extra + ["BASE"]}
            res_s = {a: {ln: [] for ln in s_lines} for a in list(arms) + extra + ["BASE"]}
            bias = {a: [] for a in list(arms) + extra}
            for s in TEST:
                te_m = m[m.season == s]
                if te_m.empty:
                    continue
                s_start = te_m.date.min()
                tr_m = m[m.date < s_start]
                tt = (tr_m[hc] + tr_m[ac]).astype(float)
                disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
                sv = L.loc[L.date < s_start, "ev_for"].dropna().astype(float)
                disp_s = float(np.clip(sv.var() / max(sv.mean(), 1e-9), 0.9, 3.0))
                lg = tr_m.assign(t=tt).groupby("competition")["t"].mean()
                if adm_w > 0:
                    adm_tr = (tr_m.rename(columns={hc: "hg2", ac: "ag2"})
                              .drop(columns=["home_goals", "away_goals"])
                              .rename(columns={"hg2": "home_goals", "ag2": "away_goals"}))
                    adm = AttackDefenseModel(dixon_coles_rho=0.0, time_decay_half_life=365.0,
                                             goal_cap=tp.ADM_CAPS.get(market, 30.0), max_goals=5).fit(adm_tr)
                preds = {}
                common = None
                for arm, feats in arms.items():
                    tr = L[L.date < s_start].dropna(subset=feats + ["ev_for"])
                    reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
                    te = L[L.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
                    te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 25)
                    pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
                    preds[arm] = pv
                    common = pv.index if common is None else common.intersection(pv.index)
                tei = te_m.set_index("match_id").loc[common]
                if adm_w > 0:
                    a_l = np.array([adm.expected_goals(r.home_team, r.away_team, 0, r.competition)[:2]
                                    for r in tei.itertuples(index=False)])
                act_h, act_a = tei[hc].to_numpy(float), tei[ac].to_numpy(float)
                act = act_h + act_a
                sides = {}
                for arm, pv in preds.items():
                    lh, la = pv.loc[common, 1].to_numpy(), pv.loc[common, 0].to_numpy()
                    if adm_w > 0:
                        lh = adm_w * a_l[:, 0] + (1 - adm_w) * lh
                        la = adm_w * a_l[:, 1] + (1 - adm_w) * la
                    sides[arm] = (lh, la)
                # RES: running residual ratio on top of DEP
                d_lh, d_la = sides["DEP"]
                tmp = pd.DataFrame({"competition": tei["competition"].to_numpy(),
                                    "date": tei["date"].to_numpy(),
                                    "act_tot": act, "pred_tot": d_lh + d_la})
                r = running_ratio(tmp, k).to_numpy()
                sides["RES"] = (d_lh * r, d_la * r)
                # gated RES: correct only when the season has really moved away
                # from the model (|r - 1| above a threshold), else leave DEP alone
                for tau in GATES:
                    rg = np.where(np.abs(r - 1.0) > tau, r, 1.0)
                    sides[f"RESg{int(tau * 100)}"] = (d_lh * rg, d_la * rg)
                base_t = tei["competition"].map(lg).fillna(float(tt.mean())).to_numpy()
                sides["BASE"] = (base_t * tr_m[hc].mean() / tt.mean(), base_t * tr_m[ac].mean() / tt.mean())
                for arm, (lh, la) in sides.items():
                    tot = lh + la
                    if arm != "BASE":
                        bias[arm].append((float((tot - act).sum()), len(act), s))
                    for ln in lines:
                        y = (act > ln).astype(float)
                        res[arm][ln].append((bll(y, tp._prob_over(tot, ln, disp)), len(y), s))
                    for ln in s_lines:
                        ys = np.concatenate([act_h, act_a]) > ln
                        ps = tp._prob_over(np.concatenate([lh, la]), ln, disp_s)
                        res_s[arm][ln].append((bll(ys.astype(float), ps), len(ys), s))

            def pool(a, seasons=None):
                a = [x for x in a if seasons is None or x[2] in seasons]
                return sum(x * n for x, n, _ in a) / max(sum(n for _, n, _ in a), 1)

            hist = TEST[:-1]
            print(f"\n===== {market.upper()}  K={k}  ({time.time() - t0:.0f}s) =====", flush=True)
            for arm in ["DEP", "LVL", "DRIFT"] + extra:
                tot_d = []
                wins = 0
                for ln in lines:
                    tot_d.append(pool(res[arm][ln], hist) - pool(res["DEP"][ln], hist))
                # folds won vs DEP, averaged over lines
                for s in hist:
                    dd = np.mean([pool(res[arm][ln], [s]) - pool(res["DEP"][ln], [s]) for ln in lines])
                    wins += dd < 0
                side_d = [pool(res_s[arm][ln], hist) - pool(res_s["DEP"][ln], hist) for ln in s_lines]
                live = np.mean([pool(res[arm][ln], ["2026-2027"]) - pool(res["DEP"][ln], ["2026-2027"])
                                for ln in lines])
                vs_base = np.mean([pool(res[arm][ln], hist) - pool(res["BASE"][ln], hist) for ln in lines])
                b = [x for x in bias[arm] if x[2] == "2026-2027"]
                live_bias = b[0][0] / b[0][1] if b else float("nan")
                print(f"  {arm:5s} tot vs DEP {np.mean(tot_d):+.5f} ({wins}/5) | sides "
                      f"{(np.mean(side_d) if side_d else float('nan')):+.5f} | vs lg-base {vs_base:+.4f}"
                      f" | 26/27 vs DEP {live:+.4f}, bias {live_bias:+.2f}/match", flush=True)


if __name__ == "__main__":
    main()
