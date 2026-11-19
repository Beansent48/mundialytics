from __future__ import annotations

"""Four event-market improvements, walk-forward validated (2021/22-2025/26).

Ideas tested (each vs the DEPLOYED baseline, no Platt on any arm):

1. SHOTS_RES  Post-prediction multiplicative residual correction for shots
              (the market-level architecture: lambda * r, r shrunk to 1 with K).
              Different from the failed in-model LVL/RES: this applies AFTER
              the full predict pipeline (ADM blend included) instead of inside
              the PoissonRegressor features. K grid: 50,70,100,150.

2. DISP_LG    Per-league NB dispersion for corners and yellows (already
              validated and deployed for fouls in experiment_disp_per_league.py;
              never tested for corners/yellows).

3. STAKES     Walk-forward standings features (position, relegation, round_frac)
              for shots and SOT (currently only used for yellows/fouls).

4. CROSS      Predicted fouls lambda as an extra feature for yellows
              (fouls<->yellows r=0.40 is the strongest within-match correlation).

Usage:
  python scripts/experiment_event_improvements.py
  python scripts/experiment_event_improvements.py --only 1,2   # run ideas 1 and 2
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.props import team_props as tp
from mundialytics.statistical_core.attack_defense_model import AttackDefenseModel

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
PREDS = [ROOT / "data/processed/enriched/understat_xg/walkforward_preds.csv",
         ROOT / "data/processed/enriched/understat_xg/walkforward_preds_hist.csv"]
TEST = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
HIST = TEST  # all are historical for scoring
ONLY_IDEAS = None
if "--only" in sys.argv:
    ONLY_IDEAS = set(int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(","))


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def prob_over(lam, line, disp):
    k = int(np.floor(line))
    lam = np.clip(lam, 0.2, 40.0)
    if isinstance(disp, (float, int)) and disp > 1.1:
        r = lam / (disp - 1.0)
        return 1.0 - nbinom.cdf(k, r, 1.0 / disp)
    if isinstance(disp, np.ndarray):
        out = np.zeros_like(lam, dtype=float)
        nb = disp > 1.1
        if nb.any():
            r = lam[nb] / (disp[nb] - 1.0)
            out[nb] = 1.0 - nbinom.cdf(k, r, 1.0 / disp[nb])
        pois = ~nb
        if pois.any():
            out[pois] = 1.0 - poisson.cdf(k, lam[pois])
        return out
    return 1.0 - poisson.cdf(k, lam)


def pool(entries, seasons=None):
    entries = [x for x in entries if seasons is None or x[2] in seasons]
    if not entries:
        return float("nan")
    return sum(x * n for x, n, _ in entries) / max(sum(n for _, n, _ in entries), 1)


def build_long_rows(m, hc, ac, hl=5, extra_cols=None):
    """Build long-format rows with rollings, ASYM, and optionally lambdas/positions."""
    has_lam = "lh" in m.columns
    has_pos = "pos_home" in m.columns
    base = ["match_id", "date"]
    h_map = {"home_team": "team", "away_team": "opp", hc: "ev_for", ac: "ev_against",
             "home_goals": "gf", "away_goals": "ga"}
    a_map = {"away_team": "team", "home_team": "opp", ac: "ev_for", hc: "ev_against",
             "away_goals": "gf", "home_goals": "ga"}
    if has_lam:
        h_map.update({"lh": "lam_t", "la": "lam_o"})
        a_map.update({"la": "lam_t", "lh": "lam_o"})
    if has_pos:
        h_map.update({"pos_home": "pos_t", "pos_away": "pos_o", "played_home": "played"})
        a_map.update({"pos_away": "pos_t", "pos_home": "pos_o", "played_home": "played"})
    for col in (extra_cols or []):
        h_map[col] = col
        a_map[col] = col
    h = m[base + list(h_map)].rename(columns=h_map).assign(is_home=1)
    a = m[base + list(a_map)].rename(columns=a_map).assign(is_home=0)
    lr = pd.concat([h, a], ignore_index=True).sort_values(["team", "date", "match_id"])
    tkey = lr["team"]
    for col in ["ev_for", "ev_against", "gf", "ga"]:
        shifted = lr.groupby("team", sort=False)[col].shift(1)
        g = shifted.groupby(tkey, sort=False)
        for w in tp.WINDOWS:
            lr[f"{col}_r{w}"] = g.rolling(w, min_periods=3).mean().reset_index(level=0, drop=True)
        use_hl = hl if col in ("ev_for", "ev_against") else 5
        lr[f"{col}_ewm"] = g.ewm(halflife=use_hl, min_periods=3).mean().reset_index(level=0, drop=True)
    opp_src = [f"ev_against_r{w}" for w in tp.WINDOWS] + ["ev_against_ewm", "gf_ewm", "ga_ewm"]
    opp = lr[["match_id", "team"] + opp_src].rename(
        columns={"team": "opp", **{c: f"opp_{c}" for c in opp_src}})
    lr = lr.merge(opp, on=["match_id", "opp"], how="left")
    lr["delta"] = (lr["gf_ewm"] + lr["opp_ga_ewm"]) / 2 - (lr["opp_gf_ewm"] + lr["ga_ewm"]) / 2
    lr["abs_delta"] = lr["delta"].abs()
    if has_lam:
        lr["delta_lam"] = lr["lam_t"] - lr["lam_o"]
        lr["abs_delta_lam"] = lr["delta_lam"].abs()
    if has_pos:
        lr["round_frac"] = (lr["played"] / 38.0).clip(0, 1)
        lr["pos_diff_abs"] = (lr["pos_t"] - lr["pos_o"]).abs()
        lr["releg_battle"] = (((lr["pos_t"] >= 15) | (lr["pos_o"] >= 15))
                              & (lr["round_frac"] > 0.6)).astype(float)
    return lr


def get_features(market, use_lam=True, add_stakes=False, add_cross=False,
                 add_lg_lvl=False, add_ref_dev=False):
    base = ([f"ev_for_r{w}" for w in tp.WINDOWS] + ["ev_for_ewm"]
            + [f"opp_ev_against_r{w}" for w in tp.WINDOWS] + ["opp_ev_against_ewm"]
            + ["is_home", "delta", "abs_delta"])
    if use_lam:
        base += ["delta_lam", "abs_delta_lam"]
    if market in tp.STAKES_MARKETS or add_stakes:
        base += ["pos_diff_abs", "releg_battle", "round_frac"]
    if add_lg_lvl:
        base += ["lg_lvl"]
    if add_ref_dev:
        base += ["ref_dev"]
    if add_cross:
        base += ["pred_fouls_total"]
    return base


def running_ratio_post(act, pred, comp, date, k):
    """Post-prediction residual ratio, per competition, from EARLIER dates."""
    df = pd.DataFrame({"act": act, "pred": pred, "comp": comp, "date": date})
    r = np.ones(len(df))
    for c, g in df.groupby("comp"):
        by_day = g.groupby("date")[["act", "pred"]].sum()
        cum_a = by_day["act"].cumsum().shift(1, fill_value=0.0)
        cum_p = by_day["pred"].cumsum().shift(1, fill_value=0.0)
        mu = float(g["pred"].mean())
        ratio = (cum_a + k * mu) / (cum_p + k * mu)
        idx = g.index
        r[idx] = g["date"].map(ratio).to_numpy()
    return r


def main():
    t_start = time.time()
    lam_parts = []
    for p in PREDS:
        if p.exists():
            lam_parts.append(pd.read_csv(p)[["match_id", "lh", "la"]])
    lam = pd.concat(lam_parts, ignore_index=True).drop_duplicates("match_id") if lam_parts else None

    full = pd.read_csv(FOUND, low_memory=False)
    full["date"] = pd.to_datetime(full["date"], errors="coerce")
    full = full[full["season"] >= "2014-2015"].dropna(subset=["home_goals", "away_goals", "date"])
    if lam is not None:
        full = full.merge(lam, on="match_id", how="left")
    else:
        full["lh"] = np.nan
        full["la"] = np.nan
    full, team_pos = tp.TeamPropsModel._add_positions(full)

    # ══════════════════════════════════════════════════════════════════════
    # IDEA 1: Post-prediction shots residual correction
    # ══════════════════════════════════════════════════════════════════════
    if ONLY_IDEAS is None or 1 in ONLY_IDEAS:
        print("\n" + "=" * 80)
        print("IDEA 1: SHOTS POST-PREDICTION RESIDUAL CORRECTION")
        print("=" * 80)

        K_GRID = [50, 70, 100, 150, 200]
        for market in ["shots", "sot", "corners"]:
            hc, ac, lines = tp.MARKETS[market]
            s_lines = tp.SIDE_LINES.get(market, [])
            hl = tp.EWM_HL.get(market, 5)
            adm_w = tp.ADM_W.get(market, 0.0)
            m = full.copy()
            for c in (hc, ac):
                m[c] = pd.to_numeric(m[c], errors="coerce")
            m = m.dropna(subset=[hc, ac])
            lr = build_long_rows(m, hc, ac, hl=hl)
            lr = lr.merge(m[["match_id", "competition", "season"]], on="match_id", how="left")
            lr["delta_lam"] = lr["delta_lam"].fillna(lr["delta"])
            lr["abs_delta_lam"] = lr["delta_lam"].abs()

            feats = get_features(market)

            print(f"\n--- {market.upper()} ---")
            for k in K_GRID:
                res_dep = {ln: [] for ln in lines}
                res_cor = {ln: [] for ln in lines}
                res_dep_s = {ln: [] for ln in s_lines}
                res_cor_s = {ln: [] for ln in s_lines}
                for s in TEST:
                    te_m = m[m.season == s]
                    if te_m.empty:
                        continue
                    s_start = te_m.date.min()
                    tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
                    if len(tr) < 2000:
                        continue
                    reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
                        tr[feats], tr["ev_for"].clip(lower=0))
                    te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
                    if te.empty:
                        continue
                    te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 25)
                    pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
                    if pv.empty:
                        continue
                    tei = te_m.set_index("match_id").loc[pv.index]
                    lh, la = pv[1].to_numpy(), pv[0].to_numpy()
                    if adm_w > 0:
                        adm_tr = (m[m.date < s_start].rename(columns={hc: "hg2", ac: "ag2"})
                                  .drop(columns=["home_goals", "away_goals"])
                                  .rename(columns={"hg2": "home_goals", "ag2": "away_goals"}))
                        adm = AttackDefenseModel(dixon_coles_rho=0.0, time_decay_half_life=365.0,
                                                  goal_cap=tp.ADM_CAPS.get(market, 30.0),
                                                  max_goals=5).fit(adm_tr)
                        a_l = np.array([adm.expected_goals(h, a, 0, c)[:2]
                                        for h, a, c in zip(tei["home_team"], tei["away_team"],
                                                           tei["competition"])])
                        lh = adm_w * a_l[:, 0] + (1 - adm_w) * lh
                        la = adm_w * a_l[:, 1] + (1 - adm_w) * la

                    act_h = tei[hc].to_numpy(float)
                    act_a = tei[ac].to_numpy(float)
                    act = act_h + act_a
                    tot_dep = lh + la

                    # Post-prediction residual ratio
                    r = running_ratio_post(act, tot_dep,
                                           tei["competition"].to_numpy(),
                                           tei["date"].to_numpy(), k)
                    lh_c, la_c = lh * r, la * r
                    tot_cor = lh_c + la_c

                    tr_m = m[m.date < s_start]
                    tt = (tr_m[hc] + tr_m[ac]).astype(float)
                    disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
                    sv = lr.loc[lr.date < s_start, "ev_for"].dropna().astype(float)
                    disp_s = float(np.clip(sv.var() / max(sv.mean(), 1e-9), 0.9, 3.0))

                    for ln in lines:
                        y = (act > ln).astype(float)
                        res_dep[ln].append((bll(y, prob_over(tot_dep, ln, disp)), len(y), s))
                        res_cor[ln].append((bll(y, prob_over(tot_cor, ln, disp)), len(y), s))
                    for ln in s_lines:
                        ys = np.concatenate([act_h, act_a]) > ln
                        p_dep = prob_over(np.concatenate([lh, la]), ln, disp_s)
                        p_cor = prob_over(np.concatenate([lh_c, la_c]), ln, disp_s)
                        res_dep_s[ln].append((bll(ys.astype(float), p_dep), len(ys), s))
                        res_cor_s[ln].append((bll(ys.astype(float), p_cor), len(ys), s))

                # Score
                wins_t, wins_s = 0, 0
                deltas_t, deltas_s = [], []
                for ln in lines:
                    d = pool(res_cor[ln]) - pool(res_dep[ln])
                    deltas_t.append(d)
                for s in TEST:
                    dd = np.mean([pool(res_cor[ln], [s]) - pool(res_dep[ln], [s]) for ln in lines])
                    wins_t += dd < 0
                for ln in s_lines:
                    d = pool(res_cor_s[ln]) - pool(res_dep_s[ln])
                    deltas_s.append(d)
                for s in TEST:
                    dd = np.mean([pool(res_cor_s[ln], [s]) - pool(res_dep_s[ln], [s]) for ln in s_lines])
                    wins_s += dd < 0 if s_lines else 0

                print(f"  K={k:3d}: totals {np.mean(deltas_t):+.5f} ({wins_t}/5) | "
                      f"sides {np.mean(deltas_s):+.5f} ({wins_s}/5)" if s_lines else
                      f"  K={k:3d}: totals {np.mean(deltas_t):+.5f} ({wins_t}/5)")

    # ══════════════════════════════════════════════════════════════════════
    # IDEA 2: Per-league NB dispersion for corners and yellows
    # ══════════════════════════════════════════════════════════════════════
    if ONLY_IDEAS is None or 2 in ONLY_IDEAS:
        print("\n" + "=" * 80)
        print("IDEA 2: PER-LEAGUE NB DISPERSION FOR CORNERS AND YELLOWS")
        print("=" * 80)

        for market in ["corners", "yellows", "sot"]:
            hc, ac, lines = tp.MARKETS[market]
            hl = tp.EWM_HL.get(market, 5)
            adm_w = tp.ADM_W.get(market, 0.0)
            m = full.copy()
            for c in (hc, ac):
                m[c] = pd.to_numeric(m[c], errors="coerce")
            m = m.dropna(subset=[hc, ac])
            lr = build_long_rows(m, hc, ac, hl=hl)
            lr = lr.merge(m[["match_id", "competition", "season"]], on="match_id", how="left")
            lr["delta_lam"] = lr["delta_lam"].fillna(lr["delta"])
            lr["abs_delta_lam"] = lr["delta_lam"].abs()
            feats = get_features(market)

            print(f"\n--- {market.upper()} ---")
            res_g = {ln: [] for ln in lines}
            res_l = {ln: [] for ln in lines}
            disp_report = None
            for s in TEST:
                te_m = m[m.season == s]
                if te_m.empty:
                    continue
                s_start = te_m.date.min()
                tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
                if len(tr) < 2000:
                    continue
                reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
                    tr[feats], tr["ev_for"].clip(lower=0))
                te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
                if te.empty:
                    continue
                te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 25)
                pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
                if pv.empty:
                    continue
                tei = te_m.set_index("match_id").loc[pv.index]
                lh, la = pv[1].to_numpy(), pv[0].to_numpy()
                if adm_w > 0:
                    adm_tr = (m[m.date < s_start].rename(columns={hc: "hg2", ac: "ag2"})
                              .drop(columns=["home_goals", "away_goals"])
                              .rename(columns={"hg2": "home_goals", "ag2": "away_goals"}))
                    adm = AttackDefenseModel(dixon_coles_rho=0.0, time_decay_half_life=365.0,
                                              goal_cap=tp.ADM_CAPS.get(market, 30.0),
                                              max_goals=5).fit(adm_tr)
                    a_l = np.array([adm.expected_goals(h, a, 0, c)[:2]
                                    for h, a, c in zip(tei["home_team"], tei["away_team"],
                                                       tei["competition"])])
                    lh = adm_w * a_l[:, 0] + (1 - adm_w) * lh
                    la = adm_w * a_l[:, 1] + (1 - adm_w) * la
                tot = lh + la

                tr_m = m[m.date < s_start]
                tt = (tr_m[hc] + tr_m[ac]).astype(float)
                disp_g = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
                lg_disp = (tr_m.assign(t=tt).groupby("competition")["t"]
                           .apply(lambda s2: float(np.clip(s2.var() / max(s2.mean(), 1e-9),
                                                            1.02, 3.0))))
                disp_report = lg_disp.round(3).to_dict()
                disp_l = tei["competition"].map(lg_disp).fillna(disp_g).to_numpy()
                act = tei[[hc, ac]].sum(axis=1).astype(float).to_numpy()

                for ln in lines:
                    y = (act > ln).astype(float)
                    res_g[ln].append((bll(y, prob_over(tot, ln, disp_g)), len(y), s))
                    res_l[ln].append((bll(y, prob_over(tot, ln, disp_l)), len(y), s))

            print(f"  Per-league dispersions (last fold): {disp_report}")
            for ln in lines:
                wins = sum(1 for g, l in zip(res_g[ln], res_l[ln]) if l[0] < g[0])
                d = pool(res_l[ln]) - pool(res_g[ln])
                folds = " ".join(f"{g[2][-2:]}{'+' if l[0] < g[0] else '-'}"
                                 for g, l in zip(res_g[ln], res_l[ln]))
                print(f"  O/U {ln}: global {pool(res_g[ln]):.5f} | per-lg {pool(res_l[ln]):.5f} "
                      f"delta {d:+.5f} ({wins}/5) [{folds}]")

    # ══════════════════════════════════════════════════════════════════════
    # IDEA 3: Stakes features for shots and SOT
    # ══════════════════════════════════════════════════════════════════════
    if ONLY_IDEAS is None or 3 in ONLY_IDEAS:
        print("\n" + "=" * 80)
        print("IDEA 3: STAKES FEATURES (position, relegation, round) FOR SHOTS AND SOT")
        print("=" * 80)

        for market in ["shots", "sot"]:
            hc, ac, lines = tp.MARKETS[market]
            s_lines = tp.SIDE_LINES.get(market, [])
            hl = tp.EWM_HL.get(market, 5)
            adm_w = tp.ADM_W.get(market, 0.0)
            m = full.copy()
            for c in (hc, ac):
                m[c] = pd.to_numeric(m[c], errors="coerce")
            m = m.dropna(subset=[hc, ac])
            lr = build_long_rows(m, hc, ac, hl=hl)
            lr = lr.merge(m[["match_id", "competition", "season"]], on="match_id", how="left")
            lr["delta_lam"] = lr["delta_lam"].fillna(lr["delta"])
            lr["abs_delta_lam"] = lr["delta_lam"].abs()

            feats_no = get_features(market, add_stakes=False)
            feats_yes = get_features(market, add_stakes=True)

            print(f"\n--- {market.upper()} ---")
            res_no = {ln: [] for ln in lines + s_lines}
            res_yes = {ln: [] for ln in lines + s_lines}
            for s in TEST:
                te_m = m[m.season == s]
                if te_m.empty:
                    continue
                s_start = te_m.date.min()
                for feats, res_dict in [(feats_no, res_no), (feats_yes, res_yes)]:
                    tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
                    if len(tr) < 2000:
                        continue
                    reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
                        tr[feats], tr["ev_for"].clip(lower=0))
                    te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
                    if te.empty:
                        continue
                    te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 25)
                    pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
                    if pv.empty:
                        continue
                    tei = te_m.set_index("match_id").loc[pv.index]
                    lh, la = pv[1].to_numpy(), pv[0].to_numpy()
                    if adm_w > 0:
                        adm_tr = (m[m.date < s_start].rename(columns={hc: "hg2", ac: "ag2"})
                                  .drop(columns=["home_goals", "away_goals"])
                                  .rename(columns={"hg2": "home_goals", "ag2": "away_goals"}))
                        adm = AttackDefenseModel(dixon_coles_rho=0.0, time_decay_half_life=365.0,
                                                  goal_cap=tp.ADM_CAPS.get(market, 30.0),
                                                  max_goals=5).fit(adm_tr)
                        a_l = np.array([adm.expected_goals(h, a, 0, c)[:2]
                                        for h, a, c in zip(tei["home_team"], tei["away_team"],
                                                           tei["competition"])])
                        lh = adm_w * a_l[:, 0] + (1 - adm_w) * lh
                        la = adm_w * a_l[:, 1] + (1 - adm_w) * la

                    act_h = tei[hc].to_numpy(float)
                    act_a = tei[ac].to_numpy(float)
                    act = act_h + act_a
                    tot = lh + la

                    tr_m = m[m.date < s_start]
                    tt = (tr_m[hc] + tr_m[ac]).astype(float)
                    disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
                    sv = lr.loc[lr.date < s_start, "ev_for"].dropna().astype(float)
                    disp_s = float(np.clip(sv.var() / max(sv.mean(), 1e-9), 0.9, 3.0))

                    for ln in lines:
                        y = (act > ln).astype(float)
                        res_dict[ln].append((bll(y, prob_over(tot, ln, disp)), len(y), s))
                    for ln in s_lines:
                        ys = np.concatenate([act_h, act_a]) > ln
                        ps = prob_over(np.concatenate([lh, la]), ln, disp_s)
                        res_dict[ln].append((bll(ys.astype(float), ps), len(ys), s))

            for ln in lines:
                wins = sum(1 for g, l in zip(res_no[ln], res_yes[ln]) if l[0] < g[0])
                d = pool(res_yes[ln]) - pool(res_no[ln])
                print(f"  O/U {ln}: no-stakes {pool(res_no[ln]):.5f} | +stakes {pool(res_yes[ln]):.5f} "
                      f"delta {d:+.5f} ({wins}/5)")
            for ln in s_lines:
                wins = sum(1 for g, l in zip(res_no[ln], res_yes[ln]) if l[0] < g[0])
                d = pool(res_yes[ln]) - pool(res_no[ln])
                print(f"  side {ln}: no-stakes {pool(res_no[ln]):.5f} | +stakes {pool(res_yes[ln]):.5f} "
                      f"delta {d:+.5f} ({wins}/5)")

    # ══════════════════════════════════════════════════════════════════════
    # IDEA 4: Cross-event: predicted fouls -> yellows
    # ══════════════════════════════════════════════════════════════════════
    if ONLY_IDEAS is None or 4 in ONLY_IDEAS:
        print("\n" + "=" * 80)
        print("IDEA 4: CROSS-EVENT — PREDICTED FOULS LAMBDA AS FEATURE FOR YELLOWS")
        print("=" * 80)

        # First, build fouls predictions per match
        hc_f, ac_f, _ = tp.MARKETS["fouls"]
        hl_f = tp.EWM_HL.get("fouls", 5)
        m_f = full.copy()
        for c in (hc_f, ac_f):
            m_f[c] = pd.to_numeric(m_f[c], errors="coerce")
        m_f = m_f.dropna(subset=[hc_f, ac_f])
        lr_f = build_long_rows(m_f, hc_f, ac_f, hl=hl_f)
        lr_f = lr_f.merge(m_f[["match_id", "competition", "season"]], on="match_id", how="left")
        lr_f["delta_lam"] = lr_f["delta_lam"].fillna(lr_f["delta"])
        lr_f["abs_delta_lam"] = lr_f["delta_lam"].abs()
        feats_f = get_features("fouls")

        # Now yellows
        market = "yellows"
        hc_y, ac_y, lines = tp.MARKETS[market]
        s_lines = tp.SIDE_LINES.get(market, [])
        hl_y = tp.EWM_HL.get(market, 5)
        adm_w = tp.ADM_W.get(market, 0.0)
        m_y = full.copy()
        for c in (hc_y, ac_y):
            m_y[c] = pd.to_numeric(m_y[c], errors="coerce")
        m_y = m_y.dropna(subset=[hc_y, ac_y])
        lr_y = build_long_rows(m_y, hc_y, ac_y, hl=hl_y)
        lr_y = lr_y.merge(m_y[["match_id", "competition", "season"]], on="match_id", how="left")
        lr_y["delta_lam"] = lr_y["delta_lam"].fillna(lr_y["delta"])
        lr_y["abs_delta_lam"] = lr_y["delta_lam"].abs()
        feats_y_base = get_features(market)
        feats_y_cross = get_features(market, add_cross=True)

        res_base = {ln: [] for ln in lines + s_lines}
        res_cross = {ln: [] for ln in lines + s_lines}

        for s in TEST:
            te_m_y = m_y[m_y.season == s]
            te_m_f = m_f[m_f.season == s]
            if te_m_y.empty:
                continue
            s_start = te_m_y.date.min()

            # Train fouls model and predict fouls for the test set
            tr_f = lr_f[lr_f.date < s_start].dropna(subset=feats_f + ["ev_for"])
            if len(tr_f) < 2000:
                continue
            reg_f = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
                tr_f[feats_f], tr_f["ev_for"].clip(lower=0))
            te_f = lr_f[lr_f.match_id.isin(set(te_m_f.match_id))].dropna(subset=feats_f).copy()
            if te_f.empty:
                continue
            te_f["pred_fouls"] = np.clip(reg_f.predict(te_f[feats_f]), 0.1, 25)
            pv_f = te_f.pivot_table(index="match_id", columns="is_home", values="pred_fouls").dropna()
            fouls_total = (pv_f[1] + pv_f[0]).rename("pred_fouls_total")

            # Also predict fouls for TRAINING yellows (walk-forward fouls in training window)
            # For the training window, we need a fouls model trained on data before that
            all_f = lr_f.dropna(subset=feats_f).copy()
            # Simple approach: predict fouls for all historical matches using their own pre-data
            # Actually, for cross-feature we need fouls predictions at TRAINING time too
            # Use the same fouls model (trained up to s_start) to predict fouls for all dates < s_start
            tr_f_all = lr_f[lr_f.date < s_start].dropna(subset=feats_f).copy()
            tr_f_all["pred_fouls"] = np.clip(reg_f.predict(tr_f_all[feats_f]), 0.1, 25)
            pv_f_tr = tr_f_all.pivot_table(index="match_id", columns="is_home",
                                            values="pred_fouls").dropna()
            fouls_total_tr = (pv_f_tr[1] + pv_f_tr[0]).rename("pred_fouls_total")

            # Merge fouls predictions into yellows long rows
            lr_y_s = lr_y.copy()
            # For each match_id, the fouls total is the same for both sides
            fouls_all = pd.concat([fouls_total_tr, fouls_total])
            fouls_all = fouls_all[~fouls_all.index.duplicated()]
            lr_y_s = lr_y_s.merge(fouls_all.reset_index().rename(columns={"match_id": "match_id"}),
                                   on="match_id", how="left")

            for feats, res_dict in [(feats_y_base, res_base), (feats_y_cross, res_cross)]:
                tr = lr_y_s[lr_y_s.date < s_start].dropna(subset=feats + ["ev_for"])
                if len(tr) < 2000:
                    continue
                reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
                    tr[feats], tr["ev_for"].clip(lower=0))
                te = lr_y_s[lr_y_s.match_id.isin(set(te_m_y.match_id))].dropna(subset=feats).copy()
                if te.empty:
                    continue
                te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 25)
                pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
                if pv.empty:
                    continue
                tei = te_m_y.set_index("match_id").loc[pv.index]
                lh, la = pv[1].to_numpy(), pv[0].to_numpy()
                if adm_w > 0:
                    adm_tr = (m_y[m_y.date < s_start].rename(columns={hc_y: "hg2", ac_y: "ag2"})
                              .drop(columns=["home_goals", "away_goals"])
                              .rename(columns={"hg2": "home_goals", "ag2": "away_goals"}))
                    adm = AttackDefenseModel(dixon_coles_rho=0.0, time_decay_half_life=365.0,
                                              goal_cap=tp.ADM_CAPS.get(market, 30.0),
                                              max_goals=5).fit(adm_tr)
                    a_l = np.array([adm.expected_goals(h, a, 0, c)[:2]
                                    for h, a, c in zip(tei["home_team"], tei["away_team"],
                                                       tei["competition"])])
                    lh = adm_w * a_l[:, 0] + (1 - adm_w) * lh
                    la = adm_w * a_l[:, 1] + (1 - adm_w) * la
                act_h = tei[hc_y].to_numpy(float)
                act_a = tei[ac_y].to_numpy(float)
                act = act_h + act_a
                tot = lh + la

                tr_m = m_y[m_y.date < s_start]
                tt = (tr_m[hc_y] + tr_m[ac_y]).astype(float)
                disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
                sv = lr_y_s.loc[lr_y_s.date < s_start, "ev_for"].dropna().astype(float)
                disp_s = float(np.clip(sv.var() / max(sv.mean(), 1e-9), 0.9, 3.0))

                for ln in lines:
                    y = (act > ln).astype(float)
                    res_dict[ln].append((bll(y, prob_over(tot, ln, disp)), len(y), s))
                for ln in s_lines:
                    ys = np.concatenate([act_h, act_a]) > ln
                    ps = prob_over(np.concatenate([lh, la]), ln, disp_s)
                    res_dict[ln].append((bll(ys.astype(float), ps), len(ys), s))

        print(f"\n--- YELLOWS with cross-event fouls feature ---")
        for ln in lines:
            wins = sum(1 for g, l in zip(res_base[ln], res_cross[ln]) if l[0] < g[0])
            d = pool(res_cross[ln]) - pool(res_base[ln])
            print(f"  O/U {ln}: base {pool(res_base[ln]):.5f} | +fouls {pool(res_cross[ln]):.5f} "
                  f"delta {d:+.5f} ({wins}/5)")
        for ln in s_lines:
            wins = sum(1 for g, l in zip(res_base[ln], res_cross[ln]) if l[0] < g[0])
            d = pool(res_cross[ln]) - pool(res_base[ln])
            print(f"  side {ln}: base {pool(res_base[ln]):.5f} | +fouls {pool(res_cross[ln]):.5f} "
                  f"delta {d:+.5f} ({wins}/5)")

    print(f"\nTotal time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
