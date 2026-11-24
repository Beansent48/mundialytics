from __future__ import annotations

"""Round-3 event experiments — 2026-11-24

IDEA A: Shots post-prediction multiplicative level.
  RES_K for shots failed 2/5 — the within-model feature added noise in normal
  seasons (shots R²(prev→curr)=0.16). Try a HIGHER shrinkage K (200, 300, 500)
  so normal seasons barely move but the z=2.74 drift in 26/27 still gets caught.

IDEA B: Cross-event fouls→yellows.
  Fouls↔yellows r=0.40 is the strongest within-match cross-event correlation.
  Add predicted fouls lambda as an extra feature for the yellows model.
  The two models share features, so the marginal value may be small.

IDEA C: Corners halflife tuning.
  Corners already use hl=12. Test hl=8, 10, 15, 20 to find the optimum.

Walk-forward 2021/22–2025/26, 5 folds, scored on O/U log-loss.
"""

import gc
import sys
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


def prob_over(lam, line, disp):
    k = int(np.floor(line))
    lam = np.clip(lam, 0.2, 40.0)
    if np.isscalar(disp):
        if disp <= 1.1:
            return 1.0 - poisson.cdf(k, lam)
        disp = np.full_like(np.asarray(lam, float), disp)
    disp = np.maximum(np.asarray(disp, float), 1.001)
    r = np.asarray(lam, float) / (disp - 1.0)
    return 1.0 - nbinom.cdf(k, r, 1.0 / disp)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def add_positions(m):
    m = m.sort_values("date").copy()
    rows = []
    for (_, _), g in m.groupby(["competition", "season"], sort=False):
        pts, played = {}, {}
        for r in g.itertuples(index=False):
            def rank(t):
                p = pts.get(t, 0)
                return 1 + sum(1 for v in pts.values() if v > p)
            rh, ra = rank(r.home_team), rank(r.away_team)
            rows.append((r.match_id, rh, ra, played.get(r.home_team, 0)))
            hgo, ago = r.home_goals, r.away_goals
            pts[r.home_team] = pts.get(r.home_team, 0) + (3 if hgo > ago else (1 if hgo == ago else 0))
            pts[r.away_team] = pts.get(r.away_team, 0) + (3 if ago > hgo else (1 if hgo == ago else 0))
            played[r.home_team] = played.get(r.home_team, 0) + 1
            played[r.away_team] = played.get(r.away_team, 0) + 1
    pos = pd.DataFrame(rows, columns=["match_id", "pos_home", "pos_away", "played_home"])
    return m.merge(pos, on="match_id", how="left")


def build_lr(df, hc, ac, hl=5, with_stakes=False):
    m = df.dropna(subset=[hc, ac, "home_goals", "away_goals", "date"]).copy()
    for c in [hc, ac, "home_goals", "away_goals"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=[hc, ac, "home_goals", "away_goals"])
    if with_stakes and "pos_home" not in m.columns:
        m = add_positions(m)

    rows = []
    for r in m.itertuples(index=False):
        base = dict(match_id=r.match_id, date=r.date, competition=r.competition,
                    team=r.home_team, opp=r.away_team, is_home=1,
                    ev_for=getattr(r, hc), ev_against=getattr(r, ac),
                    gf=r.home_goals, ga=r.away_goals)
        if with_stakes and hasattr(r, "pos_home"):
            base.update(pos_t=r.pos_home, pos_o=r.pos_away, played=r.played_home)
        rows.append(base)
        base2 = dict(match_id=r.match_id, date=r.date, competition=r.competition,
                     team=r.away_team, opp=r.home_team, is_home=0,
                     ev_for=getattr(r, ac), ev_against=getattr(r, hc),
                     gf=r.away_goals, ga=r.home_goals)
        if with_stakes and hasattr(r, "pos_away"):
            base2.update(pos_t=r.pos_away, pos_o=r.pos_home, played=r.played_home)
        rows.append(base2)
    lr = pd.DataFrame(rows).sort_values(["team", "date", "match_id"])

    for col in ["ev_for", "ev_against", "gf", "ga"]:
        shifted = lr.groupby("team", sort=False)[col].shift(1)
        g = shifted.groupby(lr["team"], sort=False)
        for w in W:
            lr[f"{col}_r{w}"] = g.rolling(w, min_periods=3).mean().reset_index(level=0, drop=True)
        use_hl = hl if col in ("ev_for", "ev_against") else 5
        lr[f"{col}_ewm"] = g.ewm(halflife=use_hl, min_periods=3).mean().reset_index(level=0, drop=True)

    opp_src = [f"ev_against_r{w}" for w in W] + ["ev_against_ewm", "gf_ewm", "ga_ewm"]
    opp = lr[["match_id", "team"] + opp_src].rename(
        columns={"team": "opp", **{c: f"opp_{c}" for c in opp_src}})
    lr = lr.merge(opp, on=["match_id", "opp"], how="left")
    lr["delta"] = (lr["gf_ewm"] + lr["opp_ga_ewm"]) / 2 - (lr["opp_gf_ewm"] + lr["ga_ewm"]) / 2
    lr["abs_delta"] = lr["delta"].abs()
    if with_stakes and "pos_t" in lr.columns:
        lr["round_frac"] = (lr["played"] / 38.0).clip(0, 1)
        lr["pos_diff_abs"] = (lr["pos_t"] - lr["pos_o"]).abs()
        lr["releg_battle"] = (((lr["pos_t"] >= 15) | (lr["pos_o"] >= 15))
                              & (lr["round_frac"] > 0.6)).astype(float)
    return m, lr


def get_feats(with_stakes=False):
    base = ([f"ev_for_r{w}" for w in W] + ["ev_for_ewm"]
            + [f"opp_ev_against_r{w}" for w in W] + ["opp_ev_against_ewm"]
            + ["is_home", "delta", "abs_delta"])
    if with_stakes:
        base += ["pos_diff_abs", "releg_battle", "round_frac"]
    return base


# ── IDEA A: Shots post-prediction multiplicative level ────────────────────

def idea_a_shots_level(df):
    """Post-prediction multiplicative correction for shots.
    For each test fold, compute ratio = actual_total / predicted_total over the
    league-season BEFORE the test fold (i.e. from the previous season's data).
    Apply as lambda_corrected = lambda * ratio, shrunk to 1 with K matches."""
    hc, ac = "home_shots", "away_shots"
    lines = [20.5, 22.5, 24.5, 26.5]
    m, lr = build_lr(df, hc, ac, hl=5, with_stakes=True)
    feats = get_feats(with_stakes=True)

    for K in [120, 200, 300, 500]:
        print(f"\n  === IDEA A: Shots post-pred level, K={K} ===")
        res = {ln: {"base": [], "corrected": []} for ln in lines}
        for s in TEST_SEASONS:
            te_m = m[m.season == s]
            if len(te_m) == 0:
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
            tot_pred = pv[1] + pv[0]

            tei = te_m.set_index("match_id").loc[tot_pred.index]
            act_total = (tei[hc] + tei[ac]).astype(float)

            # Per-league dispersion (shots use global disp since not in LEAGUE_DISP)
            tr_tot = m[m.date < s_start]
            tt = (tr_tot[hc] + tr_tot[ac]).astype(float)
            disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))

            # Compute per-league residual ratio from the CURRENT season so far
            # For each test match, use only earlier matches in the same season+league
            # to compute the ratio (walk-forward within the test season too)
            cur = m[(m.season == s)]
            cur_lr = lr[lr.match_id.isin(set(cur.match_id))].dropna(subset=feats).copy()
            if cur_lr.empty:
                for ln in lines:
                    y = (act_total > ln).astype(float).to_numpy()
                    p = prob_over(tot_pred.to_numpy(), ln, disp)
                    res[ln]["base"].append((bll(y, p), len(y), s))
                    res[ln]["corrected"].append((bll(y, p), len(y), s))
                continue
            cur_lr["pred"] = np.clip(reg.predict(cur_lr[feats]), 0.1, 25)
            cur_pv = cur_lr.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
            if cur_pv.empty or 0 not in cur_pv.columns or 1 not in cur_pv.columns:
                for ln in lines:
                    y = (act_total > ln).astype(float).to_numpy()
                    p = prob_over(tot_pred.to_numpy(), ln, disp)
                    res[ln]["base"].append((bll(y, p), len(y), s))
                    res[ln]["corrected"].append((bll(y, p), len(y), s))
                continue
            cur_pred = cur_pv[1] + cur_pv[0]
            cur_info = cur.set_index("match_id").loc[cur_pred.index]
            cur_act = (cur_info[hc] + cur_info[ac]).astype(float)

            # Build walk-forward ratio per league: for each test match, use only
            # matches in the same league+season with date STRICTLY BEFORE this match
            ratio_per_match = {}
            for comp, cg in cur_info.groupby("competition"):
                cg_sorted = cg.sort_values("date")
                c_pred = cur_pred.loc[cg_sorted.index]
                c_act = cur_act.loc[cg_sorted.index]
                cs_act = c_act.cumsum()
                cs_pred = c_pred.cumsum()
                cn = pd.Series(range(1, len(cg_sorted) + 1), index=cg_sorted.index, dtype=float)
                # Shift: use only BEFORE this match
                prev_act = cs_act.shift(1, fill_value=0.0)
                prev_pred = cs_pred.shift(1, fill_value=0.0)
                prev_n = cn.shift(1, fill_value=0.0)
                mu = float(c_pred.mean()) if len(c_pred) > 0 else float(tt.mean())
                ratio = (prev_act + K * mu) / (prev_pred + K * mu)
                for mid in cg_sorted.index:
                    ratio_per_match[mid] = float(ratio.loc[mid])

            corr = np.array([ratio_per_match.get(mid, 1.0) for mid in tot_pred.index])
            tot_corrected = tot_pred.to_numpy() * corr

            for ln in lines:
                y = (act_total > ln).astype(float).to_numpy()
                p_base = prob_over(tot_pred.to_numpy(), ln, disp)
                p_corr = prob_over(tot_corrected, ln, disp)
                res[ln]["base"].append((bll(y, p_base), len(y), s))
                res[ln]["corrected"].append((bll(y, p_corr), len(y), s))

        pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
        for ln in lines:
            if not res[ln]["base"]:
                continue
            folds = " ".join(f"{g[2][-2:]}{'+' if p[0] < g[0] else '-'}"
                             for g, p in zip(res[ln]["base"], res[ln]["corrected"]))
            d = pool(res[ln]["corrected"]) - pool(res[ln]["base"])
            print(f"  O/U {ln}: base {pool(res[ln]['base']):.5f} | +level {pool(res[ln]['corrected']):.5f} "
                  f"delta {d:+.5f}  [{folds}]")
    gc.collect()


# ── IDEA B: Cross-event fouls→yellows ─────────────────────────────────────

def idea_b_cross_fouls_yellows(df):
    """Add predicted fouls lambda as extra feature to the yellows model."""
    hc_y, ac_y = "home_yellow_cards", "away_yellow_cards"
    hc_f, ac_f = "home_fouls", "away_fouls"
    lines = [2.5, 3.5, 4.5, 5.5, 6.5]

    # Build fouls model infrastructure
    m_f, lr_f = build_lr(df, hc_f, ac_f, hl=5, with_stakes=True)
    feats_f = get_feats(with_stakes=True)

    # Build yellows model
    m_y, lr_y = build_lr(df, hc_y, ac_y, hl=5, with_stakes=True)
    feats_y = get_feats(with_stakes=True)
    feats_y_ext = feats_y + ["pred_fouls"]

    res = {ln: {"base": [], "cross": []} for ln in lines}
    for s in TEST_SEASONS:
        # Train fouls model
        te_m_f = m_f[m_f.season == s]
        te_m_y = m_y[m_y.season == s]
        if len(te_m_y) == 0:
            continue
        s_start = te_m_y.date.min()
        tr_f = lr_f[lr_f.date < s_start].dropna(subset=feats_f + ["ev_for"])
        tr_y = lr_y[lr_y.date < s_start].dropna(subset=feats_y + ["ev_for"])
        if len(tr_f) < 2000 or len(tr_y) < 2000:
            continue
        reg_f = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
            tr_f[feats_f], tr_f["ev_for"].clip(lower=0))

        # Add predicted fouls to yellows training data
        tr_y_copy = tr_y.copy()
        tr_y_copy["pred_fouls"] = np.clip(reg_f.predict(tr_y_copy[feats_f]), 0.1, 25)
        tr_y_ext = tr_y_copy.dropna(subset=feats_y_ext + ["ev_for"])

        # Baseline yellows model
        reg_y_base = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
            tr_y[feats_y], tr_y["ev_for"].clip(lower=0))

        # Cross-event yellows model
        reg_y_cross = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
            tr_y_ext[feats_y_ext], tr_y_ext["ev_for"].clip(lower=0))

        te_y = lr_y[lr_y.match_id.isin(set(te_m_y.match_id))].dropna(subset=feats_y).copy()
        if te_y.empty:
            continue

        # Baseline predictions
        te_y["pred_base"] = np.clip(reg_y_base.predict(te_y[feats_y]), 0.1, 25)

        # Cross-event predictions
        te_y["pred_fouls"] = np.clip(reg_f.predict(te_y[feats_f]), 0.1, 25)
        te_y_ext_valid = te_y.dropna(subset=feats_y_ext)
        te_y_ext_valid["pred_cross"] = np.clip(
            reg_y_cross.predict(te_y_ext_valid[feats_y_ext]), 0.1, 25)
        te_y = te_y.merge(te_y_ext_valid[["match_id", "is_home", "pred_cross"]],
                          on=["match_id", "is_home"], how="left")
        te_y["pred_cross"] = te_y["pred_cross"].fillna(te_y["pred_base"])

        pv_base = te_y.pivot_table(index="match_id", columns="is_home",
                                   values="pred_base").dropna()
        pv_cross = te_y.pivot_table(index="match_id", columns="is_home",
                                    values="pred_cross").dropna()
        common = pv_base.index.intersection(pv_cross.index)
        if len(common) == 0:
            continue
        tot_base = pv_base.loc[common, 1] + pv_base.loc[common, 0]
        tot_cross = pv_cross.loc[common, 1] + pv_cross.loc[common, 0]

        tei = te_m_y.set_index("match_id").loc[common]
        act = (tei[hc_y] + tei[ac_y]).astype(float)

        tr_tot = m_y[m_y.date < s_start]
        tt = (tr_tot[hc_y] + tr_tot[ac_y]).astype(float)
        # Per-league disp (yellows is in LEAGUE_DISP)
        disp_g = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
        lg_disp = (tr_tot.assign(t=tt).groupby("competition")["t"]
                   .apply(lambda s2: float(np.clip(s2.var() / max(s2.mean(), 1e-9), 1.02, 3.0))))
        disp_l = tei["competition"].map(lg_disp).fillna(disp_g).to_numpy()

        for ln in lines:
            y = (act > ln).astype(float).to_numpy()
            p_base = prob_over(tot_base.to_numpy(), ln, disp_l)
            p_cross = prob_over(tot_cross.to_numpy(), ln, disp_l)
            res[ln]["base"].append((bll(y, p_base), len(y), s))
            res[ln]["cross"].append((bll(y, p_cross), len(y), s))

    pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
    print("\n===== IDEA B: Cross-event fouls -> yellows =====")
    for ln in lines:
        if not res[ln]["base"]:
            continue
        folds = " ".join(f"{g[2][-2:]}{'+' if p[0] < g[0] else '-'}"
                         for g, p in zip(res[ln]["base"], res[ln]["cross"]))
        d = pool(res[ln]["cross"]) - pool(res[ln]["base"])
        print(f"  O/U {ln}: base {pool(res[ln]['base']):.5f} | +fouls {pool(res[ln]['cross']):.5f} "
              f"delta {d:+.5f}  [{folds}]")
    gc.collect()


# ── IDEA C: Corners halflife tuning ───────────────────────────────────────

def idea_c_corners_halflife(df):
    """Test corners with different EWMA halflife values."""
    hc, ac = "home_corners", "away_corners"
    lines = [7.5, 8.5, 9.5, 10.5, 11.5]

    for hl in [8, 10, 12, 15, 20]:
        print(f"\n  === IDEA C: Corners hl={hl} ===")
        m, lr = build_lr(df, hc, ac, hl=hl, with_stakes=False)
        feats = get_feats(with_stakes=False)

        res = {ln: [] for ln in lines}
        for s in TEST_SEASONS:
            te_m = m[m.season == s]
            if len(te_m) == 0:
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
            tot = pv[1] + pv[0]

            tr_tot = m[m.date < s_start]
            tt = (tr_tot[hc] + tr_tot[ac]).astype(float)
            # Per-league disp (corners is in LEAGUE_DISP)
            disp_g = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
            lg_disp = (tr_tot.assign(t=tt).groupby("competition")["t"]
                       .apply(lambda s2: float(np.clip(s2.var() / max(s2.mean(), 1e-9), 1.02, 3.0))))
            tei = te_m.set_index("match_id").loc[tot.index]
            disp_l = tei["competition"].map(lg_disp).fillna(disp_g).to_numpy()
            act = (tei[hc] + tei[ac]).astype(float)

            for ln in lines:
                y = (act > ln).astype(float).to_numpy()
                p = prob_over(tot.to_numpy(), ln, disp_l)
                res[ln].append((bll(y, p), len(y), s))

        pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
        for ln in lines:
            if not res[ln]:
                continue
            print(f"  O/U {ln}: LL {pool(res[ln]):.5f}  [{', '.join(f'{r[2][-2:]}={r[0]:.4f}' for r in res[ln])}]")
        del lr
        gc.collect()


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print("Loading foundation...", flush=True)
    df = pd.read_csv(FOUND, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["season"] >= "2014-2015"]
    print(f"  {len(df)} rows loaded in {time.time()-t0:.1f}s\n", flush=True)

    ideas = sys.argv[1:] if len(sys.argv) > 1 else ["a", "b", "c"]

    if "a" in ideas:
        print("=" * 60, flush=True)
        print("IDEA A: Shots post-prediction multiplicative level", flush=True)
        print("=" * 60, flush=True)
        idea_a_shots_level(df)

    if "b" in ideas:
        print("\n" + "=" * 60, flush=True)
        print("IDEA B: Cross-event fouls -> yellows", flush=True)
        print("=" * 60, flush=True)
        idea_b_cross_fouls_yellows(df)

    if "c" in ideas:
        print("\n" + "=" * 60, flush=True)
        print("IDEA C: Corners halflife tuning", flush=True)
        print("=" * 60, flush=True)
        idea_c_corners_halflife(df)

    print(f"\n\nTotal time: {time.time()-t0:.0f}s", flush=True)
