from __future__ import annotations

"""Round-2 event experiments — 2026-11-19

IDEA A: Platt recalibration for yellows WITH per-league NB dispersion.
  Yellows was in NO_PLATT (Platt hurt it with global disp). With per-league
  disp now deployed, the raw probs may have shifted enough that Platt helps.

IDEA B: SOT from shots decomposition.
  Instead of predicting SOT independently, decompose as:
    SOT_lambda = shots_lambda × team_sot_rate
  where sot_rate = smoothed historical SOT/shots ratio per team.
  Theory: the ratio is team-skill-dependent and more stable than raw SOT.

Walk-forward 2021/22–2025/26, 5 folds, scored on O/U log-loss.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
W = (5, 10, 19)
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
STAKES_FEATURES = True


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


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build_lr(df, hc, ac, hl=5, with_stakes=False):
    m = df.dropna(subset=[hc, ac, "home_goals", "away_goals", "date"]).copy()
    for c in [hc, ac, "home_goals", "away_goals"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=[hc, ac, "home_goals", "away_goals"])

    if with_stakes and "pos_home" not in m.columns:
        m = add_positions(m)

    rows = []
    for r in m.itertuples(index=False):
        base = dict(match_id=r.match_id, date=r.date, team=r.home_team,
                    opp=r.away_team, is_home=1,
                    ev_for=getattr(r, hc), ev_against=getattr(r, ac),
                    gf=r.home_goals, ga=r.away_goals)
        if with_stakes and hasattr(r, "pos_home"):
            base.update(pos_t=r.pos_home, pos_o=r.pos_away, played=r.played_home)
        rows.append(base)
        base2 = dict(match_id=r.match_id, date=r.date, team=r.away_team,
                     opp=r.home_team, is_home=0,
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


def get_feats(with_stakes=False):
    base = ([f"ev_for_r{w}" for w in W] + ["ev_for_ewm"]
            + [f"opp_ev_against_r{w}" for w in W] + ["opp_ev_against_ewm"]
            + ["is_home", "delta", "abs_delta"])
    if with_stakes:
        base += ["pos_diff_abs", "releg_battle", "round_frac"]
    return base


# ── IDEA A: Platt for yellows with per-league NB disp ──────────────────────

def idea_a_platt_yellows(df):
    """Compare yellows O/U: (1) per-league disp NO Platt vs (2) per-league disp + Platt."""
    hc, ac = "home_yellow_cards", "away_yellow_cards"
    lines = [2.5, 3.5, 4.5, 5.5, 6.5]
    m, lr = build_lr(df, hc, ac, hl=5, with_stakes=True)
    feats = get_feats(with_stakes=True)

    res = {ln: {"base": [], "platt": []} for ln in lines}
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
        if te.empty:
            continue
        te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 25)
        pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
        if pv.empty:
            continue
        tot = pv[1] + pv[0]

        tr_tot = m[m.date < s_start]
        tt = (tr_tot[hc] + tr_tot[ac]).astype(float)
        disp_g = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
        lg_disp = (tr_tot.assign(t=tt).groupby("competition")["t"]
                   .apply(lambda s2: float(np.clip(s2.var() / max(s2.mean(), 1e-9), 1.02, 3.0))))

        tei = te_m.set_index("match_id").loc[tot.index]
        disp_l = tei["competition"].map(lg_disp).fillna(disp_g).to_numpy()
        act = tei[[hc, ac]].sum(axis=1).astype(float)

        # Platt: fit on EARLIER seasons' OOF preds with per-league disp
        platt_coefs = {}
        platt_seasons = sorted(s2 for s2 in m["season"].unique()
                               if s2 >= "2016-2017" and s2 < s)
        platt_preds = {ln: [] for ln in lines}
        for ps in platt_seasons:
            ps_m = m[m.season == ps]
            ps_start = ps_m.date.min()
            ps_tr = lr[lr.date < ps_start].dropna(subset=feats + ["ev_for"])
            if len(ps_tr) < 2000:
                continue
            ps_reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
                ps_tr[feats], ps_tr["ev_for"].clip(lower=0))
            ps_te = lr[lr.match_id.isin(set(ps_m.match_id))].dropna(subset=feats).copy()
            if ps_te.empty:
                continue
            ps_te["pred"] = np.clip(ps_reg.predict(ps_te[feats]), 0.1, 25)
            ps_pv = ps_te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
            if ps_pv.empty:
                continue
            ps_tot = ps_pv[1] + ps_pv[0]
            ps_tr2 = m[m.date < ps_start]
            ps_tt = (ps_tr2[hc] + ps_tr2[ac]).astype(float)
            ps_dg = float(np.clip(ps_tt.var() / max(ps_tt.mean(), 1e-9), 0.8, 3.0))
            ps_ldp = (ps_tr2.assign(t=ps_tt).groupby("competition")["t"]
                      .apply(lambda s3: float(np.clip(s3.var() / max(s3.mean(), 1e-9), 1.02, 3.0))))
            ps_tei = ps_m.set_index("match_id").loc[ps_tot.index]
            ps_dl = ps_tei["competition"].map(ps_ldp).fillna(ps_dg).to_numpy()
            ps_act = ps_tei[[hc, ac]].sum(axis=1).astype(float)
            for ln in lines:
                p = prob_over(ps_tot.to_numpy(), ln, ps_dl)
                y = (ps_act > ln).astype(int).to_numpy()
                platt_preds[ln].append((p, y))

        for ln in lines:
            if platt_preds[ln]:
                x_all = logit(np.concatenate([p for p, _ in platt_preds[ln]]))
                y_all = np.concatenate([y for _, y in platt_preds[ln]])
                if y_all.min() < y_all.max():
                    pl = LogisticRegression(C=1e6, max_iter=1000).fit(
                        x_all.reshape(-1, 1), y_all)
                    platt_coefs[ln] = (float(pl.coef_[0][0]), float(pl.intercept_[0]))

        for ln in lines:
            y = (act > ln).astype(float).to_numpy()
            p_base = prob_over(tot.to_numpy(), ln, disp_l)
            res[ln]["base"].append((bll(y, p_base), len(y), s))
            if ln in platt_coefs:
                a, b = platt_coefs[ln]
                z = a * logit(p_base) + b
                p_platt = 1.0 / (1.0 + np.exp(-z))
            else:
                p_platt = p_base
            res[ln]["platt"].append((bll(y, p_platt), len(y), s))

    pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
    print("\n===== IDEA A: Platt for yellows with per-league NB disp =====")
    for ln in lines:
        if not res[ln]["base"]:
            continue
        folds = " ".join(f"{g[2][-2:]}{'+' if p[0] < g[0] else '-'}"
                         for g, p in zip(res[ln]["base"], res[ln]["platt"]))
        d = pool(res[ln]["platt"]) - pool(res[ln]["base"])
        print(f"  O/U {ln}: base LL {pool(res[ln]['base']):.5f} | +Platt {pool(res[ln]['platt']):.5f} "
              f"delta {d:+.5f}  [{folds}]")


# ── IDEA B: SOT from shots decomposition ───────────────────────────────────

def idea_b_sot_decomposition(df):
    """Compare SOT: (1) baseline independent prediction vs (2) SOT = shots_lambda * sot_rate.
    sot_rate is a team-level smoothed SOT/shots ratio (EWMA)."""
    hc_sot, ac_sot = "home_sot", "away_sot"
    hc_sh, ac_sh = "home_shots", "away_shots"
    lines = [6.5, 7.5, 8.5, 9.5]

    m = df.dropna(subset=[hc_sot, ac_sot, hc_sh, ac_sh, "home_goals", "away_goals", "date"]).copy()
    for c in [hc_sot, ac_sot, hc_sh, ac_sh, "home_goals", "away_goals"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=[hc_sot, ac_sot, hc_sh, ac_sh, "home_goals", "away_goals"])
    m = add_positions(m)

    # Build long rows for shots and SOT independently
    _, lr_sot = build_lr(df, hc_sot, ac_sot, hl=5, with_stakes=True)
    _, lr_sh = build_lr(df, hc_sh, ac_sh, hl=5, with_stakes=True)

    feats_sot = get_feats(with_stakes=True)
    feats_sh = get_feats(with_stakes=True)

    # Build SOT/shots ratio rolling per team
    rows = []
    for r in m.itertuples(index=False):
        rows.append(dict(match_id=r.match_id, date=r.date, team=r.home_team,
                         sot=getattr(r, hc_sot), shots=getattr(r, hc_sh)))
        rows.append(dict(match_id=r.match_id, date=r.date, team=r.away_team,
                         sot=getattr(r, ac_sot), shots=getattr(r, ac_sh)))
    ratio_df = pd.DataFrame(rows).sort_values(["team", "date", "match_id"])
    ratio_df["sot_rate"] = (ratio_df["sot"] / ratio_df["shots"].clip(lower=1))
    shifted = ratio_df.groupby("team", sort=False)["sot_rate"].shift(1)
    ratio_df["sot_rate_ewm"] = (shifted.groupby(ratio_df["team"], sort=False)
                                 .ewm(halflife=10, min_periods=5).mean()
                                 .reset_index(level=0, drop=True))
    ratio_map = ratio_df.set_index(["match_id", "team"])["sot_rate_ewm"]

    res = {ln: {"base": [], "decomp": []} for ln in lines}
    for s in TEST_SEASONS:
        te_m = m[m.season == s]
        if len(te_m) == 0:
            continue
        s_start = te_m.date.min()

        # Baseline SOT
        tr_sot = lr_sot[lr_sot.date < s_start].dropna(subset=feats_sot + ["ev_for"])
        if len(tr_sot) < 2000:
            continue
        reg_sot = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
            tr_sot[feats_sot], tr_sot["ev_for"].clip(lower=0))
        te_sot = lr_sot[lr_sot.match_id.isin(set(te_m.match_id))].dropna(subset=feats_sot).copy()
        if te_sot.empty:
            continue
        te_sot["pred"] = np.clip(reg_sot.predict(te_sot[feats_sot]), 0.1, 25)
        pv_sot = te_sot.pivot_table(index="match_id", columns="is_home", values="pred").dropna()

        # Shots model
        tr_sh = lr_sh[lr_sh.date < s_start].dropna(subset=feats_sh + ["ev_for"])
        if len(tr_sh) < 2000:
            continue
        reg_sh = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
            tr_sh[feats_sh], tr_sh["ev_for"].clip(lower=0))
        te_sh = lr_sh[lr_sh.match_id.isin(set(te_m.match_id))].dropna(subset=feats_sh).copy()
        if te_sh.empty:
            continue
        te_sh["pred"] = np.clip(reg_sh.predict(te_sh[feats_sh]), 0.1, 25)
        pv_sh = te_sh.pivot_table(index="match_id", columns="is_home", values="pred").dropna()

        common = pv_sot.index.intersection(pv_sh.index)
        if len(common) == 0:
            continue

        tot_base = pv_sot.loc[common, 1] + pv_sot.loc[common, 0]

        # Decomposition: SOT = shots_lambda * team_sot_rate
        decomp_h, decomp_a = [], []
        for mid in common:
            te_m_row = te_m[te_m.match_id == mid]
            if te_m_row.empty:
                continue
            ht = te_m_row.iloc[0].home_team
            at = te_m_row.iloc[0].away_team
            rate_h = ratio_map.get((mid, ht), np.nan)
            rate_a = ratio_map.get((mid, at), np.nan)
            if np.isnan(rate_h) or np.isnan(rate_a):
                rate_h = 0.36 if np.isnan(rate_h) else rate_h
                rate_a = 0.36 if np.isnan(rate_a) else rate_a
            decomp_h.append(pv_sh.loc[mid, 1] * rate_h)
            decomp_a.append(pv_sh.loc[mid, 0] * rate_a)

        tot_decomp = np.array(decomp_h) + np.array(decomp_a)

        tr_tot = m[m.date < s_start]
        tt = (tr_tot[hc_sot] + tr_tot[ac_sot]).astype(float)
        disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))

        tei = te_m.set_index("match_id").loc[common]
        act = tei[[hc_sot, ac_sot]].sum(axis=1).astype(float)

        for ln in lines:
            y = (act > ln).astype(float).to_numpy()
            p_base = prob_over(tot_base.to_numpy(), ln, disp)
            p_decomp = prob_over(tot_decomp, ln, disp)
            res[ln]["base"].append((bll(y, p_base), len(y), s))
            res[ln]["decomp"].append((bll(y, p_decomp), len(y), s))

    pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
    print("\n===== IDEA B: SOT from shots decomposition =====")
    for ln in lines:
        if not res[ln]["base"]:
            continue
        folds = " ".join(f"{g[2][-2:]}{'+' if d[0] < g[0] else '-'}"
                         for g, d in zip(res[ln]["base"], res[ln]["decomp"]))
        delta = pool(res[ln]["decomp"]) - pool(res[ln]["base"])
        print(f"  O/U {ln}: base LL {pool(res[ln]['base']):.5f} | decomp {pool(res[ln]['decomp']):.5f} "
              f"delta {delta:+.5f}  [{folds}]")


# ── IDEA C: Per-league NB dispersion for SOT ──────────────────────────────

def idea_c_sot_league_disp(df):
    """Per-league NB disp for SOT — round 1 showed 6.5 at 4/5, re-confirm."""
    hc, ac = "home_sot", "away_sot"
    lines = [6.5, 7.5, 8.5, 9.5]
    m, lr = build_lr(df, hc, ac, hl=5, with_stakes=True)
    feats = get_feats(with_stakes=True)

    res = {ln: {"g": [], "l": []} for ln in lines}
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
        if te.empty:
            continue
        te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 25)
        pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
        if pv.empty:
            continue
        tot = pv[1] + pv[0]
        tr_tot = m[m.date < s_start]
        tt = (tr_tot[hc] + tr_tot[ac]).astype(float)
        disp_g = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
        lg_disp = (tr_tot.assign(t=tt).groupby("competition")["t"]
                   .apply(lambda s2: float(np.clip(s2.var() / max(s2.mean(), 1e-9), 1.02, 3.0))))
        tei = te_m.set_index("match_id").loc[tot.index]
        disp_l = tei["competition"].map(lg_disp).fillna(disp_g).to_numpy()
        act = tei[[hc, ac]].sum(axis=1).astype(float)
        for ln in lines:
            y = (act > ln).astype(float).to_numpy()
            res[ln]["g"].append((bll(y, prob_over(tot.to_numpy(), ln, disp_g)), len(y), s))
            res[ln]["l"].append((bll(y, prob_over(tot.to_numpy(), ln, disp_l)), len(y), s))

    pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
    print("\n===== IDEA C: Per-league NB disp for SOT =====")
    for ln in lines:
        if not res[ln]["g"]:
            continue
        folds = " ".join(f"{g[2][-2:]}{'+' if l[0] < g[0] else '-'}"
                         for g, l in zip(res[ln]["g"], res[ln]["l"]))
        d = pool(res[ln]["l"]) - pool(res[ln]["g"])
        print(f"  O/U {ln}: global-disp LL {pool(res[ln]['g']):.5f} | per-league {pool(res[ln]['l']):.5f} "
              f"delta {d:+.5f}  [{folds}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-sep idea letters: a,b,c")
    args = parser.parse_args()
    only = {x.strip().upper() for x in args.only.split(",") if x.strip()} if args.only else set()

    print("Loading foundation...")
    df = pd.read_csv(FOUND, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["season"] >= "2014-2015"]
    print(f"  {len(df)} rows, {df.season.nunique()} seasons\n")

    if not only or "A" in only:
        t0 = time.time()
        idea_a_platt_yellows(df)
        print(f"  ({time.time()-t0:.0f}s)")

    if not only or "B" in only:
        t0 = time.time()
        idea_b_sot_decomposition(df)
        print(f"  ({time.time()-t0:.0f}s)")

    if not only or "C" in only:
        t0 = time.time()
        idea_c_sot_league_disp(df)
        print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
