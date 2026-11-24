from __future__ import annotations

"""Round-4 event experiments — 2026-11-24

IDEA A: Cross-event shots→SOT.
  shots↔SOT r=0.62 is the strongest within-match cross-event correlation.
  Same pattern as fouls→yellows (25/25 folds in round 3).

IDEA B: Halflife tuning for yellows, fouls, shots, SOT.
  Corners went from hl=5→12→20 with monotonic gains. Test hl=3,5,8,12 for
  each market to find the optimum (5 is current default for all four).

IDEA C: Per-league NB dispersion for shots.
  Shots global disp=1.39 (high). Fouls, yellows, corners all benefited from
  per-league disp. Shots hasn't been tested.

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


# ── IDEA A: Cross-event shots→SOT ────────────────────────────────────────

def idea_a_cross_shots_sot(df):
    """Add predicted shots lambda as extra feature to the SOT model."""
    hc_s, ac_s = "home_sot", "away_sot"
    hc_sh, ac_sh = "home_shots", "away_shots"
    lines = [6.5, 7.5, 8.5, 9.5]

    m_sh, lr_sh = build_lr(df, hc_sh, ac_sh, hl=5, with_stakes=True)
    feats_sh = get_feats(with_stakes=True)

    m_s, lr_s = build_lr(df, hc_s, ac_s, hl=5, with_stakes=True)
    feats_s = get_feats(with_stakes=True)
    feats_s_ext = feats_s + ["pred_shots"]

    res = {ln: {"base": [], "cross": []} for ln in lines}
    for s in TEST_SEASONS:
        te_m_s = m_s[m_s.season == s]
        if len(te_m_s) == 0:
            continue
        s_start = te_m_s.date.min()

        tr_sh = lr_sh[lr_sh.date < s_start].dropna(subset=feats_sh + ["ev_for"])
        tr_s = lr_s[lr_s.date < s_start].dropna(subset=feats_s + ["ev_for"])
        if len(tr_sh) < 2000 or len(tr_s) < 2000:
            continue
        reg_sh = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
            tr_sh[feats_sh], tr_sh["ev_for"].clip(lower=0))

        tr_s_copy = tr_s.copy()
        tr_s_copy["pred_shots"] = np.clip(reg_sh.predict(tr_s_copy[feats_sh]), 0.1, 25)
        tr_s_ext = tr_s_copy.dropna(subset=feats_s_ext + ["ev_for"])

        reg_s_base = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
            tr_s[feats_s], tr_s["ev_for"].clip(lower=0))
        reg_s_cross = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
            tr_s_ext[feats_s_ext], tr_s_ext["ev_for"].clip(lower=0))

        te_s = lr_s[lr_s.match_id.isin(set(te_m_s.match_id))].dropna(subset=feats_s).copy()
        if te_s.empty:
            continue

        te_s["pred_base"] = np.clip(reg_s_base.predict(te_s[feats_s]), 0.1, 25)
        te_s["pred_shots"] = np.clip(reg_sh.predict(te_s[feats_sh]), 0.1, 25)
        te_s_ext = te_s.dropna(subset=feats_s_ext)
        te_s_ext = te_s_ext.copy()
        te_s_ext["pred_cross"] = np.clip(
            reg_s_cross.predict(te_s_ext[feats_s_ext]), 0.1, 25)
        te_s = te_s.merge(te_s_ext[["match_id", "is_home", "pred_cross"]],
                          on=["match_id", "is_home"], how="left")
        te_s["pred_cross"] = te_s["pred_cross"].fillna(te_s["pred_base"])

        pv_base = te_s.pivot_table(index="match_id", columns="is_home",
                                   values="pred_base").dropna()
        pv_cross = te_s.pivot_table(index="match_id", columns="is_home",
                                    values="pred_cross").dropna()
        common = pv_base.index.intersection(pv_cross.index)
        if len(common) == 0:
            continue
        tot_base = pv_base.loc[common, 1] + pv_base.loc[common, 0]
        tot_cross = pv_cross.loc[common, 1] + pv_cross.loc[common, 0]

        tei = te_m_s.set_index("match_id").loc[common]
        act = (tei[hc_s] + tei[ac_s]).astype(float)

        tr_tot = m_s[m_s.date < s_start]
        tt = (tr_tot[hc_s] + tr_tot[ac_s]).astype(float)
        disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))

        for ln in lines:
            y = (act > ln).astype(float).to_numpy()
            p_base = prob_over(tot_base.to_numpy(), ln, disp)
            p_cross = prob_over(tot_cross.to_numpy(), ln, disp)
            res[ln]["base"].append((bll(y, p_base), len(y), s))
            res[ln]["cross"].append((bll(y, p_cross), len(y), s))

    pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
    print("\n===== IDEA A: Cross-event shots -> SOT =====")
    for ln in lines:
        if not res[ln]["base"]:
            continue
        folds = " ".join(f"{g[2][-2:]}{'+' if p[0] < g[0] else '-'}"
                         for g, p in zip(res[ln]["base"], res[ln]["cross"]))
        d = pool(res[ln]["cross"]) - pool(res[ln]["base"])
        print(f"  O/U {ln}: base {pool(res[ln]['base']):.5f} | +shots {pool(res[ln]['cross']):.5f} "
              f"delta {d:+.5f}  [{folds}]")
    gc.collect()


# ── IDEA B: Halflife tuning for yellows/fouls/shots/SOT ──────────────────

def idea_b_halflife_tuning(df):
    """Test different EWMA halflife values for each market."""
    markets = [
        ("yellows", "home_yellow_cards", "away_yellow_cards", [2.5, 3.5, 4.5, 5.5, 6.5], True),
        ("fouls", "home_fouls", "away_fouls", [19.5, 21.5, 23.5, 25.5], True),
        ("shots", "home_shots", "away_shots", [20.5, 22.5, 24.5, 26.5], True),
        ("sot", "home_sot", "away_sot", [6.5, 7.5, 8.5, 9.5], True),
    ]
    halflives = [3, 5, 8, 12]

    for name, hc, ac, lines, with_stakes in markets:
        print(f"\n  -- {name.upper()} --")
        for hl in halflives:
            m, lr = build_lr(df, hc, ac, hl=hl, with_stakes=with_stakes)
            feats = get_feats(with_stakes=with_stakes)
            use_lg_disp = name in {"fouls", "yellows", "corners"}

            avg_ll = []
            for ln in lines:
                fold_lls = []
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
                    if use_lg_disp:
                        disp_g = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
                        lg_disp = (tr_tot.assign(t=tt).groupby("competition")["t"]
                                   .apply(lambda s2: float(np.clip(s2.var() / max(s2.mean(), 1e-9), 1.02, 3.0))))
                        tei = te_m.set_index("match_id").loc[tot.index]
                        disp = tei["competition"].map(lg_disp).fillna(disp_g).to_numpy()
                    else:
                        disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
                        tei = te_m.set_index("match_id").loc[tot.index]
                    act = (tei[hc] + tei[ac]).astype(float)
                    y = (act > ln).astype(float).to_numpy()
                    p = prob_over(tot.to_numpy(), ln, disp)
                    fold_lls.append(bll(y, p))
                if fold_lls:
                    avg_ll.append(np.mean(fold_lls))

            overall = np.mean(avg_ll) if avg_ll else 0
            line_strs = " | ".join(f"{ll:.4f}" for ll in avg_ll)
            marker = " <-- current" if hl == 5 else ""
            print(f"    hl={hl:>2}: avg={overall:.5f}  [{line_strs}]{marker}")
            del lr
            gc.collect()


# ── IDEA C: Per-league NB dispersion for shots ───────────────────────────

def idea_c_shots_league_disp(df):
    """Test per-league NB dispersion for shots (currently global disp=1.39)."""
    hc, ac = "home_shots", "away_shots"
    lines = [20.5, 22.5, 24.5, 26.5]
    m, lr = build_lr(df, hc, ac, hl=5, with_stakes=True)
    feats = get_feats(with_stakes=True)

    res = {ln: {"global": [], "league": []} for ln in lines}
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
        disp_g = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
        lg_disp = (tr_tot.assign(t=tt).groupby("competition")["t"]
                   .apply(lambda s2: float(np.clip(s2.var() / max(s2.mean(), 1e-9), 1.02, 3.0))))

        tei = te_m.set_index("match_id").loc[tot.index]
        disp_l = tei["competition"].map(lg_disp).fillna(disp_g).to_numpy()
        act = (tei[hc] + tei[ac]).astype(float)

        for ln in lines:
            y = (act > ln).astype(float).to_numpy()
            p_g = prob_over(tot.to_numpy(), ln, disp_g)
            p_l = prob_over(tot.to_numpy(), ln, disp_l)
            res[ln]["global"].append((bll(y, p_g), len(y), s))
            res[ln]["league"].append((bll(y, p_l), len(y), s))

    pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
    print("\n===== IDEA C: Per-league NB dispersion for shots =====")
    # Print per-league dispersions for reference
    tt_all = (df[hc].astype(float) + df[ac].astype(float)).dropna()
    print(f"  Global disp: {tt_all.var() / max(tt_all.mean(), 1e-9):.3f}")
    for comp, g in df.groupby("competition"):
        t = (g[hc].astype(float) + g[ac].astype(float)).dropna()
        if len(t) > 100:
            print(f"  {comp}: disp={t.var()/max(t.mean(),1e-9):.3f}, mean={t.mean():.1f}")
    print()
    for ln in lines:
        if not res[ln]["global"]:
            continue
        folds = " ".join(f"{g[2][-2:]}{'+' if p[0] < g[0] else '-'}"
                         for g, p in zip(res[ln]["global"], res[ln]["league"]))
        d = pool(res[ln]["league"]) - pool(res[ln]["global"])
        print(f"  O/U {ln}: global {pool(res[ln]['global']):.5f} | per-lg {pool(res[ln]['league']):.5f} "
              f"delta {d:+.5f}  [{folds}]")
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
        print("IDEA A: Cross-event shots -> SOT", flush=True)
        print("=" * 60, flush=True)
        idea_a_cross_shots_sot(df)

    if "b" in ideas:
        print("\n" + "=" * 60, flush=True)
        print("IDEA B: Halflife tuning", flush=True)
        print("=" * 60, flush=True)
        idea_b_halflife_tuning(df)

    if "c" in ideas:
        print("\n" + "=" * 60, flush=True)
        print("IDEA C: Per-league NB dispersion for shots", flush=True)
        print("=" * 60, flush=True)
        idea_c_shots_league_disp(df)

    print(f"\n\nTotal time: {time.time()-t0:.0f}s", flush=True)
