"""Event-model round 6: literature-inspired ideas.

  A. Heteroscedastic dispersion: NB dispersion as function of |delta|
     (Yip et al. 2021: goal supremacy affects VARIANCE, not mean)
  B. XGBoost with Poisson loss vs PoissonRegressor (sanity check)
  C. Evaluate Geometric-Poisson applicability (corner clustering)
  D. Check possession data availability and correlation

Walk-forward 5-fold validation on O/U log-loss.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor

ROOT = Path(r"C:\Users\Vicente\Desktop\BetBot\mundialytics_betting_engine")
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

EVENTS = {
    "corners": ("home_corners", "away_corners", [7.5, 8.5, 9.5, 10.5, 11.5]),
    "yellows": ("home_yellow_cards", "away_yellow_cards", [2.5, 3.5, 4.5, 5.5, 6.5]),
    "fouls":   ("home_fouls", "away_fouls", [19.5, 21.5, 23.5, 25.5]),
    "shots":   ("home_shots", "away_shots", [20.5, 22.5, 24.5, 26.5]),
    "sot":     ("home_sot", "away_sot", [6.5, 7.5, 8.5, 9.5]),
}
W = (5, 10, 19)
FOLDS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
EWM_HL = {"corners": 20, "yellows": 12, "fouls": 8, "sot": 12, "shots": 5}
STAKES_MARKETS = {"yellows", "fouls", "sot", "shots"}

df = pd.read_csv(ROOT / "data/processed/foundation_big5_multi_season.csv", low_memory=False)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df[df["season"] >= "2014-2015"].copy()
for name, (hc, ac, _) in EVENTS.items():
    df[hc] = pd.to_numeric(df[hc], errors="coerce")
    df[ac] = pd.to_numeric(df[ac], errors="coerce")
df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
parts = []
for rel in ["data/processed/enriched/understat_xg/walkforward_preds.csv",
            "data/processed/enriched/understat_xg/walkforward_preds_hist.csv"]:
    p = ROOT / rel
    if p.exists():
        parts.append(pd.read_csv(p)[["match_id", "lh", "la"]])
if parts:
    lam_df = pd.concat(parts, ignore_index=True).drop_duplicates("match_id")
    df = df.merge(lam_df, on="match_id", how="left")
else:
    df["lh"] = np.nan; df["la"] = np.nan
USE_LAM = df["lh"].notna().mean() > 0.3

def add_positions(full):
    full = full.sort_values("date").copy()
    rows = []
    for (_, _), g in full.groupby(["competition", "season"], sort=False):
        pts, played = {}, {}
        for r in g.itertuples(index=False):
            def rank(t):
                p_ = pts.get(t, 0)
                return 1 + sum(1 for v in pts.values() if v > p_)
            rows.append((r.match_id, rank(r.home_team), rank(r.away_team), played.get(r.home_team, 0)))
            hg, ag = r.home_goals, r.away_goals
            pts[r.home_team] = pts.get(r.home_team, 0) + (3 if hg > ag else (1 if hg == ag else 0))
            pts[r.away_team] = pts.get(r.away_team, 0) + (3 if ag > hg else (1 if hg == ag else 0))
            played[r.home_team] = played.get(r.home_team, 0) + 1
            played[r.away_team] = played.get(r.away_team, 0) + 1
    pos = pd.DataFrame(rows, columns=["match_id", "pos_home", "pos_away", "played_home"])
    return full.merge(pos, on="match_id", how="left")

df = df.dropna(subset=["home_goals", "away_goals"])
df = add_positions(df)


def make_long(m, hc, ac, hl=5):
    h_map = {"home_team": "team", "away_team": "opp", hc: "ev_for", ac: "ev_against",
             "home_goals": "gf", "away_goals": "ga"}
    a_map = {"away_team": "team", "home_team": "opp", ac: "ev_for", hc: "ev_against",
             "away_goals": "gf", "home_goals": "ga"}
    if USE_LAM:
        h_map.update({"lh": "lam_t", "la": "lam_o"})
        a_map.update({"la": "lam_t", "lh": "lam_o"})
    if "pos_home" in m.columns:
        h_map.update({"pos_home": "pos_t", "pos_away": "pos_o", "played_home": "played"})
        a_map.update({"pos_away": "pos_t", "pos_home": "pos_o", "played_home": "played"})
    base = ["match_id", "date"]
    h = m[base + list(h_map)].rename(columns=h_map).assign(is_home=1)
    a = m[base + list(a_map)].rename(columns=a_map).assign(is_home=0)
    lr = pd.concat([h, a], ignore_index=True).sort_values(["team", "date", "match_id"])
    tkey = lr["team"]
    for col in ["ev_for", "ev_against", "gf", "ga"]:
        shifted = lr.groupby("team", sort=False)[col].shift(1)
        g = shifted.groupby(tkey, sort=False)
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
    if USE_LAM:
        lr["delta_lam"] = lr["lam_t"] - lr["lam_o"]
        lr["abs_delta_lam"] = lr["delta_lam"].abs()
    if "pos_t" in lr.columns:
        lr["round_frac"] = (lr["played"] / 38.0).clip(0, 1)
        lr["pos_diff_abs"] = (lr["pos_t"] - lr["pos_o"]).abs()
        lr["releg_battle"] = (((lr["pos_t"] >= 15) | (lr["pos_o"] >= 15))
                              & (lr["round_frac"] > 0.6)).astype(float)
    return lr


def base_feats(market):
    f = ([f"ev_for_r{w}" for w in W] + ["ev_for_ewm"]
         + [f"opp_ev_against_r{w}" for w in W] + ["opp_ev_against_ewm"]
         + ["is_home", "delta", "abs_delta"])
    if USE_LAM:
        f += ["delta_lam", "abs_delta_lam"]
    if market in STAKES_MARKETS:
        f += ["pos_diff_abs", "releg_battle", "round_frac"]
    return f


# ======================================================================
# A. HETEROSCEDASTIC DISPERSION: disp = f(|delta|)
# ======================================================================
# Yip et al.: goal supremacy doesn't affect mean corners but DOES affect
# variance. Close matches = lower variance, lopsided = higher.
# Implementation: bin matches by |delta| into 3 groups (close/medium/far),
# compute NB dispersion per bin, use the matching bin at predict time.
print("=" * 70)
print("A. HETEROSCEDASTIC DISPERSION: NB dispersion as f(|delta|)")
print("=" * 70)

def compute_binned_disp(train_m, hc, ac, lr_train, feats):
    """Compute NB dispersion per |delta| bin from training data."""
    te_lr = lr_train.dropna(subset=feats + ["ev_for"]).copy()
    reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(te_lr[feats], te_lr["ev_for"].clip(lower=0))
    te_lr["pred"] = np.clip(reg.predict(te_lr[feats]), 0.1, 25)
    pv = te_lr.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
    pv_ad = te_lr.pivot_table(index="match_id", columns="is_home", values="abs_delta").dropna()
    tot_pred = pv[1] + pv[0]
    abs_d = pv_ad[1]  # same for both sides
    tei = train_m.set_index("match_id").loc[tot_pred.index]
    act = (tei[hc] + tei[ac]).astype(float)
    # bin by |delta|
    cuts = [0, 0.3, 0.8, 999]
    labels = ["close", "medium", "far"]
    bins = pd.cut(abs_d, cuts, labels=labels)
    disps = {}
    for label in labels:
        mask = bins == label
        if mask.sum() < 100:
            disps[label] = 1.2
            continue
        a = act[mask].to_numpy()
        p = tot_pred[mask].to_numpy()
        # method of moments: var(actual) / mean(pred) but accounting for Poisson component
        residual_var = np.var(a - p) - np.mean(p)  # excess variance over Poisson
        disps[label] = float(np.clip(1.0 + residual_var / max(np.mean(p), 1), 0.9, 3.0))
    return reg, disps


def eval_hetero_disp(lr, m, feats, hc, ac, lines, season):
    """Evaluate heteroscedastic dispersion vs global dispersion."""
    te_m = m[m.season == season]
    s0 = te_m.date.min()
    tr_m = m[m.date < s0]
    tr = lr[lr.date < s0].dropna(subset=feats + ["ev_for"])
    te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
    if len(tr) < 2000 or te.empty:
        return None, None
    # Global dispersion baseline
    reg_g = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
    te["pred"] = np.clip(reg_g.predict(te[feats]), 0.1, 25)
    pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
    if pv.empty:
        return None, None
    tot = pv[1] + pv[0]
    tei = te_m.set_index("match_id").loc[tot.index]
    act = (tei[hc] + tei[ac]).astype(float)
    tt = (tr_m[hc] + tr_m[ac]).astype(float)
    disp_global = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))

    # Heteroscedastic: compute binned disps from training data
    pv_ad = te.pivot_table(index="match_id", columns="is_home", values="abs_delta").dropna()
    abs_d_te = pv_ad.reindex(tot.index)[1].fillna(0.5)
    # Train binned dispersions
    _, binned = compute_binned_disp(tr_m, hc, ac, tr, feats)

    cuts = [0, 0.3, 0.8, 999]
    labels = ["close", "medium", "far"]
    bins_te = pd.cut(abs_d_te, cuts, labels=labels)
    disp_per_match = bins_te.map(binned).astype(float).fillna(disp_global)

    def ll_lines(lam_arr, disp_arr, actual, lines):
        lls = []
        for ln in lines:
            y = (actual > ln).astype(float).to_numpy()
            k = int(np.floor(ln))
            lam = np.clip(lam_arr, 0.2, 40)
            d = np.asarray(disp_arr)
            # per-match NB
            r_ = lam / np.clip(d - 1, 0.01, 10)
            p = 1 - nbinom.cdf(k, r_, 1.0 / d)
            p = np.clip(p, 1e-6, 1 - 1e-6)
            ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
            lls.append(ll)
        return float(np.mean(lls))

    ll_g = ll_lines(tot.to_numpy(), np.full(len(tot), disp_global), act, lines)
    ll_h = ll_lines(tot.to_numpy(), disp_per_match.to_numpy(), act, lines)
    return ll_g, ll_h


for name, (hc, ac, lines) in EVENTS.items():
    m = df.dropna(subset=[hc, ac]).copy()
    if len(m) < 5000:
        continue
    hl = EWM_HL.get(name, 5)
    lr = make_long(m, hc, ac, hl=hl)
    feats = base_feats(name)
    print(f"\n  {name.upper():>8}:")

    # First show empirical dispersion by |delta| bin
    for s in ["2024-2025"]:
        te_m = m[m.season == s]
        s0 = te_m.date.min()
        tr_m = m[m.date < s0]
        tr = lr[lr.date < s0].dropna(subset=feats + ["ev_for"])
        _, binned = compute_binned_disp(tr_m, hc, ac, tr, feats)
        tt_g = (tr_m[hc] + tr_m[ac]).astype(float)
        dg = float(np.clip(tt_g.var() / max(tt_g.mean(), 1e-9), 0.8, 3.0))
        print(f"    Empirical disp ({s}): global={dg:.3f}  close={binned['close']:.3f}  medium={binned['medium']:.3f}  far={binned['far']:.3f}")

    wins = 0
    for s in FOLDS:
        g, h = eval_hetero_disp(lr, m, feats, hc, ac, lines, s)
        if g is None:
            print(f"    {s}: skip")
            continue
        d = h - g
        tag = "<-- BETTER" if d < -0.0001 else ("WORSE" if d > 0.0001 else "~")
        if d < -0.0001:
            wins += 1
        print(f"    {s}: global={g:.5f}  hetero={h:.5f}  diff={d:+.5f}  {tag}")
    print(f"    WINS: {wins}/{len(FOLDS)}")


# ======================================================================
# B. XGBOOST WITH POISSON LOSS vs PoissonRegressor
# ======================================================================
print(f"\n{'='*70}")
print("B. XGBOOST (Poisson loss) vs sklearn PoissonRegressor")
print("=" * 70)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  xgboost not installed, trying lightgbm...")
    try:
        import lightgbm as lgb
        HAS_LGB = True
    except ImportError:
        HAS_LGB = False
        print("  Neither xgboost nor lightgbm installed. SKIP.")

if HAS_XGB or (not HAS_XGB and 'HAS_LGB' in dir() and HAS_LGB):
    for name in ["corners", "shots", "yellows"]:
        hc, ac, lines = EVENTS[name]
        m = df.dropna(subset=[hc, ac]).copy()
        if len(m) < 5000:
            continue
        hl = EWM_HL.get(name, 5)
        lr = make_long(m, hc, ac, hl=hl)
        feats = base_feats(name)
        print(f"\n  {name.upper():>8}:")
        wins_xgb = 0
        for s in FOLDS:
            te_m = m[m.season == s]
            s0 = te_m.date.min()
            tr = lr[lr.date < s0].dropna(subset=feats + ["ev_for"])
            te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
            if len(tr) < 2000 or te.empty:
                continue
            # sklearn baseline
            reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
            te["pred_sk"] = np.clip(reg.predict(te[feats]), 0.1, 25)

            # XGBoost/LightGBM
            if HAS_XGB:
                model = xgb.XGBRegressor(
                    objective="count:poisson", n_estimators=200, max_depth=4,
                    learning_rate=0.05, reg_alpha=0.1, reg_lambda=1.0,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    verbosity=0)
                model.fit(tr[feats], tr["ev_for"].clip(lower=0))
            else:
                model = lgb.LGBMRegressor(
                    objective="poisson", n_estimators=200, max_depth=4,
                    learning_rate=0.05, reg_alpha=0.1, reg_lambda=1.0,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    verbose=-1)
                model.fit(tr[feats], tr["ev_for"].clip(lower=0))
            te["pred_xgb"] = np.clip(model.predict(te[feats]), 0.1, 25)

            tt = (m[m.date < s0][hc] + m[m.date < s0][ac]).astype(float)
            disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))

            def score(pred_col):
                pv = te.pivot_table(index="match_id", columns="is_home", values=pred_col).dropna()
                if pv.empty:
                    return None
                tot = pv[1] + pv[0]
                tei = te_m.set_index("match_id").loc[tot.index]
                act = (tei[hc] + tei[ac]).astype(float)
                lls = []
                for ln in lines:
                    y = (act > ln).astype(float).to_numpy()
                    k = int(np.floor(ln))
                    lam = np.clip(tot.to_numpy(), 0.2, 40)
                    if disp > 1.1:
                        r_ = lam / (disp - 1)
                        p = 1 - nbinom.cdf(k, r_, 1/disp)
                    else:
                        p = 1 - poisson.cdf(k, lam)
                    p = np.clip(p, 1e-6, 1-1e-6)
                    ll = float(-(y * np.log(p) + (1-y) * np.log(1-p)).mean())
                    lls.append(ll)
                return float(np.mean(lls))

            ll_sk = score("pred_sk")
            ll_xgb = score("pred_xgb")
            if ll_sk is None or ll_xgb is None:
                continue
            d = ll_xgb - ll_sk
            lib = "XGB" if HAS_XGB else "LGB"
            tag = f"<-- {lib} BETTER" if d < -0.0001 else (f"sklearn BETTER" if d > 0.0001 else "~")
            if d < -0.0001:
                wins_xgb += 1
            print(f"    {s}: sklearn={ll_sk:.5f}  {lib}={ll_xgb:.5f}  diff={d:+.5f}  {tag}")
        print(f"    {lib} WINS: {wins_xgb}/{len(FOLDS)}")


# ======================================================================
# C. GEOMETRIC-POISSON APPLICABILITY (corner clustering)
# ======================================================================
print(f"\n{'='*70}")
print("C. GEOMETRIC-POISSON: is corner clustering real in our data?")
print("=" * 70)
# If corners cluster (a corner leads to another corner via rebounds/clearances),
# the geometric-Poisson fits better than NB. We test this by measuring the
# variance-to-mean ratio and comparing NB vs Poisson log-likelihood.
# Also check: is the SIDE-level dispersion different from the TOTAL dispersion?
# (Clustering would inflate total more than side.)
for name in ["corners", "yellows", "shots"]:
    hc, ac, _ = EVENTS[name]
    h = pd.to_numeric(df[hc], errors="coerce")
    a = pd.to_numeric(df[ac], errors="coerce")
    valid = h.notna() & a.notna()
    hv, av, tv = h[valid], a[valid], (h+a)[valid]
    vmr_side = float((hv.var() + av.var()) / 2 / max((hv.mean() + av.mean()) / 2, 1))
    vmr_total = float(tv.var() / max(tv.mean(), 1))
    # If clustering is real, VMR(total) > VMR(side) because clustering adds
    # covariance between home and away corners
    print(f"  {name:>8}: VMR(side)={vmr_side:.4f}  VMR(total)={vmr_total:.4f}  "
          f"ratio={vmr_total/max(vmr_side,0.01):.3f}  "
          f"{'CLUSTERING' if vmr_total > vmr_side * 1.05 else 'no clustering'}")

print("\n  Geometric-Poisson conclusion:")
print("  If VMR(total) >> VMR(side), clustering is real and geometric-Poisson")
print("  would help. If ratio ~ 1.0, the NB already captures the overdispersion.")

# ======================================================================
# D. POSSESSION DATA AVAILABILITY
# ======================================================================
print(f"\n{'='*70}")
print("D. POSSESSION: data availability and correlation with events")
print("=" * 70)
poss_cols = [c for c in df.columns if "poss" in c.lower()]
print(f"  Possession columns in foundation: {poss_cols}")
if poss_cols:
    for pc in poss_cols:
        df[pc] = pd.to_numeric(df[pc], errors="coerce")
        cov = df[pc].notna().mean() * 100
        print(f"    {pc}: {cov:.1f}% coverage")
        if cov > 30:
            for name, (hc, ac, _) in EVENTS.items():
                t = pd.to_numeric(df[hc], errors="coerce") + pd.to_numeric(df[ac], errors="coerce")
                r = t.corr(df[pc])
                print(f"      r({pc}, total_{name}) = {r:+.3f}")
else:
    print("  No possession columns found.")
    # Check ESPN team match data
    espn_p = ROOT / "data/processed/espn_team_match_current.csv"
    if espn_p.exists():
        espn = pd.read_csv(espn_p)
        poss_c = [c for c in espn.columns if "poss" in c.lower()]
        print(f"  ESPN team match current has: {poss_c}")
        if poss_c:
            print(f"    Rows: {len(espn)}, coverage: {espn[poss_c[0]].notna().mean()*100:.1f}%")
    # Check SofaScore data
    for pat in ["data/processed/sofascore*", "data/processed/enriched/sofascore*"]:
        import glob
        for fp in glob.glob(str(ROOT / pat)):
            ss = pd.read_csv(fp, nrows=5)
            poss_c = [c for c in ss.columns if "poss" in c.lower()]
            if poss_c:
                print(f"  {Path(fp).name} has possession: {poss_c}")

print(f"\n{'='*70}")
print("DONE round 6.")
print("=" * 70)
