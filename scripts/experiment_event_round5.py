"""Event-model round 5: deep exploration of untapped signals.

Candidates (from explore_events.py diagnostics):
  A. Opponent ev_FOR rollings — partial r: shots +0.202, fouls +0.189, SOT +0.159
     Currently we only use opp_ev_AGAINST (how many events opponents concede to
     this team's opponent); opp_ev_FOR captures "how aggressive is the opponent"
     — a genuinely different signal.
  B. Goals-total lambda as feature — r(goals, SOT) = +0.544, r(goals, shots) = +0.257.
     We have delta_lam and |delta_lam| but NOT lam_total = lam_t + lam_o.
     "Is this a high-scoring game" is distinct from "who's favored."
  C. xG rollings as features — 97.4% coverage; rolling team xG captures shot
     quality, different from shot quantity (ev_for).

Walk-forward 5-fold validation on O/U log-loss, same protocol as rounds 1-4.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor

ROOT = Path(r"C:\Users\Vicente\Desktop\BetBot\mundialytics_betting_engine")
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore", category=UserWarning)
EVENTS = {
    "corners": ("home_corners", "away_corners", [7.5, 8.5, 9.5, 10.5, 11.5]),
    "yellows": ("home_yellow_cards", "away_yellow_cards", [2.5, 3.5, 4.5, 5.5, 6.5]),
    "fouls":   ("home_fouls", "away_fouls", [19.5, 21.5, 23.5, 25.5]),
    "shots":   ("home_shots", "away_shots", [20.5, 22.5, 24.5, 26.5]),
    "sot":     ("home_sot", "away_sot", [6.5, 7.5, 8.5, 9.5]),
}
W = (5, 10, 19)
FOLDS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
# deployed EWMA halflives
EWM_HL = {"corners": 20, "yellows": 12, "fouls": 8, "sot": 12, "shots": 5}

df = pd.read_csv(ROOT / "data/processed/foundation_big5_multi_season.csv", low_memory=False)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df[df["season"] >= "2014-2015"].copy()
for name, (hc, ac, _) in EVENTS.items():
    df[hc] = pd.to_numeric(df[hc], errors="coerce")
    df[ac] = pd.to_numeric(df[ac], errors="coerce")
df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")

# load walk-forward lambdas
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

# load xG
for col in ["home_xg", "away_xg"]:
    df[col] = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")

# -- Stakes features (walk-forward positions) -----
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

STAKES_MARKETS = {"yellows", "fouls", "sot", "shots"}
df = df.dropna(subset=["home_goals", "away_goals"])
df = add_positions(df)


def make_long(m, hc, ac, hl=5, include_opp_for=False, include_lam_total=False, include_xg=False):
    """Build long-format rows; optionally add new features."""
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
    if include_xg:
        h_map.update({"home_xg": "xg_for", "away_xg": "xg_against"})
        a_map.update({"away_xg": "xg_for", "home_xg": "xg_against"})
    base = ["match_id", "date"]
    h = m[base + list(h_map)].rename(columns=h_map).assign(is_home=1)
    a = m[base + list(a_map)].rename(columns=a_map).assign(is_home=0)
    lr = pd.concat([h, a], ignore_index=True).sort_values(["team", "date", "match_id"])
    tkey = lr["team"]
    cols_to_roll = ["ev_for", "ev_against", "gf", "ga"]
    if include_xg:
        cols_to_roll += ["xg_for", "xg_against"]
    for col in cols_to_roll:
        shifted = lr.groupby("team", sort=False)[col].shift(1)
        g = shifted.groupby(tkey, sort=False)
        for w in W:
            lr[f"{col}_r{w}"] = g.rolling(w, min_periods=3).mean().reset_index(level=0, drop=True)
        use_hl = hl if col in ("ev_for", "ev_against") else 5
        lr[f"{col}_ewm"] = g.ewm(halflife=use_hl, min_periods=3).mean().reset_index(level=0, drop=True)
    # opponent's ev_against rollings (what we already have)
    opp_src = [f"ev_against_r{w}" for w in W] + ["ev_against_ewm", "gf_ewm", "ga_ewm"]
    if include_opp_for:
        opp_src += [f"ev_for_r{w}" for w in W] + ["ev_for_ewm"]
    if include_xg:
        opp_src += [f"xg_for_r{w}" for w in W] + ["xg_for_ewm"]
    opp = lr[["match_id", "team"] + opp_src].rename(
        columns={"team": "opp", **{c: f"opp_{c}" for c in opp_src}})
    lr = lr.merge(opp, on=["match_id", "opp"], how="left")
    lr["delta"] = (lr["gf_ewm"] + lr["opp_ga_ewm"]) / 2 - (lr["opp_gf_ewm"] + lr["ga_ewm"]) / 2
    lr["abs_delta"] = lr["delta"].abs()
    if USE_LAM:
        lr["delta_lam"] = lr["lam_t"] - lr["lam_o"]
        lr["abs_delta_lam"] = lr["delta_lam"].abs()
        if include_lam_total:
            lr["lam_total"] = lr["lam_t"] + lr["lam_o"]
    if "pos_t" in lr.columns:
        lr["round_frac"] = (lr["played"] / 38.0).clip(0, 1)
        lr["pos_diff_abs"] = (lr["pos_t"] - lr["pos_o"]).abs()
        lr["releg_battle"] = (((lr["pos_t"] >= 15) | (lr["pos_o"] >= 15))
                              & (lr["round_frac"] > 0.6)).astype(float)
    return lr


def base_feats(market, lam=True, stakes=True):
    f = ([f"ev_for_r{w}" for w in W] + ["ev_for_ewm"]
         + [f"opp_ev_against_r{w}" for w in W] + ["opp_ev_against_ewm"]
         + ["is_home", "delta", "abs_delta"])
    if lam and USE_LAM:
        f += ["delta_lam", "abs_delta_lam"]
    if stakes and market in STAKES_MARKETS:
        f += ["pos_diff_abs", "releg_battle", "round_frac"]
    return f


def eval_ou(lr, m, feats, hc, ac, lines, season, disp_global=None, league_disp=None):
    """Single fold: train < season, test = season. Returns avg LL over lines."""
    te_m = m[m.season == season]
    s0 = te_m.date.min()
    tr = lr[lr.date < s0].dropna(subset=feats + ["ev_for"])
    te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
    if len(tr) < 2000 or te.empty:
        return None
    reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
    te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 25)
    pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
    if pv.empty:
        return None
    tot = pv[1] + pv[0]
    tei = te_m.set_index("match_id").loc[tot.index]
    act = (tei[hc] + tei[ac]).astype(float)

    if disp_global is None:
        tt = (m[m.date < s0][hc] + m[m.date < s0][ac]).astype(float)
        disp_global = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))

    lls = []
    for ln in lines:
        y = (act > ln).astype(float).to_numpy()
        k = int(np.floor(ln))
        lam = np.clip(tot.to_numpy(), 0.2, 40)
        if disp_global > 1.1:
            r_ = lam / (disp_global - 1)
            p = 1 - nbinom.cdf(k, r_, 1/disp_global)
        else:
            p = 1 - poisson.cdf(k, lam)
        p = np.clip(p, 1e-6, 1-1e-6)
        ll = float(-(y * np.log(p) + (1-y) * np.log(1-p)).mean())
        lls.append(ll)
    return float(np.mean(lls))


# ======================================================================
# A. OPPONENT EV_FOR ROLLINGS
# ======================================================================
print("=" * 70)
print("A. OPPONENT EV_FOR ROLLINGS (the opponent's OWN attack generation rate)")
print("=" * 70)
for name, (hc, ac, lines) in EVENTS.items():
    m = df.dropna(subset=[hc, ac]).copy()
    if len(m) < 5000:
        continue
    hl = EWM_HL.get(name, 5)
    # baseline: without opp_ev_for
    lr_base = make_long(m, hc, ac, hl=hl, include_opp_for=False)
    feats_base = base_feats(name)
    # candidate: with opp_ev_for
    lr_cand = make_long(m, hc, ac, hl=hl, include_opp_for=True)
    feats_cand = feats_base + [f"opp_ev_for_r{w}" for w in W] + ["opp_ev_for_ewm"]
    print(f"\n  {name.upper():>8}:")
    wins = 0
    for s in FOLDS:
        b = eval_ou(lr_base, m, feats_base, hc, ac, lines, s)
        c = eval_ou(lr_cand, m, feats_cand, hc, ac, lines, s)
        if b is None or c is None:
            print(f"    {s}: skip")
            continue
        d = c - b
        tag = "<-- BETTER" if d < -0.0001 else ("WORSE" if d > 0.0001 else "~")
        if d < -0.0001:
            wins += 1
        print(f"    {s}: base={b:.5f}  +opp_for={c:.5f}  diff={d:+.5f}  {tag}")
    print(f"    WINS: {wins}/{len(FOLDS)}")

# ======================================================================
# B. GOALS-TOTAL LAMBDA AS FEATURE (lam_t + lam_o)
# ======================================================================
print(f"\n{'='*70}")
print("B. GOALS-TOTAL LAMBDA (lam_t + lam_o as direct feature)")
print("=" * 70)
if USE_LAM:
    for name in ["sot", "shots", "corners"]:
        hc, ac, lines = EVENTS[name]
        m = df.dropna(subset=[hc, ac]).copy()
        if len(m) < 5000:
            continue
        hl = EWM_HL.get(name, 5)
        lr_base = make_long(m, hc, ac, hl=hl, include_lam_total=False)
        feats_base = base_feats(name)
        lr_cand = make_long(m, hc, ac, hl=hl, include_lam_total=True)
        feats_cand = feats_base + ["lam_total"]
        print(f"\n  {name.upper():>8}:")
        wins = 0
        for s in FOLDS:
            b = eval_ou(lr_base, m, feats_base, hc, ac, lines, s)
            c = eval_ou(lr_cand, m, feats_cand, hc, ac, lines, s)
            if b is None or c is None:
                print(f"    {s}: skip")
                continue
            d = c - b
            tag = "<-- BETTER" if d < -0.0001 else ("WORSE" if d > 0.0001 else "~")
            if d < -0.0001:
                wins += 1
            print(f"    {s}: base={b:.5f}  +lam_total={c:.5f}  diff={d:+.5f}  {tag}")
        print(f"    WINS: {wins}/{len(FOLDS)}")
else:
    print("  SKIP: no lambda coverage")

# ======================================================================
# C. xG ROLLINGS AS FEATURE (shot quality proxy)
# ======================================================================
print(f"\n{'='*70}")
print("C. xG ROLLINGS (rolling team xG for/against as features)")
print("=" * 70)
for name in ["sot", "shots", "corners"]:
    hc, ac, lines = EVENTS[name]
    m = df.dropna(subset=[hc, ac, "home_xg", "away_xg"]).copy()
    if len(m) < 5000:
        print(f"\n  {name.upper():>8}: SKIP (n={len(m)} < 5000)")
        continue
    hl = EWM_HL.get(name, 5)
    lr_base = make_long(m, hc, ac, hl=hl, include_xg=False)
    feats_base = base_feats(name)
    lr_cand = make_long(m, hc, ac, hl=hl, include_xg=True)
    # own xG rollings + opponent's xG rollings
    feats_cand = feats_base + [f"xg_for_r{w}" for w in W] + ["xg_for_ewm"] + [f"opp_xg_for_r{w}" for w in W] + ["opp_xg_for_ewm"]
    print(f"\n  {name.upper():>8}:")
    wins = 0
    for s in FOLDS:
        b = eval_ou(lr_base, m, feats_base, hc, ac, lines, s)
        c = eval_ou(lr_cand, m, feats_cand, hc, ac, lines, s)
        if b is None or c is None:
            print(f"    {s}: skip")
            continue
        d = c - b
        tag = "<-- BETTER" if d < -0.0001 else ("WORSE" if d > 0.0001 else "~")
        if d < -0.0001:
            wins += 1
        print(f"    {s}: base={b:.5f}  +xg_roll={c:.5f}  diff={d:+.5f}  {tag}")
    print(f"    WINS: {wins}/{len(FOLDS)}")

# ======================================================================
# D. COMBINED: opp_for + lam_total (best ideas together)
# ======================================================================
print(f"\n{'='*70}")
print("D. COMBINED: opp_ev_for + lam_total (stacked)")
print("=" * 70)
if USE_LAM:
    for name in ["sot", "shots"]:
        hc, ac, lines = EVENTS[name]
        m = df.dropna(subset=[hc, ac]).copy()
        if len(m) < 5000:
            continue
        hl = EWM_HL.get(name, 5)
        lr_base = make_long(m, hc, ac, hl=hl)
        feats_base = base_feats(name)
        lr_comb = make_long(m, hc, ac, hl=hl, include_opp_for=True, include_lam_total=True)
        feats_comb = (feats_base + [f"opp_ev_for_r{w}" for w in W] + ["opp_ev_for_ewm"]
                      + ["lam_total"])
        print(f"\n  {name.upper():>8}:")
        wins = 0
        for s in FOLDS:
            b = eval_ou(lr_base, m, feats_base, hc, ac, lines, s)
            c = eval_ou(lr_comb, m, feats_comb, hc, ac, lines, s)
            if b is None or c is None:
                print(f"    {s}: skip")
                continue
            d = c - b
            tag = "<-- BETTER" if d < -0.0001 else ("WORSE" if d > 0.0001 else "~")
            if d < -0.0001:
                wins += 1
            print(f"    {s}: base={b:.5f}  +combined={c:.5f}  diff={d:+.5f}  {tag}")
        print(f"    WINS: {wins}/{len(FOLDS)}")
else:
    print("  SKIP: no lambda coverage")

print("\n" + "=" * 70)
print("DONE. Deploy any idea with >= 4/5 wins.")
print("=" * 70)
