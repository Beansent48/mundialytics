from __future__ import annotations

"""Test idea 1 (home/away-split rolling xG form) and idea 2 (time-varying home
advantage via a reduced-crowd/COVID flag) for the xG-rate lambda source.

Walk-forward rolling features (shifted), temporal folds. Each variant's predicted
match xG is blended 0.60*xr + 0.40*goals-AD (deployed config) and scored on 1X2 RPS.
Variants:
  A baseline : mixed rolling form + global is_home  (= current deployed model)
  B venue    : rolling form split by venue (home matches vs away matches)
  C venue+cov: B plus a reduced-crowd flag and its is_home interaction
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.statistical_core.attack_defense_model import AttackDefenseModel  # noqa: E402
from mundialytics.statistical_core.distributions import outcome_probabilities  # noqa: E402

COVERED = ["2014-2015","2015-2016","2016-2017","2017-2018","2018-2019","2019-2020",
           "2020-2021","2021-2022","2022-2023","2023-2024","2024-2025","2025-2026"]
TEST_SEASONS = ["2020-2021","2021-2022","2022-2023","2023-2024","2024-2025"]
WINDOWS = (5, 10, 19)
# Broadly empty / heavily-reduced stadiums across the Big5.
COVID_LO, COVID_HI = pd.Timestamp("2020-03-08"), pd.Timestamp("2021-06-30")


def build(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        rows.append({"match_id": r.match_id, "date": r.date, "team": r.home_team, "opp": r.away_team,
                     "is_home": 1, "xg_for": r.home_xg, "xg_against": r.away_xg})
        rows.append({"match_id": r.match_id, "date": r.date, "team": r.away_team, "opp": r.home_team,
                     "is_home": 0, "xg_for": r.away_xg, "xg_against": r.home_xg})
    lr = pd.DataFrame(rows).sort_values(["team", "date", "match_id"])
    # mixed rolling (baseline)
    for col in ["xg_for", "xg_against"]:
        for w in WINDOWS:
            lr[f"{col}_mix{w}"] = lr.groupby("team", group_keys=False)[col].apply(lambda s: s.shift(1).rolling(w, min_periods=3).mean())
    # venue-split rolling (own form at this venue)
    for col in ["xg_for", "xg_against"]:
        for w in WINDOWS:
            lr[f"{col}_ven{w}"] = lr.groupby(["team", "is_home"], group_keys=False)[col].apply(lambda s: s.shift(1).rolling(w, min_periods=2).mean())
    # opponent's against-rate at the OPPOSITE venue (join the two rows of each match)
    opp_ven = lr[["match_id", "team"] + [f"xg_against_ven{w}" for w in WINDOWS]].rename(
        columns={"team": "opp", **{f"xg_against_ven{w}": f"opp_against_ven{w}" for w in WINDOWS}})
    opp_mix = lr[["match_id", "team"] + [f"xg_against_mix{w}" for w in WINDOWS]].rename(
        columns={"team": "opp", **{f"xg_against_mix{w}": f"opp_against_mix{w}" for w in WINDOWS}})
    lr = lr.merge(opp_ven, on=["match_id", "opp"]).merge(opp_mix, on=["match_id", "opp"])
    lr["covid"] = ((lr.date >= COVID_LO) & (lr.date <= COVID_HI)).astype(float)
    lr["cov_home"] = lr.covid * lr.is_home
    return lr


def fit_predict(lr, feats, s_start, test_ids):
    tr = lr[lr.date < s_start].dropna(subset=feats + ["xg_for"])
    te = lr[lr.match_id.isin(test_ids)].dropna(subset=feats)
    m = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["xg_for"].clip(lower=0))
    te = te.copy(); te["pred"] = np.clip(m.predict(te[feats]), 0.05, 6.0)
    return te.set_index(["match_id", "is_home"])["pred"]


def rps(lam_h, lam_a, ad, test):
    r = []
    for _, row in test.iterrows():
        adl = ad.expected_goals(row.home_team, row.away_team, int(row.get("neutral", 0)), row.competition)[:2]
        lh = 0.6 * lam_h.get(row.match_id, np.nan) + 0.4 * adl[0]
        la = 0.6 * lam_a.get(row.match_id, np.nan) + 0.4 * adl[1]
        if np.isnan(lh) or np.isnan(la):
            continue
        p = outcome_probabilities(float(np.clip(lh, .05, 6)), float(np.clip(la, .05, 6)), max_goals=10, dixon_coles_rho=-0.07)
        oh = row.oc == "home"; od = row.oc == "draw"
        r.append(0.5 * ((p["p_home_win"] - oh) ** 2 + (p["p_home_win"] + p["p_draw"] - oh - od) ** 2))
    return float(np.mean(r)), len(r)


def main() -> None:
    df = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv", low_memory=False)
    df = df[(df.season.isin(COVERED)) & (df.xg_available == True)].copy()  # noqa: E712
    for c in ["home_goals","away_goals","home_xg","away_xg"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df.date, errors="coerce")
    df = df.dropna(subset=["home_goals","away_goals","home_xg","away_xg","date"]).sort_values("date")
    df["oc"] = np.where(df.home_goals > df.away_goals, "home", np.where(df.home_goals < df.away_goals, "away", "draw"))
    lr = build(df)

    F_mix = [f"xg_for_mix{w}" for w in WINDOWS] + [f"opp_against_mix{w}" for w in WINDOWS] + ["is_home"]
    F_ven = [f"xg_for_ven{w}" for w in WINDOWS] + [f"opp_against_ven{w}" for w in WINDOWS] + ["is_home"]
    F_cov = F_ven + ["covid", "cov_home"]

    agg = {"A": [], "B": [], "C": []}
    print(f"{'season':10s} {'A mixed':>9s} {'B venue':>9s} {'C +covid':>9s}")
    for s in TEST_SEASONS:
        test = df[df.season == s].copy(); s_start = test.date.min(); train = df[df.date < s_start]
        if len(test) == 0 or len(train) < 500: continue
        t0 = time.time()
        ad = AttackDefenseModel(time_decay_half_life=None).fit(train)
        tids = set(test.match_id)
        res = {}
        for key, F in [("A", F_mix), ("B", F_ven), ("C", F_cov)]:
            pred = fit_predict(lr, F, s_start, tids)
            lam_h = {mid: pred.get((mid, 1), np.nan) for mid in tids}
            lam_a = {mid: pred.get((mid, 0), np.nan) for mid in tids}
            res[key] = rps(lam_h, lam_a, ad, test)
            agg[key].append(res[key])
        print(f"{s:10s} {res['A'][0]:9.4f} {res['B'][0]:9.4f} {res['C'][0]:9.4f}   ({time.time()-t0:.0f}s)", flush=True)

    def pool(a): return sum(r * n for r, n in a) / sum(n for _, n in a)
    print("\n===== POOLED RPS =====")
    for k, name in [("A", "mixed (baseline)"), ("B", "venue-split"), ("C", "venue-split + covid")]:
        print(f"  {name:22s} {pool(agg[k]):.4f}   dvs-A {pool(agg[k])-pool(agg['A']):+.4f}")


if __name__ == "__main__":
    main()
