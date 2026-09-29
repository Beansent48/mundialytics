from __future__ import annotations

"""Player props: does the league-SEASON level fix player shots and cards too?

The live 2026/27 log has player shots under-predicted (says 14% over, 17%
happen) and player yellows over-predicted (14% vs 12%) — the same season-level
shift as the team markets (shots +12%, yellows -7% in all five leagues). The
player recipe carries no league term: shots are a career rate blended 50/50
with the last 15 appearances, cards are career-only.

Variant: mu x f^beta, with
  f = league-season level before this date, shrunk to the previous season
      with K matches   (the team experiment's lg_lvl, scripts/experiment_league_level.py)
      / the league's mean over the three seasons before (the era a career rate
      mostly reflects)
The level comes from the foundation's team totals (football-data), so the same
number is available when serving. Applied to shots (both lines) and yellows;
goals and assists are left alone (early goal shifts are noise — measured).

Harness = scripts/backtest_player_props.py (imported, identical panel and
recipe), folds 2021/22-2025/26, scored against the deployed mu.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import backtest_player_props as bp  # noqa: E402

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
KS = [40, 70, 120]
BETAS = [0.5, 1.0]
STATS = {"shots": ("home_shots", "away_shots"), "yellow_cards": ("home_yellow_cards", "away_yellow_cards")}


def p_ge(mu, k, disp=1.0):
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, 10)
    if disp > 1.05:
        r = mu / (disp - 1.0)
        return 1 - nbinom.cdf(k - 1, r, 1.0 / disp)
    return 1 - poisson.cdf(k - 1, mu)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def level_factors(k: int) -> pd.DataFrame:
    """(competition, date) -> factor per stat: shrunk current level / previous-3 mean."""
    f = pd.read_csv(FOUND, low_memory=False)
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    f = f[f.season >= "2010-2011"].dropna(subset=["date"])
    out = None
    for stat, (hc, ac) in STATS.items():
        g0 = f.dropna(subset=[hc, ac]).copy()
        g0["tot"] = pd.to_numeric(g0[hc], errors="coerce") + pd.to_numeric(g0[ac], errors="coerce")
        smean = g0.groupby(["competition", "season"])["tot"].mean()
        rows = []
        for (comp, season), g in g0.groupby(["competition", "season"]):
            prevs = sorted(s for (c, s) in smean.index if c == comp and s < season)
            if not prevs:
                continue
            prev = smean[(comp, prevs[-1])]
            ref = float(np.mean([smean[(comp, s)] for s in prevs[-3:]]))
            by_day = g.groupby("date")["tot"].agg(["sum", "size"])
            cs = by_day["sum"].cumsum().shift(1, fill_value=0.0)
            cn = by_day["size"].cumsum().shift(1, fill_value=0)
            lvl = (cs + k * prev) / (cn + k)
            rows.append(pd.DataFrame({"competition": comp, "date": by_day.index,
                                      f"f_{stat}": (lvl / ref).to_numpy()}))
        d = pd.concat(rows, ignore_index=True)
        out = d if out is None else out.merge(d, on=["competition", "date"], how="outer")
    return out


def main() -> None:
    t0 = time.time()
    pm = bp.add_context(bp.build_panel())
    found = pd.read_csv(bp.FOUND, low_memory=False, usecols=["provider_match_id", "competition"])
    found["game_id"] = pd.to_numeric(found["provider_match_id"], errors="coerce")
    pm = pm.merge(found.dropna(subset=["game_id"]).drop_duplicates("game_id")[["game_id", "competition"]],
                  on="game_id", how="left")
    print(f"panel ready ({time.time() - t0:.0f}s), competition known for "
          f"{pm['competition'].notna().mean():.1%}", flush=True)

    played = pm["minutes"] > 0
    train = pm[(~pm["season"].isin(bp.TEST_SEASONS)) & played]
    test_mask = pm["season"].isin(bp.TEST_SEASONS) & pm["team_lam"].notna() & played
    cols = ["xg", "goals", "shots", "xa", "assists", "yellow_cards"]
    pri = train.groupby("pgroup").apply(
        lambda gr: pd.Series({c: gr[c].sum() / max(gr["minutes"].sum(), 1) * 90.0 for c in cols}),
        include_groups=False)
    glob = {c: train[c].sum() / max(train["minutes"].sum(), 1) * 90.0 for c in cols}
    for c, w in {"shots": 0.5, "yellow_cards": 0.0}.items():
        prior = pm["pgroup"].map(pri[c]).fillna(glob[c])
        r_car = bp.shrunk_rate(pm[f"c_{c}"], pm["cmin"], prior)
        if w > 0:
            r_rec = bp.shrunk_rate(pm[f"rr_{c}"], pm["rmin15"], r_car, k=450.0)
            pm[f"r_{c}"] = w * r_rec + (1 - w) * r_car
        else:
            pm[f"r_{c}"] = r_car
    pos_min = train.groupby("pgroup")["minutes"].mean()
    prior_min = pm["pgroup"].map(pos_min).fillna(float(train["minutes"].mean()))
    cred_m = pm["nplayed10"].fillna(0) / (pm["nplayed10"].fillna(0) + 3.0)
    pm["exp_min"] = cred_m * pm["avg_minp10"].fillna(prior_min) + (1 - cred_m) * prior_min
    emins = pm["exp_min"].clip(20, 95) / 90.0
    af = pm["atk_factor"].fillna(1.0) ** 0.7
    pm["mu_shots"] = pm["r_shots"] * emins * af
    pm["mu_yc"] = pm["r_yellow_cards"] * emins ** 0.7
    test = pm[test_mask].copy()
    seas = test["season"].to_numpy()

    targets = {
        "shots>1.5": ("shots", lambda mu: p_ge(mu, 2, bp_disp), (test["shots"] >= 2)),
        "shots>2.5": ("shots", lambda mu: p_ge(mu, 3, bp_disp), (test["shots"] >= 3)),
        "yellow":    ("yellow_cards", lambda mu: p_ge(mu, 1), (test["yellow_cards"] >= 1)),
    }
    bp_disp = 1.3   # deployed SHOTS_DISP (train-picked in the harness)
    mu_col = {"shots": "mu_shots", "yellow_cards": "mu_yc"}

    for k in KS:
        lf = level_factors(k)
        tt = test.merge(lf, on=["competition", "date"], how="left")
        print(f"\n===== K={k}  (factor known for {tt['f_shots'].notna().mean():.1%}; "
              f"mean f_shots {tt['f_shots'].mean():.3f}, f_yc {tt['f_yellow_cards'].mean():.3f}) =====",
              flush=True)
        for name, (stat, pf, yser) in targets.items():
            y = yser.astype(float).to_numpy()
            mu0 = tt[mu_col[stat]].to_numpy()
            fac = tt[f"f_{stat}"].fillna(1.0).to_numpy()
            p0 = pf(mu0)
            line = [f"{name:10s} DEP {bll(y, p0):.5f}"]
            for beta in BETAS:
                p1 = pf(mu0 * fac ** beta)
                wins = sum(bll(y[seas == s], p1[seas == s]) < bll(y[seas == s], p0[seas == s])
                           for s in bp.TEST_SEASONS)
                line.append(f"b{beta:g} {bll(y, p1) - bll(y, p0):+.5f} ({wins}/5)")
            print("  " + " | ".join(line), flush=True)
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
