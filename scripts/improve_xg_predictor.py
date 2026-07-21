from __future__ import annotations

"""Can we predict pre-match xG better, and does it help 1X2?

Builds dedicated pre-match team-xG predictors and evaluates BOTH:
  (a) how well predicted pre-match xG tracks the ACTUAL match xG (corr/MAE) — the
      predictability ceiling; and
  (b) 1X2 RPS when the predicted xG is used directly as the scoreline lambda,
      vs a goals-features predictor of the same form.

Leakage-safe: rolling features are shifted; temporal train/test split.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.statistical_core.distributions import outcome_probabilities  # noqa: E402

COVERED = ["2014-2015","2015-2016","2016-2017","2017-2018","2018-2019","2019-2020",
           "2020-2021","2021-2022","2022-2023","2023-2024","2024-2025","2025-2026"]
TEST_FROM = "2021-08-01"
WINDOWS = (5, 10, 19)


def long_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        rows.append({"match_id": r["match_id"], "date": r["date"], "competition": r["competition"],
                     "team": r["home_team"], "opp": r["away_team"], "is_home": 1,
                     "xg_for": r["home_xg"], "xg_against": r["away_xg"],
                     "g_for": r["home_goals"], "g_against": r["away_goals"]})
        rows.append({"match_id": r["match_id"], "date": r["date"], "competition": r["competition"],
                     "team": r["away_team"], "opp": r["home_team"], "is_home": 0,
                     "xg_for": r["away_xg"], "xg_against": r["home_xg"],
                     "g_for": r["away_goals"], "g_against": r["home_goals"]})
    return pd.DataFrame(rows)


def add_rolling(lr: pd.DataFrame) -> pd.DataFrame:
    lr = lr.sort_values(["team", "date", "match_id"]).copy()
    for col in ["xg_for", "xg_against", "g_for", "g_against"]:
        for w in WINDOWS:
            lr[f"{col}_r{w}"] = (lr.groupby("team", group_keys=False)[col]
                                 .apply(lambda s: s.shift(1).rolling(w, min_periods=3).mean()))
    # opponent's rolling defensive/offensive rates, joined by opp+match
    opp = lr[["match_id", "team"] + [f"xg_against_r{w}" for w in WINDOWS] + [f"xg_for_r{w}" for w in WINDOWS]
             + [f"g_against_r{w}" for w in WINDOWS] + [f"g_for_r{w}" for w in WINDOWS]].copy()
    opp = opp.rename(columns={"team": "opp", **{f"xg_against_r{w}": f"opp_xg_against_r{w}" for w in WINDOWS},
                              **{f"xg_for_r{w}": f"opp_xg_for_r{w}" for w in WINDOWS},
                              **{f"g_against_r{w}": f"opp_g_against_r{w}" for w in WINDOWS},
                              **{f"g_for_r{w}": f"opp_g_for_r{w}" for w in WINDOWS}})
    return lr.merge(opp, on=["match_id", "opp"], how="left")


def fit_eval(lr: pd.DataFrame, feats: list[str], label: str) -> pd.Series:
    tr = lr[lr["date"] < TEST_FROM].dropna(subset=feats + ["xg_for"])
    te = lr[lr["date"] >= TEST_FROM].dropna(subset=feats + ["xg_for"])
    m = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["xg_for"].clip(lower=0))
    pred = np.clip(m.predict(te[feats]), 0.05, 6.0)
    te = te.copy(); te[f"pred_{label}"] = pred
    return te.set_index(["match_id", "is_home"])[f"pred_{label}"]


def main() -> None:
    df = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv", low_memory=False)
    df = df[(df["season"].isin(COVERED)) & (df["xg_available"] == True)].copy()  # noqa: E712
    for c in ["home_goals","away_goals","home_xg","away_xg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_goals","away_goals","home_xg","away_xg","date"]).sort_values("date")

    lr = add_rolling(long_rows(df))

    xg_feats = [f"xg_for_r{w}" for w in WINDOWS] + [f"opp_xg_against_r{w}" for w in WINDOWS] + ["is_home"]
    goal_feats = [f"g_for_r{w}" for w in WINDOWS] + [f"opp_g_against_r{w}" for w in WINDOWS] + ["is_home"]

    pred_xg = fit_eval(lr, xg_feats, "xg")
    pred_gl = fit_eval(lr, goal_feats, "gl")

    # reassemble per match (home/away)
    te = df[df["date"] >= TEST_FROM].copy()
    def lam(pred, mid, home):
        try: return float(pred.loc[(mid, home)])
        except KeyError: return np.nan
    rows = []
    for _, r in te.iterrows():
        rows.append({"actual_home_xg": r["home_xg"], "actual_away_xg": r["away_xg"],
                     "hg": int(r["home_goals"]), "ag": int(r["away_goals"]),
                     "xg_h": lam(pred_xg, r["match_id"], 1), "xg_a": lam(pred_xg, r["match_id"], 0),
                     "gl_h": lam(pred_gl, r["match_id"], 1), "gl_a": lam(pred_gl, r["match_id"], 0)})
    ev = pd.DataFrame(rows).dropna()
    ev["outcome"] = np.where(ev.hg > ev.ag, "home", np.where(ev.hg < ev.ag, "away", "draw"))
    print(f"test matches: {len(ev)}  ({TEST_FROM}+)\n")

    # (a) predictability of match xG
    def corr_mae(pcol, acol):
        return np.corrcoef(ev[pcol], ev[acol])[0,1], (ev[pcol]-ev[acol]).abs().mean()
    ch = corr_mae("xg_h", "actual_home_xg")
    print("=== (a) how well does pre-match xG predict the ACTUAL match xG? ===")
    print(f"  corr(pred_home_xg, actual_home_xg) = {ch[0]:.3f}   MAE = {ch[1]:.3f}")
    print(f"  (for context: actual home xG std = {ev['actual_home_xg'].std():.3f}; a match's xG is itself very noisy)")

    # (b) 1X2 from each lambda source
    def rps_acc(hcol, acol):
        rps=[]; correct=0
        for _, r in ev.iterrows():
            p = outcome_probabilities(float(r[hcol]), float(r[acol]), max_goals=10, dixon_coles_rho=-0.07)
            ph, pdw = p["p_home_win"], p["p_draw"]
            oh = r.outcome=="home"; od = r.outcome=="draw"
            rps.append(0.5*((ph-oh)**2+(ph+pdw-oh-od)**2))
            pick = ["home","draw","away"][int(np.argmax([p["p_home_win"],p["p_draw"],p["p_away_win"]]))]
            correct += pick==r.outcome
        return float(np.mean(rps)), correct/len(ev)
    xr = rps_acc("xg_h","xg_a"); gr = rps_acc("gl_h","gl_a")
    print("\n=== (b) 1X2 from a PURE pre-match predictor (same features form) ===")
    print(f"  xG-features   -> RPS {xr[0]:.4f}  acc {xr[1]:.3f}")
    print(f"  goals-features-> RPS {gr[0]:.4f}  acc {gr[1]:.3f}")
    print(f"  delta (xG - goals): RPS {xr[0]-gr[0]:+.4f}")

    # === (c) blend the pure xG-predictor with the strong AttackDefense (MLE) core ===
    from mundialytics.statistical_core.attack_defense_model import AttackDefenseModel
    tr_m = df[df["date"] < TEST_FROM]
    ad_g = AttackDefenseModel(time_decay_half_life=None).fit(tr_m)
    ad_x = AttackDefenseModel(time_decay_half_life=None, target="xg").fit(tr_m)

    evk = te.copy()
    evk["outcome"] = np.where(evk.home_goals > evk.away_goals, "home",
                              np.where(evk.home_goals < evk.away_goals, "away", "draw"))
    for side, home in [("h", 1), ("a", 0)]:
        evk[f"xr_{side}"] = evk["match_id"].map(lambda m, hh=home: _safe(pred_xg, m, hh))
        evk[f"gr_{side}"] = evk["match_id"].map(lambda m, hh=home: _safe(pred_gl, m, hh))
    adg = evk.apply(lambda r: ad_g.expected_goals(r.home_team, r.away_team, int(r.get("neutral", 0)), r.competition)[:2], axis=1)
    adx = evk.apply(lambda r: ad_x.expected_goals(r.home_team, r.away_team, int(r.get("neutral", 0)), r.competition)[:2], axis=1)
    evk["adg_h"], evk["adg_a"] = [t[0] for t in adg], [t[1] for t in adg]
    evk["adx_h"], evk["adx_a"] = [t[0] for t in adx], [t[1] for t in adx]
    evk = evk.dropna(subset=["xr_h", "xr_a", "adg_h", "adg_a"])

    def blend_rps(cw):
        rps = []
        for _, r in evk.iterrows():
            lh = sum(w * r[f"{c}_h"] for c, w in cw); la = sum(w * r[f"{c}_a"] for c, w in cw)
            p = outcome_probabilities(float(np.clip(lh, 0.05, 6)), float(np.clip(la, 0.05, 6)), max_goals=10, dixon_coles_rho=-0.07)
            oh = r.outcome == "home"; od = r.outcome == "draw"
            rps.append(0.5 * ((p["p_home_win"] - oh) ** 2 + (p["p_home_win"] + p["p_draw"] - oh - od) ** 2))
        return float(np.mean(rps))

    print("\n=== (c) blends with the AttackDefense core (same test set) ===")
    for name, cw in {
        "deployed proxy: 0.3*goalsRate + 0.7*AD_goals": [("gr", 0.3), ("adg", 0.7)],
        "0.3*xgRate + 0.7*AD_goals                   ": [("xr", 0.3), ("adg", 0.7)],
        "0.3*xgRate + 0.7*AD_xg                      ": [("xr", 0.3), ("adx", 0.7)],
        "0.5*xgRate + 0.5*AD_xg                      ": [("xr", 0.5), ("adx", 0.5)],
        "0.5*AD_xg  + 0.5*AD_goals                   ": [("adx", 0.5), ("adg", 0.5)],
        "pure AD_goals                               ": [("adg", 1.0)],
        "pure AD_xg                                  ": [("adx", 1.0)],
        "pure xgRate                                 ": [("xr", 1.0)],
    }.items():
        print(f"  {name}: RPS {blend_rps(cw):.4f}")
    print("\n  (deployed blended model 8-fold ~0.2057; oracle w/ actual xG ~0.164)")


def _safe(pred, mid, home):
    try:
        return float(pred.loc[(mid, home)])
    except KeyError:
        return np.nan


if __name__ == "__main__":
    main()
