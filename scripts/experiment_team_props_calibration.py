from __future__ import annotations

"""Walk-forward Platt recalibration for team props.

Team-prop ECE is 0.02-0.05 (vs <=0.013 for player props). Fix candidate:
per-market logit-scale Platt (p' = sigmoid(a*logit(p) + b)) fitted ONLY on
out-of-sample predictions of PAST seasons (strict walk-forward: to calibrate
season s we use walk-forward predictions of seasons < s, each produced by a
model trained before that season). Evaluated on the usual folds 21/22-25/26.

Report per market/line: raw LL/ECE vs calibrated LL/ECE, fold consistency.
Deploy bar: calibrated must not LOSE log-loss and should cut ECE.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
MARKETS = {
    "corners": ("home_corners", "away_corners", [8.5, 9.5, 10.5]),
    "yellows": ("home_yellow_cards", "away_yellow_cards", [3.5, 4.5, 5.5]),
    "fouls":   ("home_fouls", "away_fouls", [21.5, 23.5]),
    "shots":   ("home_shots", "away_shots", [22.5, 24.5]),
    "sot":     ("home_sot", "away_sot", [7.5, 8.5]),
}
W = (5, 10, 19)
# walk-forward predictions produced for ALL these seasons; the last five are the
# EVALUATION folds, earlier ones only feed the calibrator
PRED_SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020", "2020-2021",
                "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
TEST_SEASONS = PRED_SEASONS[-5:]


def prob_over(lam, line, disp):
    k = int(np.floor(line))
    lam = np.clip(lam, 0.2, 40.0)
    if disp > 1.1:
        r = lam / (disp - 1.0)
        return 1.0 - nbinom.cdf(k, r, 1.0 / disp)
    return 1.0 - poisson.cdf(k, lam)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ece(y, p):
    q = pd.qcut(p, 8, duplicates="drop")
    cal = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).agg(
        n=("y", "size"), pred=("p", "mean"), emp=("y", "mean"))
    return float((cal.n / cal.n.sum() * (cal.pred - cal.emp).abs()).sum())


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    df = pd.read_csv(FOUND, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["season"] >= "2014-2015"].copy()

    for market, (hc, ac, lines) in MARKETS.items():
        t0 = time.time()
        m = df.dropna(subset=[hc, ac, "date"]).copy()
        m[hc] = pd.to_numeric(m[hc], errors="coerce")
        m[ac] = pd.to_numeric(m[ac], errors="coerce")
        m = m.dropna(subset=[hc, ac])
        m = m.dropna(subset=["home_goals", "away_goals"])
        rows = []
        for r in m.itertuples(index=False):
            rows.append(dict(match_id=r.match_id, date=r.date, team=r.home_team, opp=r.away_team,
                             is_home=1, ev_for=getattr(r, hc), ev_against=getattr(r, ac),
                             gf=r.home_goals, ga=r.away_goals))
            rows.append(dict(match_id=r.match_id, date=r.date, team=r.away_team, opp=r.home_team,
                             is_home=0, ev_for=getattr(r, ac), ev_against=getattr(r, hc),
                             gf=r.away_goals, ga=r.home_goals))
        lr = pd.DataFrame(rows).sort_values(["team", "date", "match_id"])
        for col in ["ev_for", "ev_against", "gf", "ga"]:
            for w in W:
                lr[f"{col}_r{w}"] = (lr.groupby("team", group_keys=False)[col]
                                     .apply(lambda s: s.shift(1).rolling(w, min_periods=3).mean()))
            lr[f"{col}_ewm"] = (lr.groupby("team", group_keys=False)[col]
                                .apply(lambda s: s.shift(1).ewm(halflife=5, min_periods=3).mean()))
        opp_src = [f"ev_against_r{w}" for w in W] + ["ev_against_ewm", "gf_ewm", "ga_ewm"]
        opp = lr[["match_id", "team"] + opp_src].rename(
            columns={"team": "opp", **{c: f"opp_{c}" for c in opp_src}})
        lr = lr.merge(opp, on=["match_id", "opp"], how="left")
        # ASYM (validated all 5 markets): expected supremacy of this side
        lr["delta"] = ((lr["gf_ewm"] + lr["opp_ga_ewm"]) / 2 - (lr["opp_gf_ewm"] + lr["ga_ewm"]) / 2)
        lr["abs_delta"] = lr["delta"].abs()
        feats = ([f"ev_for_r{w}" for w in W] + ["ev_for_ewm"]
                 + [f"opp_ev_against_r{w}" for w in W] + ["opp_ev_against_ewm"]
                 + ["is_home", "delta", "abs_delta"])

        # 1) walk-forward predictions for ALL PRED_SEASONS
        preds = {}   # season -> DataFrame(match_id, tot_pred, act, disp)
        for s in PRED_SEASONS:
            te_m = m[m.season == s]
            if len(te_m) == 0:
                continue
            s_start = te_m.date.min()
            tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
            if len(tr) < 2000:
                continue
            reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
            te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
            te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 25)
            pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
            tot = pv[1] + pv[0]
            tr_tot = m[m.date < s_start]
            tt = (tr_tot[hc] + tr_tot[ac]).astype(float)
            disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
            act = te_m.set_index("match_id").loc[tot.index, [hc, ac]].sum(axis=1).astype(float)
            preds[s] = pd.DataFrame({"tot": tot, "act": act, "disp": disp})

        # 2) per line: Platt fit on past seasons' preds, applied to test season
        print(f"\n===== {market.upper()} ({time.time()-t0:.0f}s) =====")
        for ln in lines:
            raw_res, cal_res = [], []
            folds = []
            for s in TEST_SEASONS:
                if s not in preds:
                    continue
                past = [preds[ps] for ps in PRED_SEASONS if ps < s and ps in preds]
                if not past:
                    continue
                pa = pd.concat(past)
                x_cal = logit(prob_over(pa["tot"].to_numpy(), ln, float(pa["disp"].iloc[0])))
                y_cal = (pa["act"] > ln).astype(int).to_numpy()
                platt = LogisticRegression(C=1e6, max_iter=1000).fit(x_cal.reshape(-1, 1), y_cal)
                cur = preds[s]
                p_raw = prob_over(cur["tot"].to_numpy(), ln, float(cur["disp"].iloc[0]))
                p_cal = platt.predict_proba(logit(p_raw).reshape(-1, 1))[:, 1]
                y = (cur["act"] > ln).astype(float).to_numpy()
                raw_res.append((y, p_raw))
                cal_res.append((y, p_cal))
                folds.append(f"{s[-2:]}{'+' if bll(y, p_cal) < bll(y, p_raw) else '-'}")
            y_all = np.concatenate([y for y, _ in raw_res])
            praw = np.concatenate([p for _, p in raw_res])
            pcal = np.concatenate([p for _, p in cal_res])
            print(f"  O/U {ln:>4}: LL raw {bll(y_all, praw):.4f} -> cal {bll(y_all, pcal):.4f} "
                  f"(d {bll(y_all, pcal)-bll(y_all, praw):+.4f}) | ECE raw {ece(y_all, praw):.4f} -> "
                  f"cal {ece(y_all, pcal):.4f}  [{' '.join(folds)}]", flush=True)


if __name__ == "__main__":
    main()
