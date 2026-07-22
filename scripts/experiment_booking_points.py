from __future__ import annotations

"""Booking points market (UK standard: 10 pts/yellow, 25 pts/red) at lines
30.5 / 40.5 / 50.5 — buildable now: HR/AR are 100% present in every raw
football-data file across all 5 leagues (checked 2026-07-22).

Model: total yellows via the deployed per-side recipe (rollings + ASYM ->
PoissonRegressor, NB dispersion) + total reds via (a) league-mean rate or
(b) + team rolling red tendency (A/B'd — reds may be pure noise at team
level). Points distribution: grid convolution of Y ~ NB and R ~ Poisson.
Baseline: league means through the same distribution machinery.
"""

import glob
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
W = (5, 10, 19)
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
LINES = [30.5, 40.5, 50.5]
DIV_LEAGUE = {"E0": "EPL", "SP1": "LaLiga", "D1": "Bundesliga", "I1": "SerieA", "F1": "Ligue1"}


def load_raw() -> pd.DataFrame:
    rows = []
    for p in glob.glob(str(ROOT / "data/raw/football_data/**/*.csv"), recursive=True):
        m = re.search(r"(\d{2})(\d{2})_(E0|SP1|D1|I1|F1)\.csv$", p)
        if not m:
            continue
        y1 = int(m.group(1))
        season = f"20{m.group(1)}-20{m.group(2)}"
        if y1 < 14:
            continue
        try:
            df = pd.read_csv(p, encoding="latin-1", on_bad_lines="skip",
                             usecols=lambda c: c in {"Date", "HomeTeam", "AwayTeam",
                                                     "HY", "AY", "HR", "AR", "FTHG", "FTAG"})
        except Exception:
            continue
        df["season"], df["comp"] = season, DIV_LEAGUE[m.group(3)]
        rows.append(df)
    r = pd.concat(rows, ignore_index=True)
    r["date"] = pd.to_datetime(r["Date"], dayfirst=True, errors="coerce", format="mixed")
    r = r.dropna(subset=["date", "HomeTeam"]).drop_duplicates(subset=["date", "HomeTeam", "AwayTeam"])
    for c in ["HY", "AY", "HR", "AR", "FTHG", "FTAG"]:
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r.dropna(subset=["HY", "AY", "HR", "AR", "FTHG", "FTAG"])
    r["match_id"] = np.arange(len(r))
    return r


def p_pts_over(lam_y: np.ndarray, disp_y: float, lam_r: np.ndarray, line: float) -> np.ndarray:
    """P(10Y + 25R > line), Y ~ NB(mean lam_y, var disp*lam_y), R ~ Poisson(lam_r)."""
    lam_y = np.clip(np.asarray(lam_y, float), 0.2, 25)
    lam_r = np.clip(np.asarray(lam_r, float), 0.01, 3)
    ry = lam_y / (disp_y - 1.0)
    out = np.zeros(len(lam_y))
    for r_cnt in range(0, 7):
        pr = poisson.pmf(r_cnt, lam_r)
        # need 10*Y > line - 25*r  ->  Y > (line - 25*r)/10
        thr = np.floor((line - 25 * r_cnt) / 10.0)
        py_over = np.where(thr < 0, 1.0, 1.0 - nbinom.cdf(thr, ry, 1.0 / disp_y))
        out += pr * py_over
    return np.clip(out, 0, 1)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    t0 = time.time()
    m = load_raw()
    print(f"raw matches 2014+: {len(m)} | mean Y {(m.HY+m.AY).mean():.2f} R {(m.HR+m.AR).mean():.2f} "
          f"pts {(10*(m.HY+m.AY)+25*(m.HR+m.AR)).mean():.1f}")

    rows = []
    for r in m.itertuples(index=False):
        rows.append(dict(match_id=r.match_id, date=r.date, season=r.season, comp=r.comp,
                         team=str(r.HomeTeam).lower(), opp=str(r.AwayTeam).lower(), is_home=1,
                         ev_for=r.HY, ev_against=r.AY, red_for=r.HR, gf=r.FTHG, ga=r.FTAG))
        rows.append(dict(match_id=r.match_id, date=r.date, season=r.season, comp=r.comp,
                         team=str(r.AwayTeam).lower(), opp=str(r.HomeTeam).lower(), is_home=0,
                         ev_for=r.AY, ev_against=r.HY, red_for=r.AR, gf=r.FTAG, ga=r.FTHG))
    lr = pd.DataFrame(rows).sort_values(["team", "date", "match_id"])
    for col in ["ev_for", "ev_against", "gf", "ga", "red_for"]:
        for w in W:
            lr[f"{col}_r{w}"] = (lr.groupby("team", group_keys=False)[col]
                                 .apply(lambda s: s.shift(1).rolling(w, min_periods=3).mean()))
        lr[f"{col}_ewm"] = (lr.groupby("team", group_keys=False)[col]
                            .apply(lambda s: s.shift(1).ewm(halflife=5, min_periods=3).mean()))
    opp_src = [f"ev_against_r{w}" for w in W] + ["ev_against_ewm", "gf_ewm", "ga_ewm"]
    opp = lr[["match_id", "team"] + opp_src].rename(
        columns={"team": "opp", **{c: f"opp_{c}" for c in opp_src}})
    lr = lr.merge(opp, on=["match_id", "opp"], how="left")
    lr["delta"] = (lr["gf_ewm"] + lr["opp_ga_ewm"]) / 2 - (lr["opp_gf_ewm"] + lr["ga_ewm"]) / 2
    lr["abs_delta"] = lr["delta"].abs()
    feats = ([f"ev_for_r{w}" for w in W] + ["ev_for_ewm"]
             + [f"opp_ev_against_r{w}" for w in W] + ["opp_ev_against_ewm"]
             + ["is_home", "delta", "abs_delta"])

    for red_mode in ["league", "team"]:
        res = {ln: {"m": [], "b": []} for ln in LINES}
        for s in TEST_SEASONS:
            te_m = m[m.season == s]
            if len(te_m) == 0:
                continue
            s_start = pd.to_datetime(te_m.date.min())
            tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
            if len(tr) < 2000:
                continue
            reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
            te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
            te["pred_y"] = np.clip(reg.predict(te[feats]), 0.1, 15)
            tr_m = m[pd.to_datetime(m.date) < s_start]
            # red lambda per side
            lg_red = tr_m.assign(rt=(tr_m.HR + tr_m.AR) / 2).groupby("comp")["rt"].mean()
            if red_mode == "team":
                te["pred_r"] = (0.5 * te["red_for_ewm"].fillna(0)
                                + 0.5 * te["comp"].map(lg_red).fillna(float(lg_red.mean())))
            else:
                te["pred_r"] = te["comp"].map(lg_red).fillna(float(lg_red.mean()))
            pvy = te.pivot_table(index="match_id", columns="is_home", values="pred_y").dropna()
            pvr = te.pivot_table(index="match_id", columns="is_home", values="pred_r").dropna()
            idx = pvy.index.intersection(pvr.index)
            lam_y, lam_r = (pvy.loc[idx, 1] + pvy.loc[idx, 0]), (pvr.loc[idx, 1] + pvr.loc[idx, 0])
            tt_y = (tr_m.HY + tr_m.AY).astype(float)
            disp_y = float(np.clip(tt_y.var() / max(tt_y.mean(), 1e-9), 1.05, 3.0))
            tei = te_m.set_index("match_id").loc[idx]
            base_y = tei["comp"].map(tr_m.assign(t=tt_y).groupby("comp")["t"].mean()).to_numpy()
            base_r = tei["comp"].map(lg_red * 2).to_numpy()
            act_pts = (10 * (tei.HY + tei.AY) + 25 * (tei.HR + tei.AR)).astype(float)
            for ln in LINES:
                y = (act_pts > ln).astype(float).to_numpy()
                res[ln]["m"].append((bll(y, p_pts_over(lam_y.to_numpy(), disp_y, lam_r.to_numpy(), ln)), len(y), s))
                res[ln]["b"].append((bll(y, p_pts_over(base_y, disp_y, base_r, ln)), len(y), s))
        pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
        print(f"\n===== BOOKING PTS (reds={red_mode}, {time.time()-t0:.0f}s) =====")
        for ln in LINES:
            folds = " ".join(f"{s_[2][-2:]}{'+' if s_[0] < b_[0] else '-'}"
                             for s_, b_ in zip(res[ln]["m"], res[ln]["b"]))
            print(f"  O {ln}: LL {pool(res[ln]['m']):.4f} vs lg-base {pool(res[ln]['b']):.4f} "
                  f"delta {pool(res[ln]['m'])-pool(res[ln]['b']):+.4f}  [{folds}]", flush=True)


if __name__ == "__main__":
    main()
