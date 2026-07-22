from __future__ import annotations

"""Team-props backtest: corners, yellow cards, fouls, shots, SOT — match totals
at standard bookmaker lines.

Candidate = the proven rate-model recipe (per-side rolling event-for/against,
windows 5/10/19 + EWMA hl5, opponent rates, is_home -> PoissonRegressor per side;
total lambda = home+away). Dispersion is measured on train residuals: if totals
are over-dispersed (var/mean > 1.1), Negative Binomial replaces Poisson for the
O/U probabilities. Baseline = league base-rate lambda under the same distribution.
Walk-forward folds 2021/22-2025/26. Foundation event coverage ~88% since 2000;
we train on 2014+ to match the modern era.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor

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
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
SEASONS_FROM = "2014-2015"


def prob_over(total_lam: np.ndarray, line: float, disp: float) -> np.ndarray:
    """P(total > line). disp = var/mean from train; NB if > 1.1 else Poisson."""
    k = int(np.floor(line))
    lam = np.clip(total_lam, 0.2, 40.0)
    if disp > 1.1:
        # NB with mean lam, var = disp*lam  ->  r = lam/(disp-1), p = 1/disp
        r = lam / (disp - 1.0)
        return 1.0 - nbinom.cdf(k, r, 1.0 / disp)
    return 1.0 - poisson.cdf(k, lam)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    df = pd.read_csv(FOUND, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    seasons = sorted(df["season"].dropna().unique())
    df = df[df["season"] >= SEASONS_FROM].copy()

    for market, (hc, ac, lines) in MARKETS.items():
        t0 = time.time()
        m = df.dropna(subset=[hc, ac, "date"]).copy()
        m[hc] = pd.to_numeric(m[hc], errors="coerce")
        m[ac] = pd.to_numeric(m[ac], errors="coerce")
        m = m.dropna(subset=[hc, ac])
        # long rows
        rows = []
        for r in m.itertuples(index=False):
            rows.append(dict(match_id=r.match_id, date=r.date, season=r.season, team=r.home_team,
                             opp=r.away_team, is_home=1, ev_for=getattr(r, hc), ev_against=getattr(r, ac),
                             league=r.competition))
            rows.append(dict(match_id=r.match_id, date=r.date, season=r.season, team=r.away_team,
                             opp=r.home_team, is_home=0, ev_for=getattr(r, ac), ev_against=getattr(r, hc),
                             league=r.competition))
        lr = pd.DataFrame(rows).sort_values(["team", "date", "match_id"])
        for col in ["ev_for", "ev_against"]:
            for w in W:
                lr[f"{col}_r{w}"] = (lr.groupby("team", group_keys=False)[col]
                                     .apply(lambda s: s.shift(1).rolling(w, min_periods=3).mean()))
            lr[f"{col}_ewm"] = (lr.groupby("team", group_keys=False)[col]
                                .apply(lambda s: s.shift(1).ewm(halflife=5, min_periods=3).mean()))
        opp_src = [f"ev_against_r{w}" for w in W] + ["ev_against_ewm"]
        opp = lr[["match_id", "team"] + opp_src].rename(columns={"team": "opp", **{c: f"opp_{c}" for c in opp_src}})
        lr = lr.merge(opp, on=["match_id", "opp"], how="left")
        feats = ([f"ev_for_r{w}" for w in W] + ["ev_for_ewm"]
                 + [f"opp_ev_against_r{w}" for w in W] + ["opp_ev_against_ewm"] + ["is_home"])

        fold_res = {ln: {"model": [], "base": []} for ln in lines}
        disp_used = None
        for s in TEST_SEASONS:
            te_m = m[m.season == s]
            if len(te_m) == 0:
                continue
            s_start = te_m.date.min()
            tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
            if len(tr) < 2000:
                continue
            reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
            te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats)
            te = te.copy(); te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 25)
            pv = te.pivot_table(index="match_id", columns="is_home", values="pred")
            pv = pv.dropna()
            tot_pred = (pv[1] + pv[0])
            # dispersion from TRAIN totals residuals (var of total / mean of total, crude but honest)
            tr_tot = m[m.date < s_start]
            tt = (tr_tot[hc] + tr_tot[ac]).astype(float)
            disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
            disp_used = disp
            # HONEST baseline: league-specific mean total (train) — beats "global mean"
            lg_lam = tr_tot.assign(tot=tt).groupby("competition")["tot"].mean()
            te_idx = te_m.set_index("match_id").loc[tot_pred.index]
            base_lam_v = te_idx["competition"].map(lg_lam).fillna(float(tt.mean())).to_numpy()
            act = te_idx[[hc, ac]].sum(axis=1).astype(float)
            for ln in lines:
                y = (act > ln).astype(float).to_numpy()
                pmod = prob_over(tot_pred.to_numpy(), ln, disp)
                pbase = prob_over(base_lam_v, ln, disp)
                fold_res[ln]["model"].append((bll(y, pmod), len(y), s))
                fold_res[ln]["base"].append((bll(y, pbase), len(y), s))
                fold_res[ln].setdefault("yp", []).append((y, pmod))
        pool = lambda a: sum(x * n for x, n, _ in a) / max(sum(n for _, n, _ in a), 1)
        print(f"\n===== {market.upper()}  (disp={disp_used:.2f} {'NB' if disp_used and disp_used>1.1 else 'Poisson'}, {time.time()-t0:.0f}s) =====")
        for ln in lines:
            mm, bb = pool(fold_res[ln]["model"]), pool(fold_res[ln]["base"])
            n = sum(nn for _, nn, _ in fold_res[ln]["model"])
            folds = " ".join(f"{s[-2:]}{'+' if a[0] < b[0] else '-'}"
                             for a, b, s in [(x, yy, x[2]) for x, yy in zip(fold_res[ln]["model"], fold_res[ln]["base"])])
            yy_all = np.concatenate([y for y, _ in fold_res[ln]["yp"]])
            pp_all = np.concatenate([p for _, p in fold_res[ln]["yp"]])
            q = pd.qcut(pp_all, 8, duplicates="drop")
            cal = pd.DataFrame({"p": pp_all, "y": yy_all}).groupby(q, observed=True).agg(
                n=("y", "size"), pred=("p", "mean"), emp=("y", "mean"))
            ece = float((cal.n / cal.n.sum() * (cal.pred - cal.emp).abs()).sum())
            spread = f"{pp_all.min():.2f}-{pp_all.max():.2f}"
            print(f"  O/U {ln:>4}: model LL {mm:.4f} | lg-base {bb:.4f} | delta {mm-bb:+.4f} | "
                  f"ECE {ece:.4f} | rango p {spread}  (n={n})  folds[{folds}]", flush=True)


if __name__ == "__main__":
    main()
