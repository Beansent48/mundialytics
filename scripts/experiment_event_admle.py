from __future__ import annotations

"""Structural upgrade candidate for team props: Attack/Defense MLE per EVENT.

The goals engine's core is a Maher/Dixon-Coles MLE of per-team attack/defense
strengths — rolling features are only its FORM layer. Team props currently have
ONLY the form layer. Mirror test: fit AttackDefenseModel on each event count
(corners/cards/fouls/shots/sot as the 'goals' columns, rho=0 — the DC tie
correction is goal-specific; caps raised) and BLEND with the deployed rate
model: lam = w*ADM + (1-w)*rate, w grid {0, .3, .5, .7, 1}. w=0 is today's
deployed config; anything that beats it 5/5 folds gets deployed.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.statistical_core.attack_defense_model import AttackDefenseModel  # noqa: E402

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
PREDS = ROOT / "data/processed/enriched/understat_xg/walkforward_preds.csv"
PREDS_H = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_hist.csv"
W = (5, 10, 19)
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
MARKETS = {
    "corners": ("home_corners", "away_corners", [8.5, 9.5, 10.5], 20.0),
    "yellows": ("home_yellow_cards", "away_yellow_cards", [3.5, 4.5, 5.5], 12.0),
    "fouls":   ("home_fouls", "away_fouls", [21.5, 23.5], 35.0),
    "shots":   ("home_shots", "away_shots", [22.5, 24.5], 40.0),
    "sot":     ("home_sot", "away_sot", [7.5, 8.5], 20.0),
}
BLEND_W = [0.0, 0.3, 0.5, 0.7, 1.0]


def prob_over(lam, line, disp):
    k = int(np.floor(line))
    lam = np.clip(lam, 0.2, 60.0)
    r = lam / (disp - 1.0)
    return 1.0 - nbinom.cdf(k, r, 1.0 / disp)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def build_lr(m, hc, ac):
    rows = []
    for r in m.itertuples(index=False):
        rows.append(dict(match_id=r.match_id, date=r.date, season=r.season, team=r.home_team,
                         opp=r.away_team, is_home=1, ev_for=getattr(r, hc), ev_against=getattr(r, ac),
                         gf=r.home_goals, ga=r.away_goals, lam_t=r.lh, lam_o=r.la))
        rows.append(dict(match_id=r.match_id, date=r.date, season=r.season, team=r.away_team,
                         opp=r.home_team, is_home=0, ev_for=getattr(r, ac), ev_against=getattr(r, hc),
                         gf=r.away_goals, ga=r.home_goals, lam_t=r.la, lam_o=r.lh))
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
    lr["delta"] = (lr["gf_ewm"] + lr["opp_ga_ewm"]) / 2 - (lr["opp_gf_ewm"] + lr["ga_ewm"]) / 2
    lr["abs_delta"] = lr["delta"].abs()
    lr["delta_lam"] = lr["lam_t"] - lr["lam_o"]
    lr["abs_delta_lam"] = lr["delta_lam"].abs()
    return lr


def main() -> None:
    lam = pd.concat([pd.read_csv(PREDS)[["match_id", "lh", "la"]],
                     pd.read_csv(PREDS_H)[["match_id", "lh", "la"]]],
                    ignore_index=True).drop_duplicates("match_id")
    full = pd.read_csv(FOUND, low_memory=False)
    full["date"] = pd.to_datetime(full["date"], errors="coerce")
    full = full[full["season"] >= "2014-2015"]
    df = full.merge(lam, on="match_id", how="inner")

    feats = ([f"ev_for_r{w}" for w in W] + ["ev_for_ewm"]
             + [f"opp_ev_against_r{w}" for w in W] + ["opp_ev_against_ewm"]
             + ["is_home", "delta", "abs_delta", "delta_lam", "abs_delta_lam"])

    for market, (hc, ac, lines, cap) in MARKETS.items():
        t0 = time.time()
        m = df.dropna(subset=[hc, ac, "home_goals", "away_goals", "date", "lh", "la"]).copy()
        for c in [hc, ac, "home_goals", "away_goals"]:
            m[c] = pd.to_numeric(m[c], errors="coerce")
        m = m.dropna(subset=[hc, ac, "home_goals", "away_goals"])
        # ADM trains on the FULL event panel (no lambda need)
        mf = full.dropna(subset=[hc, ac, "date"]).copy()
        mf[hc] = pd.to_numeric(mf[hc], errors="coerce")
        mf[ac] = pd.to_numeric(mf[ac], errors="coerce")
        mf = mf.dropna(subset=[hc, ac])
        lr = build_lr(m, hc, ac)

        res = {wb: {ln: {"m": [], "b": []} for ln in lines} for wb in BLEND_W}
        for s in TEST_SEASONS:
            te_m = m[m.season == s]
            if len(te_m) == 0:
                continue
            s_start = te_m.date.min()
            # rate model (deployed config)
            tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
            reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
            te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
            te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 40)
            pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
            # ADM on events
            adm_tr = mf[mf.date < s_start].rename(columns={hc: "home_goals2", ac: "away_goals2"})
            adm_tr = adm_tr.drop(columns=["home_goals", "away_goals"]).rename(
                columns={"home_goals2": "home_goals", "away_goals2": "away_goals"})
            adm = AttackDefenseModel(dixon_coles_rho=0.0, time_decay_half_life=365.0,
                                     goal_cap=cap, max_goals=5)
            adm.fit(adm_tr)
            tei_all = te_m.set_index("match_id").loc[pv.index]
            adm_lh, adm_la = [], []
            for r in tei_all.itertuples(index=False):
                a, b, _ = adm.expected_goals(r.home_team, r.away_team, 0, r.competition)
                adm_lh.append(a)
                adm_la.append(b)
            adm_tot = np.array(adm_lh) + np.array(adm_la)
            rate_tot = (pv[1] + pv[0]).to_numpy()
            tr_tot = m[m.date < s_start]
            tt = (tr_tot[hc] + tr_tot[ac]).astype(float)
            disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 1.11, 3.0))
            lg = tr_tot.assign(t=tt).groupby("competition")["t"].mean()
            base_v = tei_all["competition"].map(lg).fillna(float(tt.mean())).to_numpy()
            act = tei_all[[hc, ac]].sum(axis=1).astype(float)
            for wb in BLEND_W:
                tot = wb * adm_tot + (1 - wb) * rate_tot
                for ln in lines:
                    y = (act > ln).astype(float).to_numpy()
                    res[wb][ln]["m"].append((bll(y, prob_over(tot, ln, disp)), len(y), s))
                    res[wb][ln]["b"].append((bll(y, prob_over(base_v, ln, disp)), len(y), s))
        pool = lambda a: sum(x * n for x, n, _ in a) / max(sum(n for _, n, _ in a), 1)
        print(f"\n===== {market.upper()} ({time.time()-t0:.0f}s) =====", flush=True)
        for wb in BLEND_W:
            tag = "rate(DEPLOYED)" if wb == 0 else ("ADM-pure" if wb == 1 else f"blend {wb}")
            deltas = " | ".join(f"O{ln} {pool(res[wb][ln]['m'])-pool(res[wb][ln]['b']):+.4f}" for ln in lines)
            f0 = " ".join(f"{s_[2][-2:]}{'+' if s_[0] < b_[0] else '-'}"
                          for s_, b_ in zip(res[wb][lines[0]]["m"], res[wb][lines[0]]["b"]))
            print(f"  {tag:15s}: {deltas}   [{f0} @O{lines[0]}]", flush=True)


if __name__ == "__main__":
    main()
