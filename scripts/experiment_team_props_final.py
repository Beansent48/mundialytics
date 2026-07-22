from __future__ import annotations

"""JOINT validation of the round-5 per-market winners before deployment
(the ASYM+Platt lesson: stacked within-market changes must be tested together).

Per-market FINAL config vs DEPLOYED (current module):
  corners: ev-EWM halflife 12 (long memory) + ADM blend w=0.3
  yellows: +STAKES features + ADM blend w=0.3
  fouls:   +STAKES features (ADM was a wash there)
  shots:   ADM blend per pending verdict (set below)
  sot:     ADM blend per pending verdict (set below)
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
SIDE_LINES = {"corners": [3.5, 4.5, 5.5], "yellows": [1.5, 2.5], "shots": [9.5, 11.5, 13.5]}
FINAL = {
    "corners": dict(hl=12, stakes=False, adm_w=0.3),
    "yellows": dict(hl=5, stakes=True, adm_w=0.3),
    "fouls":   dict(hl=5, stakes=True, adm_w=0.0),
    "shots":   dict(hl=5, stakes=False, adm_w=0.0),   # ADM dropped: side loss offset tiny total gain
    "sot":     dict(hl=5, stakes=False, adm_w=0.0),
}
if "--yellows-only" in sys.argv:
    MARKETS = {"yellows": MARKETS["yellows"]}


def prob_over(lam, line, disp):
    from scipy.stats import poisson as _poi
    k = int(np.floor(line))
    lam = np.clip(lam, 0.2, 60.0)
    if disp <= 1.05:               # side yellows disperse ~0.94 -> Poisson
        return 1.0 - _poi.cdf(k, lam)
    r = lam / (disp - 1.0)
    return 1.0 - nbinom.cdf(k, r, 1.0 / disp)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def add_positions(full: pd.DataFrame) -> pd.DataFrame:
    full = full.sort_values("date").copy()
    ph = []
    for (_, _), g in full.groupby(["competition", "season"], sort=False):
        pts: dict = {}
        played: dict = {}
        for r in g.itertuples(index=False):
            def rank(t):
                p = pts.get(t, 0)
                return 1 + sum(1 for v in pts.values() if v > p)
            ph.append((r.match_id, rank(r.home_team), rank(r.away_team), played.get(r.home_team, 0)))
            hgo, ago = r.home_goals, r.away_goals
            pts[r.home_team] = pts.get(r.home_team, 0) + (3 if hgo > ago else (1 if hgo == ago else 0))
            pts[r.away_team] = pts.get(r.away_team, 0) + (3 if ago > hgo else (1 if hgo == ago else 0))
            played[r.home_team] = played.get(r.home_team, 0) + 1
            played[r.away_team] = played.get(r.away_team, 0) + 1
    pos = pd.DataFrame(ph, columns=["match_id", "pos_home", "pos_away", "played_home"])
    return full.merge(pos, on="match_id", how="left")


def main() -> None:
    lam = pd.concat([pd.read_csv(PREDS)[["match_id", "lh", "la"]],
                     pd.read_csv(PREDS_H)[["match_id", "lh", "la"]]],
                    ignore_index=True).drop_duplicates("match_id")
    full = pd.read_csv(FOUND, low_memory=False)
    full["date"] = pd.to_datetime(full["date"], errors="coerce")
    full = full[full["season"] >= "2014-2015"].dropna(subset=["home_goals", "away_goals", "date"])
    full = add_positions(full)
    df = full.merge(lam, on="match_id", how="inner")

    for market, (hc, ac, lines, cap) in MARKETS.items():
        cfg = FINAL[market]
        t0 = time.time()
        m = df.dropna(subset=[hc, ac, "lh", "la"]).copy()
        m[hc] = pd.to_numeric(m[hc], errors="coerce")
        m[ac] = pd.to_numeric(m[ac], errors="coerce")
        m = m.dropna(subset=[hc, ac])
        mf = full.dropna(subset=[hc, ac]).copy()
        mf[hc] = pd.to_numeric(mf[hc], errors="coerce")
        mf[ac] = pd.to_numeric(mf[ac], errors="coerce")
        mf = mf.dropna(subset=[hc, ac])

        rows = []
        for r in m.itertuples(index=False):
            rows.append(dict(match_id=r.match_id, date=r.date, season=r.season, team=r.home_team,
                             opp=r.away_team, is_home=1, ev_for=getattr(r, hc), ev_against=getattr(r, ac),
                             gf=r.home_goals, ga=r.away_goals, lam_t=r.lh, lam_o=r.la,
                             pos_t=r.pos_home, pos_o=r.pos_away, played=r.played_home))
            rows.append(dict(match_id=r.match_id, date=r.date, season=r.season, team=r.away_team,
                             opp=r.home_team, is_home=0, ev_for=getattr(r, ac), ev_against=getattr(r, hc),
                             gf=r.away_goals, ga=r.home_goals, lam_t=r.la, lam_o=r.lh,
                             pos_t=r.pos_away, pos_o=r.pos_home, played=r.played_home))
        lr = pd.DataFrame(rows).sort_values(["team", "date", "match_id"])
        gb = lr.groupby("team", group_keys=False)
        for col in ["ev_for", "ev_against", "gf", "ga"]:
            for w in W:
                lr[f"{col}_r{w}"] = gb[col].apply(lambda s: s.shift(1).rolling(w, min_periods=3).mean())
            lr[f"{col}_ewm"] = gb[col].apply(lambda s: s.shift(1).ewm(halflife=5, min_periods=3).mean())
        for col in ["ev_for", "ev_against"]:
            lr[f"{col}_ewmF"] = gb[col].apply(
                lambda s: s.shift(1).ewm(halflife=cfg["hl"], min_periods=3).mean())
        lr["round_frac"] = (lr["played"] / 38.0).clip(0, 1)
        lr["pos_diff_abs"] = (lr["pos_t"] - lr["pos_o"]).abs()
        lr["releg_battle"] = (((lr["pos_t"] >= 15) | (lr["pos_o"] >= 15)) & (lr["round_frac"] > 0.6)).astype(float)
        opp_src = ([f"ev_against_r{w}" for w in W] + ["ev_against_ewm", "ev_against_ewmF", "gf_ewm", "ga_ewm"])
        opp = lr[["match_id", "team"] + opp_src].rename(
            columns={"team": "opp", **{c: f"opp_{c}" for c in opp_src}})
        lr = lr.merge(opp, on=["match_id", "opp"], how="left")
        lr["delta"] = (lr["gf_ewm"] + lr["opp_ga_ewm"]) / 2 - (lr["opp_gf_ewm"] + lr["ga_ewm"]) / 2
        lr["abs_delta"] = lr["delta"].abs()
        lr["delta_lam"] = lr["lam_t"] - lr["lam_o"]
        lr["abs_delta_lam"] = lr["delta_lam"].abs()

        dep_feats = ([f"ev_for_r{w}" for w in W] + ["ev_for_ewm"]
                     + [f"opp_ev_against_r{w}" for w in W] + ["opp_ev_against_ewm"]
                     + ["is_home", "delta", "abs_delta", "delta_lam", "abs_delta_lam"])
        fin_feats = ([f"ev_for_r{w}" for w in W] + ["ev_for_ewmF"]
                     + [f"opp_ev_against_r{w}" for w in W] + ["opp_ev_against_ewmF"]
                     + ["is_home", "delta", "abs_delta", "delta_lam", "abs_delta_lam"])
        if cfg["stakes"]:
            fin_feats += ["pos_diff_abs", "releg_battle", "round_frac"]

        s_lines = SIDE_LINES.get(market, [])
        res = {tag: {ln: {"m": [], "b": []} for ln in lines} for tag in ["DEPLOYED", "FINAL"]}
        res_s = {tag: {ln: {"m": [], "b": []} for ln in s_lines} for tag in ["DEPLOYED", "FINAL"]}
        for s in TEST_SEASONS:
            te_m = m[m.season == s]
            if len(te_m) == 0:
                continue
            s_start = te_m.date.min()
            if cfg["adm_w"] > 0:
                adm_tr = (mf[mf.date < s_start]
                          .rename(columns={hc: "hg2", ac: "ag2"})
                          .drop(columns=["home_goals", "away_goals"])
                          .rename(columns={"hg2": "home_goals", "ag2": "away_goals"}))
                adm = AttackDefenseModel(dixon_coles_rho=0.0, time_decay_half_life=365.0,
                                         goal_cap=cap, max_goals=5)
                adm.fit(adm_tr)
            for tag, feats in [("DEPLOYED", dep_feats), ("FINAL", fin_feats)]:
                tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
                reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
                te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
                te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 40)
                pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
                lh_v, la_v = pv[1].to_numpy(), pv[0].to_numpy()
                tei = te_m.set_index("match_id").loc[pv.index]
                if tag == "FINAL" and cfg["adm_w"] > 0:
                    a_l = np.array([adm.expected_goals(r.home_team, r.away_team, 0, r.competition)[:2]
                                    for r in tei.itertuples(index=False)])
                    wb = cfg["adm_w"]
                    lh_v = wb * a_l[:, 0] + (1 - wb) * lh_v
                    la_v = wb * a_l[:, 1] + (1 - wb) * la_v
                tot = lh_v + la_v
                tr_tot = m[m.date < s_start]
                tt = (tr_tot[hc] + tr_tot[ac]).astype(float)
                disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 1.11, 3.0))
                lg = tr_tot.assign(t=tt).groupby("competition")["t"].mean()
                base_v = tei["competition"].map(lg).fillna(float(tt.mean())).to_numpy()
                act = tei[[hc, ac]].sum(axis=1).astype(float)
                for ln in lines:
                    y = (act > ln).astype(float).to_numpy()
                    res[tag][ln]["m"].append((bll(y, prob_over(tot, ln, disp)), len(y), s))
                    res[tag][ln]["b"].append((bll(y, prob_over(base_v, ln, disp)), len(y), s))
                if s_lines:
                    trs_lr = lr[lr.date < s_start]
                    sv = trs_lr["ev_for"].dropna().astype(float)
                    disp_s = float(np.clip(sv.var() / max(sv.mean(), 1e-9), 0.9, 3.0))
                    side_mean_h = tr_tot[hc].mean()
                    side_mean_a = tr_tot[ac].mean()
                    lam_sides = np.concatenate([lh_v, la_v])
                    act_sides = np.concatenate([tei[hc].to_numpy(float), tei[ac].to_numpy(float)])
                    base_sides = np.concatenate([np.full(len(lh_v), side_mean_h),
                                                 np.full(len(la_v), side_mean_a)])
                    for ln in s_lines:
                        ys = (act_sides > ln).astype(float)
                        res_s[tag][ln]["m"].append((bll(ys, prob_over(lam_sides, ln, disp_s)), len(ys), s))
                        res_s[tag][ln]["b"].append((bll(ys, prob_over(base_sides, ln, disp_s)), len(ys), s))
        pool = lambda a: sum(x * n for x, n, _ in a) / max(sum(n for _, n, _ in a), 1)
        cfg_s = f"hl={cfg['hl']}, stakes={cfg['stakes']}, adm_w={cfg['adm_w']}"
        print(f"\n===== {market.upper()} ({time.time()-t0:.0f}s)  FINAL=({cfg_s}) =====", flush=True)
        for tag in ["DEPLOYED", "FINAL"]:
            deltas = " | ".join(f"O{ln} {pool(res[tag][ln]['m'])-pool(res[tag][ln]['b']):+.4f}" for ln in lines)
            f0 = " ".join(f"{s_[2][-2:]}{'+' if s_[0] < b_[0] else '-'}"
                          for s_, b_ in zip(res[tag][lines[0]]["m"], res[tag][lines[0]]["b"]))
            print(f"  {tag:9s}: {deltas}   [{f0} @O{lines[0]}]", flush=True)
        if s_lines:
            for tag in ["DEPLOYED", "FINAL"]:
                deltas = " | ".join(f"O{ln} {pool(res_s[tag][ln]['m'])-pool(res_s[tag][ln]['b']):+.4f}"
                                    for ln in s_lines)
                print(f"  SIDE {tag:9s}: {deltas}", flush=True)


if __name__ == "__main__":
    main()
