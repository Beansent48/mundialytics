from __future__ import annotations

"""Team-props candidate variants over the deployed recipe (rate model with
rolling+EWMA, opp mirror, is_home, goals-delta + lambda-delta, NB):

  GBT     HistGradientBoosting (poisson loss) instead of the linear regressor —
          do interactions matter with ~30k side-samples?
  STAKES  cards/fouls only: walk-forward league positions (points before the
          match), |pos diff|, relegation-battle flag x season stage, round_frac.
  VENUE   venue-matched extra rollings (home-only / away-only r10 of ev_for).
  ROUND   season-stage feature alone (round_frac).
  STYLE   corners only: corners-per-shot EWM (crossing-style persistence).
  EWM-HL  halflife grid {3, 8, 12} for the ev EWMs (deployed = 5).

Same folds/eval as always; anything beating DEPLOYED consistently gets in.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
PREDS = ROOT / "data/processed/enriched/understat_xg/walkforward_preds.csv"
PREDS_H = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_hist.csv"
W = (5, 10, 19)
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
MARKETS = {
    "corners": ("home_corners", "away_corners", [8.5, 9.5, 10.5]),
    "yellows": ("home_yellow_cards", "away_yellow_cards", [3.5, 4.5, 5.5]),
    "fouls":   ("home_fouls", "away_fouls", [21.5, 23.5]),
    "shots":   ("home_shots", "away_shots", [22.5, 24.5]),
    "sot":     ("home_sot", "away_sot", [7.5, 8.5]),
}


def prob_over(lam, line, disp):
    k = int(np.floor(line))
    lam = np.clip(lam, 0.2, 60.0)
    r = lam / (disp - 1.0)
    return 1.0 - nbinom.cdf(k, r, 1.0 / disp)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def add_positions(full: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward league position + games played BEFORE each match."""
    full = full.sort_values("date").copy()
    ph, pa, gh = [], [], []
    for (_, _), g in full.groupby(["competition", "season"], sort=False):
        pts: dict = {}
        played: dict = {}
        for r in g.itertuples(index=False):
            def rank(t):
                p = pts.get(t, 0)
                return 1 + sum(1 for v in pts.values() if v > p)
            ph.append((r.match_id, rank(r.home_team), rank(r.away_team),
                       played.get(r.home_team, 0)))
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

    for market, (hc, ac, lines) in MARKETS.items():
        t0 = time.time()
        m = df.dropna(subset=[hc, ac, "lh", "la"]).copy()
        for c in [hc, ac, "home_shots", "away_shots"]:
            if c in m.columns:
                m[c] = pd.to_numeric(m[c], errors="coerce")
        m = m.dropna(subset=[hc, ac])
        rows = []
        for r in m.itertuples(index=False):
            base = dict(match_id=r.match_id, date=r.date, season=r.season, comp=r.competition)
            sh_h = getattr(r, "home_shots", np.nan)
            sh_a = getattr(r, "away_shots", np.nan)
            rows.append(dict(**base, team=r.home_team, opp=r.away_team, is_home=1,
                             ev_for=getattr(r, hc), ev_against=getattr(r, ac),
                             gf=r.home_goals, ga=r.away_goals, lam_t=r.lh, lam_o=r.la,
                             pos_t=r.pos_home, pos_o=r.pos_away, played=r.played_home,
                             sh_for=sh_h))
            rows.append(dict(**base, team=r.away_team, opp=r.home_team, is_home=0,
                             ev_for=getattr(r, ac), ev_against=getattr(r, hc),
                             gf=r.away_goals, ga=r.home_goals, lam_t=r.la, lam_o=r.lh,
                             pos_t=r.pos_away, pos_o=r.pos_home, played=r.played_home,
                             sh_for=sh_a))
        lr = pd.DataFrame(rows).sort_values(["team", "date", "match_id"])
        gb = lr.groupby("team", group_keys=False)
        for col in ["ev_for", "ev_against", "gf", "ga"]:
            for w in W:
                lr[f"{col}_r{w}"] = gb[col].apply(lambda s: s.shift(1).rolling(w, min_periods=3).mean())
            lr[f"{col}_ewm"] = gb[col].apply(lambda s: s.shift(1).ewm(halflife=5, min_periods=3).mean())
        for hl in (3, 8, 12):
            lr[f"ev_for_ewm{hl}"] = gb["ev_for"].apply(lambda s: s.shift(1).ewm(halflife=hl, min_periods=3).mean())
            lr[f"ev_against_ewm{hl}"] = gb["ev_against"].apply(lambda s: s.shift(1).ewm(halflife=hl, min_periods=3).mean())
        # venue-matched r10 (rolling over same-venue rows only)
        lr["ev_for_venue_r10"] = (lr.groupby(["team", "is_home"], group_keys=False)["ev_for"]
                                  .apply(lambda s: s.shift(1).rolling(10, min_periods=3).mean()))
        if market == "corners":
            lr["cps"] = lr["ev_for"] / lr["sh_for"].clip(lower=1)
            lr["cps_ewm"] = gb["cps"].apply(lambda s: s.shift(1).ewm(halflife=8, min_periods=3).mean())
        lr["round_frac"] = (lr["played"] / 38.0).clip(0, 1)
        lr["pos_diff_abs"] = (lr["pos_t"] - lr["pos_o"]).abs()
        lr["releg_battle"] = (((lr["pos_t"] >= 15) | (lr["pos_o"] >= 15)) & (lr["round_frac"] > 0.6)).astype(float)

        opp_src = ([f"ev_against_r{w}" for w in W] + ["ev_against_ewm", "gf_ewm", "ga_ewm"]
                   + [f"ev_against_ewm{hl}" for hl in (3, 8, 12)])
        opp = lr[["match_id", "team"] + opp_src].rename(
            columns={"team": "opp", **{c: f"opp_{c}" for c in opp_src}})
        lr = lr.merge(opp, on=["match_id", "opp"], how="left")
        lr["delta"] = (lr["gf_ewm"] + lr["opp_ga_ewm"]) / 2 - (lr["opp_gf_ewm"] + lr["ga_ewm"]) / 2
        lr["abs_delta"] = lr["delta"].abs()
        lr["delta_lam"] = lr["lam_t"] - lr["lam_o"]
        lr["abs_delta_lam"] = lr["delta_lam"].abs()

        base_feats = ([f"ev_for_r{w}" for w in W] + ["ev_for_ewm"]
                      + [f"opp_ev_against_r{w}" for w in W] + ["opp_ev_against_ewm"]
                      + ["is_home", "delta", "abs_delta", "delta_lam", "abs_delta_lam"])
        variants: dict[str, tuple[str, list[str]]] = {
            "DEPLOYED": ("lin", base_feats),
            "GBT": ("gbt", base_feats),
            "+VENUE": ("lin", base_feats + ["ev_for_venue_r10"]),
            "+ROUND": ("lin", base_feats + ["round_frac"]),
        }
        if market in ("yellows", "fouls"):
            variants["+STAKES"] = ("lin", base_feats + ["pos_diff_abs", "releg_battle", "round_frac"])
        if market == "corners":
            variants["+STYLE"] = ("lin", base_feats + ["cps_ewm"])
        for hl in (3, 8, 12):
            fs = [f.replace("ev_for_ewm", f"ev_for_ewm{hl}").replace("opp_ev_against_ewm", f"opp_ev_against_ewm{hl}")
                  if f in ("ev_for_ewm", "opp_ev_against_ewm") else f for f in base_feats]
            variants[f"EWM-HL{hl}"] = ("lin", fs)

        print(f"\n===== {market.upper()} ({time.time()-t0:.0f}s prep) =====", flush=True)
        for tag, (kind, feats) in variants.items():
            res = {ln: {"m": [], "b": []} for ln in lines}
            for s in TEST_SEASONS:
                te_m = m[m.season == s]
                if len(te_m) == 0:
                    continue
                s_start = te_m.date.min()
                tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
                if len(tr) < 2000:
                    continue
                if kind == "gbt":
                    reg = HistGradientBoostingRegressor(loss="poisson", max_depth=4, max_iter=300,
                                                        learning_rate=0.06, min_samples_leaf=100,
                                                        random_state=42)
                else:
                    reg = PoissonRegressor(alpha=0.1, max_iter=1000)
                reg.fit(tr[feats], tr["ev_for"].clip(lower=0))
                te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
                te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 40)
                pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
                tot = (pv[1] + pv[0]).to_numpy()
                tr_tot = m[m.date < s_start]
                tt = (tr_tot[hc] + tr_tot[ac]).astype(float)
                disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 1.11, 3.0))
                tei = te_m.set_index("match_id").loc[pv.index]
                lg = tr_tot.assign(t=tt).groupby("competition")["t"].mean()
                base_v = tei["competition"].map(lg).fillna(float(tt.mean())).to_numpy()
                act = tei[[hc, ac]].sum(axis=1).astype(float)
                for ln in lines:
                    y = (act > ln).astype(float).to_numpy()
                    res[ln]["m"].append((bll(y, prob_over(tot, ln, disp)), len(y), s))
                    res[ln]["b"].append((bll(y, prob_over(base_v, ln, disp)), len(y), s))
            pool = lambda a: sum(x * n for x, n, _ in a) / max(sum(n for _, n, _ in a), 1)
            deltas = " | ".join(f"O{ln} {pool(res[ln]['m'])-pool(res[ln]['b']):+.4f}" for ln in lines)
            f0 = " ".join(f"{s_[2][-2:]}{'+' if s_[0] < b_[0] else '-'}"
                          for s_, b_ in zip(res[lines[0]]["m"], res[lines[0]]["b"]))
            print(f"  {tag:10s}: {deltas}   [{f0} @O{lines[0]}]", flush=True)


if __name__ == "__main__":
    main()
