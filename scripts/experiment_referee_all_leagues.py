from __future__ import annotations

"""Referee feature for cards/fouls in ALL five leagues (ESPN referees).

The deployed referee feature is EPL-only because football-data carries a
Referee column for the Premier League alone; there it beat the base recipe on
all 6 card/foul lines, 5/5 folds (experiment_referee_epl.py). ESPN's match
summary names the referee in every big-five league from 2021/22 on
(scripts/fetch_espn_referees.py -> data/processed/espn_referees.csv).

Feature (walk-forward, pre-match information — the referee is announced days
ahead): ref_dev = the referee's mean deviation of the match total from his
league's previous-season mean, over his EARLIER matches only, shrunk n/(n+20)
toward 0. A deviation rather than a raw rate so one coefficient serves five
leagues with different card cultures (LaLiga ~5 yellows, Bundesliga ~4).
Rows without a known referee get 0 (= league-typical), which is what the model
will see for a referee with no history.

Arms on the DEPLOYED per-market recipe (module feature list, no Platt). With
--joint, both arms also carry the 2026-09-29 league-season terms (fouls lg_lvl
feature, yellows running residual ratio) so the stack is tested as deployed:
  DEP   deployed recipe       REF   + ref_dev
Test folds: 2023/24, 2024/25, 2025/26 (referee history from 2021/22 needs a
season to build up) + the 2026/27 matches so far. Scored on the four leagues
that do NOT have the feature today, and on the EPL separately.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.identity.normalization import canonical_team_name  # noqa: E402
from mundialytics.props import team_props as tp  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from experiment_league_level import running_ratio  # noqa: E402

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
REFS = ROOT / "data/processed/espn_referees.csv"
PREDS = [ROOT / "data/processed/enriched/understat_xg/walkforward_preds.csv",
         ROOT / "data/processed/enriched/understat_xg/walkforward_preds_hist.csv"]
TEST = ["2023-2024", "2024-2025", "2025-2026", "2026-2027"]
SHRINK = 20.0
SIDE = {"yellows": [1.5, 2.5]}
JOINT = "--joint" in sys.argv


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def attach_referees(full: pd.DataFrame) -> pd.DataFrame:
    """Join ESPN referees on (competition, date, canonical home) with an away fallback."""
    r = pd.read_csv(REFS, dtype={"event_id": str})
    r = r[r["referee"].fillna("").astype(str).str.len() > 0].copy()
    r["date"] = pd.to_datetime(r["date"])
    r["h"] = r["home_team"].astype(str).map(canonical_team_name)
    r["a"] = r["away_team"].astype(str).map(canonical_team_name)
    f = full.copy()
    f["h"] = f["home_team"].astype(str).map(canonical_team_name)
    f["a"] = f["away_team"].astype(str).map(canonical_team_name)
    j1 = f.merge(r[["competition", "date", "h", "referee"]], on=["competition", "date", "h"], how="left")
    miss = j1["referee"].isna()
    j2 = (f[miss.to_numpy()].merge(r[["competition", "date", "a", "referee"]],
                                   on=["competition", "date", "a"], how="left"))
    j1.loc[miss, "referee"] = j2["referee"].to_numpy()
    # ESPN dates are UTC: a late kick-off can sit on the next day — try +-1 day
    for shift in (-1, 1):
        miss = j1["referee"].isna()
        if not miss.any():
            break
        rr = r.assign(date=r["date"] + pd.Timedelta(days=shift))
        j3 = f[miss.to_numpy()].merge(rr[["competition", "date", "h", "referee"]],
                                      on=["competition", "date", "h"], how="left")
        j1.loc[miss, "referee"] = j3["referee"].to_numpy()
    return j1.drop(columns=["h", "a"]).drop_duplicates("match_id")


def ref_dev(m: pd.DataFrame, hc: str, ac: str) -> pd.Series:
    """Walk-forward shrunk mean deviation of the match total from the league's
    previous-season mean, per referee, over EARLIER dates only."""
    m = m.sort_values("date").copy()
    m["tot"] = (m[hc] + m[ac]).astype(float)
    smean = m.groupby(["competition", "season"])["tot"].mean()
    prev = {}
    for (c, s) in smean.index:
        ps = [x for (cc, x) in smean.index if cc == c and x < s]
        prev[(c, s)] = smean[(c, max(ps))] if ps else smean[(c, s)]
    m["dev"] = m["tot"] - [prev[(c, s)] for c, s in zip(m["competition"], m["season"])]
    out = pd.Series(0.0, index=m["match_id"].to_numpy())
    known = m[m["referee"].notna()]
    for _, g in known.groupby("referee"):
        by_day = g.groupby("date")["dev"].agg(["sum", "size"])
        cs = by_day["sum"].cumsum().shift(1, fill_value=0.0)
        cn = by_day["size"].cumsum().shift(1, fill_value=0)
        val = cs / (cn + SHRINK)            # = mean * n/(n+SHRINK)
        out.loc[g["match_id"].to_numpy()] = g["date"].map(val).to_numpy()
    return out


def main() -> None:
    t0 = time.time()
    lam = pd.concat([pd.read_csv(p)[["match_id", "lh", "la"]] for p in PREDS if p.exists()],
                    ignore_index=True).drop_duplicates("match_id")
    full = pd.read_csv(FOUND, low_memory=False)
    full["date"] = pd.to_datetime(full["date"], errors="coerce")
    full = full[full["season"] >= "2014-2015"].dropna(subset=["home_goals", "away_goals", "date"])
    full = full.merge(lam, on="match_id", how="left")
    full = attach_referees(full)
    cov = full[full.season >= "2021-2022"].groupby("competition")["referee"].apply(lambda s: s.notna().mean())
    print("referee coverage 2021/22+:\n" + cov.round(3).to_string(), flush=True)
    full, _ = tp.TeamPropsModel._add_positions(full)
    model = tp.TeamPropsModel()
    model._use_lam = True

    for market in ["yellows", "fouls"]:
        hc, ac, lines = tp.MARKETS[market]
        m = full.copy()
        for c in (hc, ac):
            m[c] = pd.to_numeric(m[c], errors="coerce")
        m = m.dropna(subset=[hc, ac])
        rd = ref_dev(m, hc, ac)
        m["ref_dev"] = m["match_id"].map(rd).fillna(0.0)
        if JOINT:
            # as deployed: the EPL keeps its own football-data referee model, so the
            # ESPN deviation is learned (and served) on the other four leagues only
            m.loc[m["competition"] == "Premier League", "ref_dev"] = 0.0
        extra = {"ref_dev": "ref_dev"}
        joint_lvl = JOINT and market in tp.LEVEL_K
        if joint_lvl:
            lvl, _ = tp._league_levels(m, hc, ac, tp.LEVEL_K[market])
            m["lg_lvl"] = m["match_id"].map(lvl)
            extra["lg_lvl"] = "lg_lvl"
        lr = tp.TeamPropsModel._long_rows(m, hc, ac, extra=extra, hl=tp.EWM_HL.get(market, 5))
        lr = lr.merge(m[["match_id", "competition", "season"]], on="match_id", how="left")
        lr["delta_lam"] = lr["delta_lam"].fillna(lr["delta"])
        lr["abs_delta_lam"] = lr["delta_lam"].abs()
        dep = model._feature_names(market) + (["lg_lvl"] if joint_lvl else [])
        arms = {"DEP": dep, "REF": dep + ["ref_dev"]}
        rows = []
        for s in TEST:
            te_m = m[m.season == s]
            if te_m.empty:
                continue
            s_start = te_m.date.min()
            tr_m = m[m.date < s_start]
            tt = (tr_m[hc] + tr_m[ac]).astype(float)
            disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
            sv = lr.loc[lr.date < s_start, "ev_for"].dropna().astype(float)
            disp_s = float(np.clip(sv.var() / max(sv.mean(), 1e-9), 0.9, 3.0))
            pv = {}
            coef = None
            for arm, feats in arms.items():
                tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
                reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
                if arm == "REF":
                    coef = float(reg.coef_[-1])
                te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
                te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 25)
                pv[arm] = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
            idx = pv["DEP"].index.intersection(pv["REF"].index)
            tei = te_m.set_index("match_id").loc[idx]
            has_ref = tei["referee"].notna().to_numpy()
            act_h, act_a = tei[hc].to_numpy(float), tei[ac].to_numpy(float)
            for arm in arms:
                lh, la = pv[arm].loc[idx, 1].to_numpy(), pv[arm].loc[idx, 0].to_numpy()
                if JOINT and market in tp.RES_K:
                    tmp = pd.DataFrame({"competition": tei["competition"].to_numpy(),
                                        "date": tei["date"].to_numpy(),
                                        "act_tot": act_h + act_a, "pred_tot": lh + la})
                    r = running_ratio(tmp, tp.RES_K[market]).to_numpy()
                    lh, la = lh * r, la * r
                for ln in lines:
                    y = (act_h + act_a > ln).astype(float)
                    p = tp._prob_over(lh + la, ln, disp)
                    rows.append(dict(season=s, arm=arm, kind=f"O{ln}", p=p, y=y,
                                     comp=tei["competition"].to_numpy(), has_ref=has_ref))
                for ln in SIDE.get(market, []):
                    ys = np.concatenate([act_h, act_a]) > ln
                    ps = tp._prob_over(np.concatenate([lh, la]), ln, disp_s)
                    rows.append(dict(season=s, arm=arm, kind=f"side{ln}", p=ps, y=ys.astype(float),
                                     comp=np.concatenate([tei["competition"].to_numpy()] * 2),
                                     has_ref=np.concatenate([has_ref, has_ref])))
            print(f"  {market} {s}: ref_dev coef {coef:+.4f}, matches {len(idx)}, "
                  f"with referee {has_ref.mean():.1%}", flush=True)

        print(f"\n===== {market.upper()} ({time.time() - t0:.0f}s) =====")
        for scope, sel in [("4 leagues (new)", lambda c: c != "Premier League"),
                           ("Premier League", lambda c: c == "Premier League")]:
            for kind in sorted({r["kind"] for r in rows}):
                deltas, wins, n = [], 0, 0
                for s in TEST:
                    rr = {r["arm"]: r for r in rows if r["season"] == s and r["kind"] == kind}
                    if not rr:
                        continue
                    msk = np.array([sel(c) for c in rr["DEP"]["comp"]]) & rr["DEP"]["has_ref"]
                    if msk.sum() < 50:
                        continue
                    d0 = bll(rr["DEP"]["y"][msk], rr["DEP"]["p"][msk])
                    d1 = bll(rr["REF"]["y"][msk], rr["REF"]["p"][msk])
                    deltas.append((d1 - d0, int(msk.sum()), s))
                    n += int(msk.sum())
                    wins += (d1 < d0) if s != "2026-2027" else 0
                hist = [d for d in deltas if d[2] != "2026-2027"]
                pooled = sum(d * k for d, k, _ in hist) / max(sum(k for _, k, _ in hist), 1)
                live = [d for d, _, s in deltas if s == "2026-2027"]
                print(f"  {scope:16s} {kind:8s} REF-DEP {pooled:+.5f} ({wins}/{len(hist)} folds)"
                      f" | 26/27 {live[0] if live else float('nan'):+.5f}  n={n}")


if __name__ == "__main__":
    main()
