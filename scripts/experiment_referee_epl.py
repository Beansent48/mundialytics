from __future__ import annotations

"""Referee features for cards/fouls — EPL only (the one league where
football-data.co.uk carries Referee, 100% coverage every season).

Walk-forward referee tendency: expanding mean of match totals (yellows, fouls)
per referee, shifted (never sees the current match), shrunk toward the global
mean with n/(n+20). A/B on EPL-only models: base recipe vs base + ref rate.
Referee assignments are published pre-match, so this is legitimate pre-match
information in production too.
"""

import glob
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
W = (5, 10, 19)
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
MARKETS = {
    "yellows": ("home_yellow_cards", "away_yellow_cards", [3.5, 4.5, 5.5], "ref_yc"),
    "fouls":   ("home_fouls", "away_fouls", [19.5, 21.5, 23.5], "ref_foul"),
}


def load_referees() -> pd.DataFrame:
    rows = []
    for p in glob.glob(str(ROOT / "data/raw/football_data/**/*E0.csv"), recursive=True):
        m = re.search(r"(\d{4})_E0\.csv$", p)
        if not m:
            continue
        try:
            df = pd.read_csv(p, encoding="latin-1", on_bad_lines="skip",
                             usecols=lambda c: c in {"Date", "HomeTeam", "AwayTeam", "Referee",
                                                     "HY", "AY", "HF", "AF"})
        except Exception:
            continue
        if "Referee" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        rows.append(df)
    r = pd.concat(rows, ignore_index=True).dropna(subset=["date", "Referee"])
    r = r.drop_duplicates(subset=["date", "HomeTeam", "AwayTeam"])  # glob may hit season files twice
    r["home_team"] = r["HomeTeam"].astype(str).str.lower().str.strip()
    r["away_team"] = r["AwayTeam"].astype(str).str.lower().str.strip()
    r["ref"] = r["Referee"].astype(str).str.strip()
    r["tot_yc"] = pd.to_numeric(r["HY"], errors="coerce") + pd.to_numeric(r["AY"], errors="coerce")
    r["tot_f"] = pd.to_numeric(r["HF"], errors="coerce") + pd.to_numeric(r["AF"], errors="coerce")
    r = r.sort_values("date")
    for src, out in [("tot_yc", "ref_yc"), ("tot_f", "ref_foul")]:
        g = r.groupby("ref", group_keys=False)[src]
        mean_prev = g.apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
        n_prev = g.cumcount()
        glob_mean = r[src].expanding().mean().shift(1).fillna(r[src].mean())
        cred = n_prev / (n_prev + 20.0)
        r[out] = cred * mean_prev.fillna(glob_mean) + (1 - cred) * glob_mean
    return r[["date", "home_team", "away_team", "ref", "ref_yc", "ref_foul"]]


def prob_over(lam, line, disp):
    k = int(np.floor(line))
    lam = np.clip(lam, 0.2, 40.0)
    r = lam / (disp - 1.0)
    return 1.0 - nbinom.cdf(k, r, 1.0 / disp)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    refs = load_referees()
    print(f"referee rows: {len(refs)}, distinct refs: {refs.ref.nunique()}")
    print(f"ref_yc spread (p5-p95): {refs.ref_yc.quantile(0.05):.2f}-{refs.ref_yc.quantile(0.95):.2f}")

    df = pd.read_csv(FOUND, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[(df["season"] >= "2014-2015") & (df["competition"].str.contains("Premier", case=False, na=False))]
    df = df.merge(refs, on=["date", "home_team", "away_team"], how="left")
    df = df.drop_duplicates(subset=["match_id"])
    print(f"EPL matches 2014+: {len(df)}, with referee joined: {df['ref'].notna().mean():.1%}")

    for market, (hc, ac, lines, ref_col) in MARKETS.items():
        t0 = time.time()
        m = df.dropna(subset=[hc, ac, "date"]).copy()
        m[hc] = pd.to_numeric(m[hc], errors="coerce")
        m[ac] = pd.to_numeric(m[ac], errors="coerce")
        m = m.dropna(subset=[hc, ac])
        rows = []
        for r in m.itertuples(index=False):
            rv = getattr(r, ref_col)
            rows.append(dict(match_id=r.match_id, date=r.date, team=r.home_team, opp=r.away_team,
                             is_home=1, ev_for=getattr(r, hc), ev_against=getattr(r, ac), ref_rate=rv))
            rows.append(dict(match_id=r.match_id, date=r.date, team=r.away_team, opp=r.home_team,
                             is_home=0, ev_for=getattr(r, ac), ev_against=getattr(r, hc), ref_rate=rv))
        lr = pd.DataFrame(rows).sort_values(["team", "date", "match_id"])
        for col in ["ev_for", "ev_against"]:
            for w in W:
                lr[f"{col}_r{w}"] = (lr.groupby("team", group_keys=False)[col]
                                     .apply(lambda s: s.shift(1).rolling(w, min_periods=3).mean()))
            lr[f"{col}_ewm"] = (lr.groupby("team", group_keys=False)[col]
                                .apply(lambda s: s.shift(1).ewm(halflife=5, min_periods=3).mean()))
        opp_src = [f"ev_against_r{w}" for w in W] + ["ev_against_ewm"]
        opp = lr[["match_id", "team"] + opp_src].rename(
            columns={"team": "opp", **{c: f"opp_{c}" for c in opp_src}})
        lr = lr.merge(opp, on=["match_id", "opp"], how="left")
        base_feats = ([f"ev_for_r{w}" for w in W] + ["ev_for_ewm"]
                      + [f"opp_ev_against_r{w}" for w in W] + ["opp_ev_against_ewm"] + ["is_home"])
        aug_feats = base_feats + ["ref_rate"]

        print(f"\n===== {market.upper()} EPL-only ({time.time()-t0:.0f}s prep) =====")
        for tag, feats in [("BASE", base_feats), ("+REF", aug_feats)]:
            res = {ln: {"m": [], "b": []} for ln in lines}
            for s in TEST_SEASONS:
                te_m = m[m.season == s]
                if len(te_m) == 0:
                    continue
                s_start = te_m.date.min()
                tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
                if len(tr) < 1500:
                    continue
                reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
                te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
                te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 25)
                pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
                tot = pv[1] + pv[0]
                tr_tot = m[m.date < s_start]
                tt = (tr_tot[hc] + tr_tot[ac]).astype(float)
                disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 1.11, 3.0))
                tei = te_m.set_index("match_id").loc[tot.index]
                base_v = np.full(len(tot), float(tt.mean()))
                act = tei[[hc, ac]].sum(axis=1).astype(float)
                for ln in lines:
                    y = (act > ln).astype(float).to_numpy()
                    res[ln]["m"].append((bll(y, prob_over(tot.to_numpy(), ln, disp)), len(y), s))
                    res[ln]["b"].append((bll(y, prob_over(base_v, ln, disp)), len(y), s))
            pool = lambda a: sum(x * n for x, n, _ in a) / sum(n for _, n, _ in a)
            for ln in lines:
                folds = " ".join(f"{s_[2][-2:]}{'+' if s_[0] < b_[0] else '-'}"
                                 for s_, b_ in zip(res[ln]["m"], res[ln]["b"]))
                print(f"  {tag} O/U {ln}: LL {pool(res[ln]['m']):.4f} vs mean-base {pool(res[ln]['b']):.4f} "
                      f"delta {pool(res[ln]['m'])-pool(res[ln]['b']):+.4f}  [{folds}]", flush=True)


if __name__ == "__main__":
    main()
