"""Take the goals LEVEL off the market instead of off our own residuals.

Why this is the one route left. Two harnesses have now shown that reacting to
our own in-season residual fails -- shots (v0.56.0, 2/5 folds) and goals
(scripts/goals_level/, 1/6) -- and both fail the same way: in a normal season
the residual wobbles +/-2% with no persistence, so any K fast enough to catch
2026/27's +12.9% buys that noise in the other six. The problem is not the
correction, it is the ESTIMATOR: 47 of our own matches cannot separate a real
level shift from noise.

Bet365's closing over/under 2.5 price is a far sharper estimate of the same
scalar, and it is already on disk for every fold (football-data carries
B365C>2.5 from 2019/20).

**The line this deliberately does not cross.** Only a league-season LEVEL is
taken from the market, computed from prices for matches that had already been
played, and never the price of the match being predicted. Feeding the book's
per-match number straight into lambda would not be a model, it would be a copy,
and it would score well for exactly that reason -- the measured gap is that in a
logistic against the closing line our own weight is +0.042 against the book's
+1.038 (totals_gap_2026_10_05). Per-match discrimination stays entirely ours.

Forms (m shrunk to 1 with K pseudo-matches, clipped to [0.90, 1.10]):
  mkt_league  the book-implied level of THIS league-season, earlier dates only
  mkt_pooled  the same pooled over the five leagues

    python scripts/market_level/score_market_level.py [--k 20,40,70,120,200,400]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/goals_level"))
from score_level import chain  # noqa: E402

DUMPS = ROOT / "outputs/goals_level"
FD = ROOT / "data/raw/football_data"


def book_prices() -> pd.DataFrame:
    """date + raw team names -> de-vigged P(over 2.5) from the B365 closing line."""
    out = []
    for p in sorted(FD.glob("*.csv")):
        if not os.path.basename(p)[:4].isdigit():
            continue
        try:
            o = pd.read_csv(p, low_memory=False, encoding="latin-1")
        except Exception:
            continue
        hi, lo = "B365C>2.5", "B365C<2.5"
        if not {hi, lo, "Date", "HomeTeam", "AwayTeam"} <= set(o.columns):
            continue
        o = o.dropna(subset=[hi, lo, "HomeTeam", "AwayTeam"])
        if o.empty:
            continue
        io_, iu = 1.0 / o[hi], 1.0 / o[lo]
        out.append(pd.DataFrame({
            "fecha": pd.to_datetime(o["Date"], dayfirst=True).dt.strftime("%Y-%m-%d"),
            "home_team_raw": o["HomeTeam"].astype(str),
            "away_team_raw": o["AwayTeam"].astype(str),
            "p_book": (io_ / (io_ + iu)).to_numpy()}))
    return pd.concat(out, ignore_index=True).drop_duplicates(
        ["fecha", "home_team_raw", "away_team_raw"])


def implied_scale(lh: np.ndarray, la: np.ndarray, p_book: np.ndarray,
                  iters: int = 40) -> np.ndarray:
    """s such that the deployed chain on (lh*s, la*s) prices the book's P(over 2.5).

    Bisection on the whole vector at once: P(over) rises monotonically in a common
    multiplier, so 40 halvings of [0.3, 3.0] put s inside 1e-11.
    """
    lo = np.full(len(lh), 0.3)
    hi = np.full(len(lh), 3.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        _, p = chain(lh * mid, la * mid)
        too_low = p < p_book
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return (lo + hi) / 2


def running_mean(d: pd.DataFrame, by: list[str], col: str) -> pd.DataFrame:
    """Mean of `col` over matches STRICTLY BEFORE each date, per group."""
    day = (d.groupby(by + ["date"], sort=True)
             .agg(s=(col, "sum"), n=(col, "size")).reset_index())
    g = day.groupby(by, sort=False)
    day["cs"] = g["s"].cumsum() - day["s"]
    day["cn"] = g["n"].cumsum() - day["n"]
    return day[by + ["date", "cs", "cn"]]


def main() -> None:
    ks = [int(x) for x in (sys.argv[sys.argv.index("--k") + 1].split(",")
                           if "--k" in sys.argv else [20, 40, 70, 120, 200, 400])]
    from mundialytics.statistical_core.engine_utils import load_clubs_data
    d = pd.concat([pd.read_csv(p) for p in sorted(DUMPS.glob("wf_*.csv"))],
                  ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    d["fecha"] = d["date"].dt.strftime("%Y-%m-%d")

    f = load_clubs_data()
    f["fecha"] = f["date"].astype(str).str[:10]
    raw = (f[["fecha", "home_team", "away_team", "home_team_raw", "away_team_raw"]]
           .rename(columns={"home_team": "home", "away_team": "away"}))
    d = d.merge(raw, on=["fecha", "home", "away"], how="left")
    d = d.merge(book_prices(), on=["fecha", "home_team_raw", "away_team_raw"], how="left")
    print(f"{len(d)} matches | with a closing O/U line: {d.p_book.notna().sum()} "
          f"({d.p_book.notna().mean():.1%})")
    d = d.dropna(subset=["p_book"]).sort_values("date").reset_index(drop=True)

    d["s_book"] = implied_scale(d.lh.values, d.la.values, d.p_book.values)
    d["tot"] = d.hg + d.ag
    d["lam"] = d.lh + d.la
    print("\n=== what the book's level says, per fold ===")
    per = d.groupby("season").agg(n=("s_book", "size"), s_book=("s_book", "mean"),
                                  lam=("lam", "mean"), goles=("tot", "mean"))
    per["lam_libro"] = per.lam * per.s_book
    per["sesgo_nuestro"] = per.goles / per.lam
    per["sesgo_libro"] = per.goles / per.lam_libro
    print(per.round(4).to_string())

    d = d.merge(running_mean(d, ["season", "comp"], "s_book").rename(
        columns={"cs": "sl", "cn": "nl"}), on=["season", "comp", "date"], how="left")
    d = d.merge(running_mean(d, ["season"], "s_book").rename(
        columns={"cs": "sp", "cn": "np_"}), on=["season", "date"], how="left")

    o = np.where(d.hg > d.ag, 0, np.where(d.hg == d.ag, 1, 2))
    Y = np.eye(3)[o]
    over = (d.tot > 2.5).astype(float).values

    def score(m):
        P, o25 = chain(d.lh.values * m, d.la.values * m)
        return (((P.cumsum(1)[:, :2] - Y.cumsum(1)[:, :2]) ** 2).sum(1) / 2,
                -(over * np.log(np.clip(o25, 1e-9, 1))
                  + (1 - over) * np.log(np.clip(1 - o25, 1e-9, 1))))

    b_rps, b_ou = score(np.ones(len(d)))
    print(f"\nBASELINE on these {len(d)} matches: 1X2 RPS {b_rps.mean():.5f} | "
          f"O/U {b_ou.mean():.5f}")

    rows = []
    for form, s_col, n_col in (("mkt_league", "sl", "nl"), ("mkt_pooled", "sp", "np_")):
        for k in ks:
            n = np.nan_to_num(d[n_col].values.astype(float))
            s = np.nan_to_num(d[s_col].values.astype(float))
            m = np.clip((s + k * 1.0) / (n + k), 0.90, 1.10)   # sum/(n+k) shrunk to 1
            rps, ou = score(m)
            pf = d.assign(a=rps - b_rps, b=ou - b_ou).groupby("season")[["a", "b"]].mean()
            hist = pf.drop(index=[i for i in pf.index if i.startswith("2026")],
                           errors="ignore")
            rows.append({"form": form, "K": k, "m_mean": m.mean(),
                         "d_rps": (rps - b_rps).mean(),
                         "rps_hist": f"{int((hist.a < 0).sum())}/{len(hist)}",
                         "d_ou": (ou - b_ou).mean(),
                         "ou_hist": f"{int((hist.b < 0).sum())}/{len(hist)}",
                         "d_rps_2627": pf.a.get("2026-2027", np.nan),
                         "d_ou_2627": pf.b.get("2026-2027", np.nan)})
    r = pd.DataFrame(rows)
    print("\n=== level from the market (negative = better) ===")
    print(r.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    best = r.sort_values("d_ou").iloc[0]
    n = np.nan_to_num(d[("sl" if best.form == "mkt_league" else "sp") + "" ].values.astype(float)) \
        if False else None
    col_s, col_n = ("sl", "nl") if best.form == "mkt_league" else ("sp", "np_")
    m = np.clip((np.nan_to_num(d[col_s].values.astype(float)) + best.K)
                / (np.nan_to_num(d[col_n].values.astype(float)) + best.K), 0.90, 1.10)
    rps, ou = score(m)
    pf = d.assign(d_rps=rps - b_rps, d_ou=ou - b_ou).groupby("season")[["d_rps", "d_ou"]].mean()
    print(f"\n=== best on O/U: {best.form} K={int(best.K)} -- per fold ===")
    print(pf.round(5).to_string())
    dd = (ou - b_ou)
    se = dd.std(ddof=1) / np.sqrt(len(dd))
    print(f"\nO/U overall {dd.mean():+.5f} +/- {se:.5f}  t={dd.mean() / se:+.2f}")


if __name__ == "__main__":
    main()
