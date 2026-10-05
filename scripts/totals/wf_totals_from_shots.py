"""Do the shot expectations know something about TOTAL GOALS that the goal
lambdas don't?

THE SYMMETRY. Every goal market -- over/under, BTTS, the scoreline grid, the
half-time ladder -- is read off the two lambdas. Squad value and lineup absence
tilt them while PRESERVING THE TOTAL (lh*e^{s/2}, la*e^{-s/2}): they move who
wins, which is why they bought 1X2 and left over/under alone. The mirror image
has never been tested: preserve the RATIO and move the TOTAL. That is the only
shape that can improve over/under without touching 1X2.

THE SIGNAL. TeamPropsModel predicts shots and shots on target per side from a
feature set the goal model does not have -- market-specific EWMA halflives, the
MLE-strengths prior, the league-season level, the running residual -- and SoT
converts to goals at a rate that barely moves. Those numbers are computed for
every fixture already and never reach the goal markets. The question is whether
they carry anything beyond lambda_total, or merely restate it (team props takes
the engine's lambdas as its own ASYM features, so the arms are correlated by
construction and a null result is a real possibility).

THE TEST. Walk-forward 2021/22-2025/26, both models fitted per season on
everything before it, engine xG form walked forward as production walks it. Then
LEAVE-ONE-SEASON-OUT on the collected rows: a Poisson regression of actual total
goals on [log lambda_total, log sot_total, log shots_total] is fitted on the four
other seasons and applied to the held-out one, lambdas scaled by
predicted_total/lambda_total, and every goal market rebuilt from them with the
deployed parameters.

Arms:
    base      the deployed markets
    lam       correction on log lambda_total alone (does the engine mis-level
              totals at all? the control that says whether shots matter)
    sot       + log sot_total
    shots     + log sot_total + log shots_total

Bar: over-2.5 log-loss better in 5/5 seasons AND 1X2 RPS not worse. A ratio-
preserving scaling cannot move 1X2 much, but "cannot much" is not "does not".

RESULT 2026-10-05: NEGATIVE, and the control arm is why.

    market      base       lam       sot     shots
    O1.5     0.53200  +0.00008  -0.00003  +0.00018
    O2.5     0.67746  +0.00028  +0.00025  +0.00035
    O3.5     0.60371  +0.00055  +0.00064  +0.00072
    BTTS     0.68416  +0.00020  +0.00017  +0.00023
    1X2 RPS  0.20044  -0.00002  -0.00000  +0.00000

Every arm makes the goal markets slightly WORSE; O2.5 wins 2-3 seasons of 5.
1X2 is untouched, which confirms the scaling is doing what it was built to do.

The control settles it: `lam` is a FREE RE-LEVELLING of the totals -- fit a
Poisson on log lambda_total alone, four seasons in, apply to the fifth -- and it
does not help either, with a fitted coefficient of +0.971 on log lambda_total.
That coefficient is the finding. The engine's lambda_total is already an
unbiased predictor of total goals in log space, so there was no mis-levelling
for the shots to correct. SoT does carry a little signal on its own (+0.152,
with lambda's weight dropping 0.971 -> 0.886 to absorb it) but it does not
survive as better probabilities out of sample.

Nor is it a calibration problem. A LOSO Platt on the logit fits slopes of
0.87-0.94 (mild over-confidence, visible as +0.0287 over-prediction in our top
O2.5 decile) and still loses: O1.5 -0.00006 3/5, O2.5 +0.00016 3/5, O3.5
+0.00051 2/5, BTTS +0.00021 2/5. Same verdict as the 1X2 away-favourite
investigation: noise and resolution, not calibration.

THE NUMBER THAT CLOSES THE DIRECTION. Against Bet365's de-vigged closing price
on the same 8,579 matches: always-the-base-rate 0.69147, ours 0.67774, the book
0.66976 -- we capture 63% of the signal that exists, and the entire market only
holds 0.0225 of log-loss over doing nothing. A logistic regression on both
prices together puts weight +1.038 on the book and **+0.042 on ours**: once the
closing price is known, our probability adds nothing. We hold no information the
book does not already have. See [[project_market_gap_diagnosis]], which reached
the same place for 1X2 on 2026-08-26.

    python scripts/totals/wf_totals_from_shots.py            (~15 min)
    python scripts/totals/wf_totals_from_shots.py --report   (re-score the CSV)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src")]

from sklearn.linear_model import PoissonRegressor  # noqa: E402

from mundialytics.props import TeamPropsModel  # noqa: E402
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS, PredictionEngine, deployed_markets_from_lambdas)

FOUNDATION = ROOT / "data/processed/foundation_big5_multi_season.csv"
OUT = ROOT / "data/processed/logs/wf_totals_from_shots.csv"
SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
ARMS = {"lam": ["l_lam"], "sot": ["l_lam", "l_sot"], "shots": ["l_lam", "l_sot", "l_shots"]}


def rps3(y: np.ndarray, P: np.ndarray) -> float:
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return float(((cp - cy) ** 2)[:, :2].sum(axis=1).mean() / 2)


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def collect() -> pd.DataFrame:
    df = pd.read_csv(FOUNDATION, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals", "home_xg", "away_xg", "date"])
    df = df.sort_values("date")

    rows = []
    for s in SEASONS:
        test = df[df["season"] == s].sort_values("date")
        train = df[df["date"] < test["date"].min()]
        if len(test) == 0 or len(train) < 500:
            continue
        t0 = time.time()
        eng = PredictionEngine(**DEPLOYED_CLUB_ENGINE_KWARGS).fit(
            train, squad_value_asof=test["date"].min())
        tp = TeamPropsModel().fit(train, root=ROOT)
        print(f"  {s}: fitted in {time.time()-t0:.0f}s", flush=True)

        for r in test.itertuples(index=False):
            h, a = str(r.home_team), str(r.away_team)
            try:
                p = eng.predict_match(h, a, competition=str(r.competition),
                                      neutral=bool(getattr(r, "neutral", 0)))
            except Exception:
                continue
            try:
                fx = tp.predict_fixture(h, a, lam_home=p.lambda_home, lam_away=p.lambda_away)
            except Exception:
                fx = {}
            sot = (fx.get("sot") or {}).get("lambda_total")
            sh = (fx.get("shots") or {}).get("lambda_total")
            rows.append({"season": s, "match_id": r.match_id, "date": r.date,
                         "hg": int(r.home_goals), "ag": int(r.away_goals),
                         "lh": p.lambda_home, "la": p.lambda_away,
                         "sot_total": sot, "shots_total": sh,
                         "ph": p.p_home_win, "pd": p.p_draw, "pa": p.p_away_win,
                         "po15": p.p_over_15, "po25": p.p_over_25,
                         "po35": p.p_over_35, "btts": p.p_btts})
            if eng.xg_rate_model_ is not None:
                eng.xg_rate_model_.update_form(r.home_team, r.away_team, r.home_xg, r.away_xg)
        print(f"  {s}: {len(test)} matches, {time.time()-t0:.0f}s total", flush=True)

    m = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT, index=False)
    print(f"\nWROTE {OUT} ({len(m)} rows)\n", flush=True)
    return m


def report(m: pd.DataFrame) -> None:
    m = m.dropna(subset=["sot_total", "shots_total"]).copy()
    m["tot"] = m.hg + m.ag
    m["lam"] = m.lh + m.la
    m["l_lam"] = np.log(m["lam"])
    m["l_sot"] = np.log(m["sot_total"])
    m["l_shots"] = np.log(m["shots_total"])
    m["y3"] = np.where(m.hg > m.ag, 0, np.where(m.hg == m.ag, 1, 2))
    print(f"rows with both shot expectations: {len(m)}\n")

    # LOSO: fit the correction on the other seasons, apply to the held-out one
    out = {arm: {} for arm in ARMS}
    for arm, feats in ARMS.items():
        for s in sorted(m.season.unique()):
            tr, te = m[m.season != s], m[m.season == s].copy()
            mod = PoissonRegressor(alpha=1e-4, max_iter=2000).fit(tr[feats], tr["tot"])
            pred = np.clip(mod.predict(te[feats]), 0.3, 7.0)
            k = pred / te["lam"].to_numpy()          # ratio-preserving scaling
            mk = [deployed_markets_from_lambdas(lh * f, la * f)[0]
                  for lh, la, f in zip(te.lh, te.la, k)]
            te["n_po25"] = [d["p_over_25"] for d in mk]
            te["n_po15"] = [d["p_over_15"] for d in mk]
            te["n_po35"] = [d["p_over_35"] for d in mk]
            te["n_btts"] = [d["p_btts"] for d in mk]
            te["n_ph"] = [d["p_home_win"] for d in mk]
            te["n_pd"] = [d["p_draw"] for d in mk]
            te["n_pa"] = [d["p_away_win"] for d in mk]
            out[arm][s] = te

    lines = [("O1.5", "po15", "n_po15", 1.5), ("O2.5", "po25", "n_po25", 2.5),
             ("O3.5", "po35", "n_po35", 3.5)]
    print("over/under log-loss, leave-one-season-out (negative delta = better)")
    print(f"{'market':8s} {'base':>9s}" + "".join(f"{a:>11s}" for a in ARMS))
    for label, b, n, ln in lines:
        row = f"{label:8s}"
        allm = pd.concat([out['lam'][s] for s in out['lam']])
        y = (allm["tot"] > ln).astype(float).to_numpy()
        base = logloss(allm[b].to_numpy(), y)
        row += f"{base:9.5f}"
        for arm in ARMS:
            g = pd.concat([out[arm][s] for s in out[arm]])
            yy = (g["tot"] > ln).astype(float).to_numpy()
            row += f"{logloss(g[n].to_numpy(), yy)-base:+11.5f}"
        print(row)

    g0 = pd.concat([out['lam'][s] for s in out['lam']])
    yb = ((g0.hg > 0) & (g0.ag > 0)).astype(float).to_numpy()
    base_b = logloss(g0["btts"].to_numpy(), yb)
    row = f"{'BTTS':8s}{base_b:9.5f}"
    for arm in ARMS:
        g = pd.concat([out[arm][s] for s in out[arm]])
        yy = ((g.hg > 0) & (g.ag > 0)).astype(float).to_numpy()
        row += f"{logloss(g['n_btts'].to_numpy(), yy)-base_b:+11.5f}"
    print(row)

    base_r = rps3(g0["y3"].to_numpy(), g0[["ph", "pd", "pa"]].to_numpy())
    row = f"{'1X2 RPS':8s}{base_r:9.5f}"
    for arm in ARMS:
        g = pd.concat([out[arm][s] for s in out[arm]])
        row += f"{rps3(g['y3'].to_numpy(), g[['n_ph','n_pd','n_pa']].to_numpy())-base_r:+11.5f}"
    print(row)

    print("\nO2.5 log-loss per season (seasons the arm wins)")
    for arm in ARMS:
        won, deltas = 0, []
        for s in sorted(m.season.unique()):
            te = out[arm][s]
            y = (te["tot"] > 2.5).astype(float).to_numpy()
            b = logloss(te["po25"].to_numpy(), y)
            v = logloss(te["n_po25"].to_numpy(), y)
            deltas.append(v - b)
            won += v < b
        print(f"  {arm:6s} {won}/{len(deltas)}  " + " ".join(f"{d:+.5f}" for d in deltas))

    print("\nwhat the correction does to the level (pooled)")
    for arm in ARMS:
        g = pd.concat([out[arm][s] for s in out[arm]])
        feats = ARMS[arm]
        mod = PoissonRegressor(alpha=1e-4, max_iter=2000).fit(m[feats], m["tot"])
        coef = ", ".join(f"{f}={c:+.3f}" for f, c in zip(feats, mod.coef_))
        print(f"  {arm:6s} mean lambda_total {g['lam'].mean():.3f} -> "
              f"{(g['lam'] * 1.0).mean():.3f} | actual {g['tot'].mean():.3f} | {coef}")

    print("\nBar: O2.5 better in 5/5 seasons AND 1X2 RPS not worse.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="re-score the saved CSV")
    args = ap.parse_args()
    m = pd.read_csv(OUT, parse_dates=["date"]) if args.report else collect()
    report(m)


if __name__ == "__main__":
    main()
