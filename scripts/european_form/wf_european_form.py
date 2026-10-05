"""Does a club's European evidence belong in its form? Walk-forward A/B.

THE GAP. The engine trains and walks its form on domestic league matches only.
A side in Europe plays a median 8-12 more competitive matches a season, and the
xG-rate model -- the component worth 60% of the goal lambda, and the one whose
WALK-FORWARD form was the big 2026-07 win (~0.004 RPS) -- never sees one of them.

THE TEST. Nothing about the fit changes: the engine is fitted exactly as
production fits it, on league matches only. The arms differ in one thing, what
gets appended to the walk-forward form state as the season runs:

  base      league matches only                       (production today)
  eu        + European matches at pseudo-xG           (the whole idea)
  eu_half   + European matches, shrunk halfway to the team's current form mean
            (a crude damper: we cannot adjust for the opponent -- ClubElo's
            history resolves only ~37% of European opponents by name -- so a
            romp against a minnow would otherwise inflate the form outright)

Bar, per [[feedback_protect_deployed_baseline]]: 1X2 RPS better in 5/5 seasons,
AND over-2.5 log-loss not worse. The second half is not a formality -- every
context signal fitted on 1X2 alone so far bought RPS by moving totals.

RESULT 2026-10-05: NEGATIVE. Do not retry in this form.

    season       base         eu     eu_half
    2021-2022  0.19983    0.19982    0.19983
    2022-2023  0.20461    0.20460    0.20448
    2023-2024  0.19373    0.19349    0.19361
    2024-2025  0.19809    0.19851    0.19833
    2025-2026  0.20252    0.20244    0.20237
    POOLED     0.19986    0.19986    0.19981

    eu       RPS +0.00000 (4/5)   O2.5 log-loss +0.00054  worse
    eu_half  RPS -0.00004 (4/5)   O2.5 log-loss +0.00037  worse

Pooled RPS is zero, and both arms make the totals worse. What kills it is the
slice where the signal must be strongest -- a side that played in Europe within
the last 7 days (n=1,495): eu is **+0.00023 WORSE** there, eu_half -0.00002,
while on the control (neither side in Europe for 14 days, n=6,459) both are
-0.00004, i.e. noise. The effect is not diluted, it is REVERSED exactly where it
applies. And it is monotone in weight: the more of the European match you let
in, the worse it gets, which is the signature of a biased observation, not of an
underused one. European evidence as built here does not add information; it
pushes league matches out of the rolling windows (cap 19, EWMA halflife 5) and
replaces them with a different competition, a different scoring level and an
opponent nobody modelled.

Two repairs exist and NEITHER was tested, so the idea is not disproven, only
this form of it: (1) real European xG instead of goals -- needs ESPN commentary
for ~2,000 UEFA matches, since Understat never covered them; (2) an opponent
adjustment -- ClubElo's API has been 502 since September and its local histories
resolve 85% of these matches, but a Poisson attack/defence fitted on the 1,894
European matches themselves would need no external source at all. Given a
measured effect of zero and a weight gradient pointing the wrong way, neither
looked worth its build cost on 2026-10-05.

    python scripts/european_form/wf_european_form.py [--seasons 2023-2024,...]

Writes a per-match CSV next to this file so arms can be re-scored without refitting.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/european_form")]

from european_matches import (  # noqa: E402
    big_five_rows, load_european_matches, pseudo_xg)
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS, PredictionEngine)

FOUNDATION = ROOT / "data/processed/foundation_big5_multi_season.csv"
OUT = Path(__file__).with_name("wf_european_form.csv")
SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
ARMS = ("base", "eu", "eu_half")
SHRINK = 0.5          # eu_half: weight on the European match's own pseudo-xG


def rps3(y: np.ndarray, P: np.ndarray) -> float:
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return float(((cp - cy) ** 2)[:, :2].sum(axis=1).mean() / 2)


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _form_mean(xr, team: str, default: float) -> float:
    """The team's current rolling xG-for level, for the shrunk arm."""
    h = xr._hist_for.get(str(team)) or []
    return float(np.mean(h[-5:])) if len(h) >= 3 else default


def apply_european(xr, row, arm: str, league_mean: float) -> None:
    """Append one European match to an arm's form state."""
    hx, ax = float(row.home_xg), float(row.away_xg)
    if arm == "eu_half":
        hm = _form_mean(xr, row.home_team, league_mean)
        am = _form_mean(xr, row.away_team, league_mean)
        hx = SHRINK * hx + (1 - SHRINK) * hm
        ax = SHRINK * ax + (1 - SHRINK) * am
    xr.update_form(row.home_team, row.away_team, hx, ax)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default=",".join(SEASONS))
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    df = pd.read_csv(FOUNDATION, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals", "home_xg", "away_xg", "date"])
    df = df.sort_values("date")

    eu_all = pseudo_xg(load_european_matches(ROOT), df)
    print(f"European matches on disk: {len(eu_all)} (goals -> xG factor {eu_all.attrs['k']:.3f})",
          flush=True)

    rows = []
    for s in seasons:
        test = df[df["season"] == s].sort_values("date")
        train = df[df["date"] < test["date"].min()]
        if len(test) == 0 or len(train) < 500:
            print(f"  {s}: skipped (no test rows or thin history)", flush=True)
            continue

        t0 = time.time()
        fitted = PredictionEngine(**DEPLOYED_CLUB_ENGINE_KWARGS).fit(
            train, squad_value_asof=test["date"].min())
        if fitted.xg_rate_model_ is None:
            print(f"  {s}: no xG-rate model, nothing to test", flush=True)
            continue
        league_mean = float(train["home_xg"].mean())

        teams = set(test["home_team"]) | set(test["away_team"])
        eu = big_five_rows(eu_all, teams)
        eu = eu[(eu["date"] >= test["date"].min() - pd.Timedelta(days=45))
                & (eu["date"] <= test["date"].max())].sort_values("date")
        clubs = (set(eu["home_team"]) | set(eu["away_team"])) & teams
        print(f"  {s}: fitted in {time.time()-t0:.0f}s | {len(test)} league matches, "
              f"{len(eu)} European ones over {len(clubs)} clubs", flush=True)

        engines = {arm: (fitted if arm == "base" else copy.deepcopy(fitted)) for arm in ARMS}
        # per arm: how far down the European table we have already consumed
        cursor = {arm: 0 for arm in ARMS}
        eu_rec = list(eu.itertuples(index=False))

        for r in test.itertuples(index=False):
            for arm in ARMS:
                eng = engines[arm]
                if arm != "base":
                    # everything played on a STRICTLY earlier date is known
                    i = cursor[arm]
                    while i < len(eu_rec) and eu_rec[i].date < r.date:
                        apply_european(eng.xg_rate_model_, eu_rec[i], arm, league_mean)
                        i += 1
                    cursor[arm] = i
                p = eng.predict_match(str(r.home_team), str(r.away_team),
                                      competition=str(r.competition),
                                      neutral=bool(getattr(r, "neutral", 0)))
                rows.append({"season": s, "arm": arm, "match_id": r.match_id,
                             "date": r.date, "hg": int(r.home_goals), "ag": int(r.away_goals),
                             "ph": p.p_home_win, "pd": p.p_draw, "pa": p.p_away_win,
                             "po25": p.p_over_25, "lh": p.lambda_home, "la": p.lambda_away})
            for arm in ARMS:   # league match: every arm learns it, as production does
                engines[arm].xg_rate_model_.update_form(
                    r.home_team, r.away_team, r.home_xg, r.away_xg)
        print(f"  {s}: done in {time.time()-t0:.0f}s", flush=True)

    m = pd.DataFrame(rows)
    m.to_csv(OUT, index=False)
    print(f"\nWROTE {OUT} ({len(m)} rows)\n", flush=True)
    report(m)


def report(m: pd.DataFrame) -> None:
    m = m.copy()
    m["y"] = np.where(m.hg > m.ag, 0, np.where(m.hg == m.ag, 1, 2))
    m["o25"] = ((m.hg + m.ag) > 2.5).astype(float)

    def score(g):
        P = g[["ph", "pd", "pa"]].to_numpy()
        return rps3(g["y"].to_numpy(), P), logloss(g["po25"].to_numpy(), g["o25"].to_numpy())

    print("1X2 RPS by season (lower is better)")
    header = f"{'season':12s}" + "".join(f"{a:>12s}" for a in ARMS)
    print(header)
    per = {}
    for s, g in m.groupby("season"):
        line, vals = f"{s:12s}", {}
        for a in ARMS:
            r, _ = score(g[g.arm == a])
            vals[a] = r
            line += f"{r:12.5f}"
        per[s] = vals
        print(line)
    print("-" * len(header))
    line, pooled = f"{'POOLED':12s}", {}
    for a in ARMS:
        r, ll = score(m[m.arm == a])
        pooled[a] = (r, ll)
        line += f"{r:12.5f}"
    print(line)

    base = pooled["base"][0]
    print("\nvs base (negative = better)")
    for a in ARMS[1:]:
        won = sum(1 for s in per if per[s][a] < per[s]["base"])
        print(f"  {a:8s} RPS {pooled[a][0]-base:+.5f}  ({won}/{len(per)} seasons)"
              f"   O2.5 logloss {pooled[a][1]-pooled['base'][1]:+.5f}"
              f" {'(worse)' if pooled[a][1] > pooled['base'][1] else '(not worse)'}")
    print("\nBar: RPS better in every season AND O/U not worse. Anything less is a negative"
          " result -- record it and stop.")


if __name__ == "__main__":
    main()
