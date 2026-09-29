"""ClubElo rating of each side the day before every harness match, for the Elo-blend test.

ClubElo's per-club history files (data/external/clubelo/teams/, 220 clubs) list rating
periods From..To. The rating valid the day BEFORE the match is used, so the match's own
result can never leak in. Two versions:

    elo_live    rating the day before the match (moves with league, cup and European results)
    elo_frozen  rating on 1 August of the season: what production has while the ClubElo API
                is down (the local roll-forward only moves with Big Five league results)

    python scripts/elo_blend/build_clubelo_pre.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from mundialytics.statistical_core.competition.european import make_resolver  # noqa: E402

WORK = "data/external/transfermarkt/work/"
d = pd.read_pickle(WORK + "wfr.pkl")
d["date"] = pd.to_datetime(d.date)

hist = pd.concat([pd.read_csv(f, usecols=["Club", "Elo", "From", "To"])
                  for f in sorted(Path("data/external/clubelo/teams").glob("*.csv"))])
hist["From"] = pd.to_datetime(hist.From)
hist["To"] = pd.to_datetime(hist.To)
hist = hist.dropna().sort_values("From")
resolve = make_resolver(sorted(hist.Club.unique()))
teams = sorted(set(d.home) | set(d.away))
t2c = {t: resolve(t) for t in teams}
miss = [t for t, c in t2c.items() if c is None]
print(f"teams {len(teams)}, resolved {len(teams) - len(miss)}; missing: {miss}")
for t, c in sorted(t2c.items()):
    if c is not None and c.lower().replace(" ", "") not in t.replace(" ", ""):
        print(f"   check alias: {t!r} -> {c!r}")

by_club = {c: g for c, g in hist.groupby("Club")}


def rating(club, when):
    g = by_club.get(club)
    if g is None:
        return np.nan
    i = g.From.searchsorted(when, side="right") - 1
    if i < 0 or g.To.iloc[i] < when - pd.Timedelta(days=60):
        return np.nan          # no period covering (or near) that day
    return float(g.Elo.iloc[i])


rows = []
for dt, team, season in pd.concat([d[["date", "home", "season"]].rename(columns={"home": "team"}),
                                   d[["date", "away", "season"]].rename(columns={"away": "team"})]
                                  ).drop_duplicates().itertuples(index=False):
    c = t2c.get(team)
    aug1 = pd.Timestamp(f"{season[:4]}-08-01")
    rows.append({"date": dt, "team": team,
                 "elo_live": rating(c, dt - pd.Timedelta(days=1)),
                 "elo_frozen": rating(c, aug1)})
out = pd.DataFrame(rows)
print(out[["elo_live", "elo_frozen"]].notna().mean().round(4).to_string())
out[["elo_live", "elo_frozen"]] = out[["elo_live", "elo_frozen"]] / 400.0
out.to_csv(WORK + "clubelo_pre.csv", index=False)
print(f"-> {WORK}clubelo_pre.csv  ({len(out):,} team-dates)")
