"""Per team-match absence signals from Transfermarkt lineups, for the lineup-pass experiments.

Reads the Kaggle Transfermarkt dump (game_lineups + player_valuations, git-ignored) and, for
every Big Five league match since 2020/21, compares the team's regular XI (the production
definition: top-11 by starts in the last 10 league matches of the season, >=5 played) with
what actually happened that day. Every player value is the latest valuation strictly before
the match date, so nothing here looks ahead.

Columns written (one row per date x team):
    abs_cnt       share of regulars not starting (the deployed signal, on TM lineups)
    abs_val       value share of regulars not starting
    dlog_xi       log(value of today's XI) - log(value of the regular XI)
    unav_cnt/val  regulars not even in the matchday squad (injured / suspended / left out):
                  the part an injury + suspension feed could know in the morning
    bench_cnt/val regulars on the bench (rotation): only the lineup itself shows this

    python scripts/lineup_pass/build_tm_absence.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from mundialytics.features.lineup_absence import regular_xi  # noqa: E402

TM = "data/external/transfermarkt/"
OUT = TM + "work/tm_absence.csv"
LEAGUES = ["GB1", "ES1", "L1", "IT1", "FR1"]
DEFAULT_EUR = 1e6   # player with no valuation yet

cmap = pd.read_csv(TM + "work/tm_club_map.csv")
club2team = dict(zip(cmap.club_id, cmap.team))
g = pd.read_csv(TM + "games.csv", usecols=["game_id", "competition_id", "season", "date"])
g = g[g.competition_id.isin(LEAGUES) & (g.date >= "2020-07-01")]
lu = pd.read_csv(TM + "game_lineups.csv", usecols=["date", "game_id", "player_id", "club_id", "type"])
lu = lu[lu.game_id.isin(set(g.game_id))].copy()
lu["team"] = lu.club_id.map(club2team)
lu = lu.dropna(subset=["team"])
lu["date"] = pd.to_datetime(lu.date)
lu["season"] = lu.game_id.map(g.set_index("game_id").season)
lu["starter"] = lu.type == "starting_lineup"
lu = lu.rename(columns={"player_id": "player"})
print(f"lineup rows {len(lu):,}, games {lu.game_id.nunique():,}, starters/game-team "
      f"{lu.starter.sum() / lu.groupby(['game_id', 'team']).ngroups:.2f}", flush=True)

# regulars + today's squad per (date, team); values looked up in bulk afterwards
recs = []
for (season, team), grp in lu.groupby(["season", "team"]):
    hist = grp[["date", "player", "starter"]]
    for dt in sorted(grp.date.unique()):
        regs = regular_xi(hist, before=dt)
        if regs is None:
            continue
        today = grp[grp.date == dt]
        xi = set(today.loc[today.starter, "player"])
        squad = set(today.player)
        if len(xi) < 10:           # incomplete lineup record
            continue
        recs.append((pd.Timestamp(dt), team, regs, sorted(xi), squad))
print(f"team-matches with a regular XI: {len(recs):,}", flush=True)

q = pd.DataFrame([(dt, p) for dt, _, regs, xi, _ in recs for p in set(regs) | set(xi)],
                 columns=["date", "player"]).drop_duplicates()
v = pd.read_csv(TM + "player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur"])
v = v.rename(columns={"player_id": "player", "market_value_in_eur": "val"})
v["date"] = pd.to_datetime(v.date)
q = pd.merge_asof(q.sort_values("date"), v.sort_values("date"), on="date", by="player",
                  direction="backward", allow_exact_matches=False)
print(f"player-date values found: {q.val.notna().mean():.1%}", flush=True)
val = {(d, p): (x if np.isfinite(x) else DEFAULT_EUR) for d, p, x in zip(q.date, q.player, q.val)}

rows = []
for dt, team, regs, xi, squad in recs:
    rv = np.array([val[(dt, p)] for p in regs])
    xv = np.array([val[(dt, p)] for p in xi])
    out_xi = np.array([p not in xi for p in regs])
    unav = np.array([p not in squad for p in regs])
    bench = out_xi & ~unav
    rows.append({"date": dt, "team": team,
                 "abs_cnt": out_xi.mean(), "abs_val": rv[out_xi].sum() / rv.sum(),
                 "dlog_xi": np.log(xv.sum()) - np.log(rv.sum()),
                 "unav_cnt": unav.mean(), "unav_val": rv[unav].sum() / rv.sum(),
                 "bench_cnt": bench.mean(), "bench_val": rv[bench].sum() / rv.sum()})
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out.describe().round(3).to_string())
print(f"-> {OUT}")
