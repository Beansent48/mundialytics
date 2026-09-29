"""Production-faithful squad values for the backtest.

Mimics what production can do with the frozen Kaggle dump + current ESPN rosters:
  * values frozen at June 12 of the season's start year (the dump ends 2026-06-12);
  * the dump scrapes clubs of top divisions only, so a player is VISIBLE at a date only if,
    before it, he was at a club that was playing a covered top-flight league THAT season
    (point-in-time from games.csv -- the competition column on player_valuations is the
    club's current league, not the league at valuation time);
  * a visible player keeps his latest valuation before the cutoff (whatever league he is
    in now: the value does not change when he moves);
  * invisible squad members, and a random MATCH_LOSS share of visible ones (roster names
    that production fails to match), get DEFAULT_EUR;
  * club membership = latest transfer / valuation club before the snapshot (ESPN rosters in
    production), dropping ghosts with no event in the last 550 days.

usage: python scripts/squad_value/tm_value_prod.py DEFAULT_EUR MATCH_LOSS
"""
import sys

import numpy as np
import pandas as pd

ROOT = "C:/Users/Vicente/Desktop/BetBot/mundialytics_betting_engine/"
SP = "data/external/transfermarkt/work/"
TM = ROOT + "data/external/transfermarkt/"
DEFAULT_EUR = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0e6
MATCH_LOSS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
COVERED = {"GB1", "ES1", "L1", "IT1", "FR1", "NL1", "PO1", "BE1", "TR1", "SC1", "GR1", "RU1", "UKR1", "DK1"}
rng = np.random.default_rng(7)


def season_of(dates: pd.Series) -> pd.Series:
    return np.where(dates.dt.month >= 7, dates.dt.year, dates.dt.year - 1)


cmap = pd.read_csv(SP + "tm_club_map.csv")
club2team = dict(zip(cmap.club_id, cmap.team))
g = pd.read_csv(TM + "games.csv", usecols=["competition_id", "season", "home_club_id", "away_club_id"])
g = g[g.competition_id.isin(COVERED)]
covered = set(zip(g.home_club_id, g.season)) | set(zip(g.away_club_id, g.season))

v = pd.read_csv(TM + "player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur", "current_club_id"])
v["date"] = pd.to_datetime(v.date)
t = pd.read_csv(TM + "transfers.csv", usecols=["player_id", "transfer_date", "to_club_id"])
t["date"] = pd.to_datetime(t.transfer_date, errors="coerce")
t = t[t.date <= "2026-09-29"].dropna(subset=["date"])
keep = set(v.loc[v.current_club_id.isin(club2team), "player_id"]) | set(t.loc[t.to_club_id.isin(club2team), "player_id"])
v = v[v.player_id.isin(keep)]
t = t[t.player_id.isin(keep)]

# point-in-time visibility: first dated event at a club that was in a covered league that season
v["cov"] = [(c, s) in covered for c, s in zip(v.current_club_id, season_of(v.date))]
t["cov"] = [(c, s) in covered for c, s in zip(t.to_club_id, season_of(t.date))]
vis = pd.concat([v.loc[v["cov"], ["player_id", "date"]], t.loc[t["cov"], ["player_id", "date"]]]).groupby("player_id").date.min()

ev = pd.concat([v[["player_id", "date", "current_club_id"]].rename(columns={"current_club_id": "club"}),
                t[["player_id", "date", "to_club_id"]].rename(columns={"to_club_id": "club"})]).sort_values("date", kind="stable")
vv = v[["player_id", "date", "market_value_in_eur"]].sort_values("date")
lost = set(rng.choice(sorted(keep), size=int(MATCH_LOSS * len(keep)), replace=False)) if MATCH_LOSS > 0 else set()

rows = []
for s in pd.date_range("2019-07-01", "2026-06-30", freq="W-MON"):
    vc = min(pd.Timestamp(year=(s.year if s.month >= 7 else s.year - 1), month=6, day=12), s)
    lv = vv[vv.date < vc].groupby("player_id").tail(1).set_index("player_id").market_value_in_eur
    le = ev[ev.date < s].groupby("player_id").tail(1)
    le = le[le.date >= s - pd.Timedelta(days=550)]
    x = le[le.club.isin(club2team)].copy()
    visible = (x.player_id.map(vis) < vc) & ~x.player_id.isin(lost)
    val = x.player_id.map(lv)
    x["val"] = np.where(visible & val.notna(), val, DEFAULT_EUR)
    x["visible"] = visible.values
    x["team"] = x.club.map(club2team)
    for team, grp in x.groupby("team"):
        vals = np.sort(grp.val.values)[::-1]
        rows.append({"snap": s, "team": team, "v18": float(vals[:18].sum()), "vis_share": float(grp.visible.mean()),
                     "n": len(grp)})
out = pd.DataFrame(rows)
name = SP + f"tm_squad_value_prod_d{DEFAULT_EUR/1e6:g}_l{MATCH_LOSS:g}.csv"
out.to_csv(name, index=False)
print(name, out.shape, "mean visible share", round(out.vis_share.mean(), 3))
