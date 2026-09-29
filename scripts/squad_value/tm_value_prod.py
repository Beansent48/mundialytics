"""Production-faithful squad values for the backtest.

Mimics what production can do in 2026/27 with the frozen Kaggle dump + ESPN rosters:
  * values frozen at June 12 of the season's start year (dump ends 2026-06-12),
  * a player is only VISIBLE if, before that cutoff, he had a valuation/transfer while at a
    covered top-flight club (the dump scrapes top divisions only) -- a D2 lifer is invisible,
  * club membership is current (ESPN rosters in prod; latest transfer/valuation club here),
  * invisible squad members get a default value DEFAULT_EUR (roster size approximated by the
    club's player count, capped at 25).
"""
import sys

import numpy as np
import pandas as pd

ROOT = "C:/Users/Vicente/Desktop/BetBot/mundialytics_betting_engine/"
SP = "data/external/transfermarkt/work/"
TM = ROOT + "data/external/transfermarkt/"
DEFAULT_EUR = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0e6
COVERED = {"GB1", "ES1", "L1", "IT1", "FR1", "NL1", "PO1", "BE1", "TR1", "SC1", "GR1", "RU1", "UKR1", "DK1"}

cmap = pd.read_csv(SP + "tm_club_map.csv")
club2team = dict(zip(cmap.club_id, cmap.team))
v = pd.read_csv(TM + "player_valuations.csv")
v["date"] = pd.to_datetime(v.date)
t = pd.read_csv(TM + "transfers.csv", usecols=["player_id", "transfer_date", "to_club_id", "from_club_id"])
t["date"] = pd.to_datetime(t.transfer_date, errors="coerce")
t = t[t.date <= "2026-09-29"]
clubs = pd.read_csv(TM + "clubs.csv", usecols=["club_id", "domestic_competition_id"])
covered_clubs = set(clubs.loc[clubs.domestic_competition_id.isin(COVERED), "club_id"])
keep = set(v.loc[v.current_club_id.isin(club2team), "player_id"]) | set(t.loc[t.to_club_id.isin(club2team), "player_id"])
v = v[v.player_id.isin(keep)]
t = t[t.player_id.isin(keep)]
ev = pd.concat([v[["player_id", "date", "current_club_id"]].rename(columns={"current_club_id": "club"}),
                t[["player_id", "date", "to_club_id"]].rename(columns={"to_club_id": "club"})]).sort_values("date", kind="stable")
vv = v[["player_id", "date", "market_value_in_eur"]].sort_values("date")
# visibility: first date the player is valued while at a covered club
vis = v[v.player_club_domestic_competition_id.isin(COVERED)].groupby("player_id").date.min()

rows = []
for s in pd.date_range("2019-07-01", "2026-06-30", freq="W-MON"):
    vc = min(pd.Timestamp(year=(s.year if s.month >= 7 else s.year - 1), month=6, day=12), s)
    lv = vv[vv.date < vc].groupby("player_id").tail(1)
    lv = lv[lv.date >= vc - pd.Timedelta(days=550)]
    lc = ev[ev.date < s].groupby("player_id").tail(1)[["player_id", "club"]]
    x = lc.merge(lv, on="player_id", how="left")
    x = x[x.club.isin(club2team)]
    visible = x.player_id.map(vis) < vc
    x["val"] = np.where(visible & x.market_value_in_eur.notna(), x.market_value_in_eur, DEFAULT_EUR)
    # drop long-gone ghosts: a player with no valuation in 550 days AND invisible is noise
    x = x[visible | (x.player_id.map(vis).isna()) | x.market_value_in_eur.notna()]
    x["team"] = x.club.map(club2team)
    for team, g in x.groupby("team"):
        vals = np.sort(g.val.values)[::-1][:25]
        rows.append({"snap": s, "team": team, "v18": float(vals[:18].sum()),
                     "vis_share": float(visible[g.index].mean())})
out = pd.DataFrame(rows)
out.to_csv(SP + f"tm_squad_value_prod_{int(DEFAULT_EUR/1e5)}.csv", index=False)
print(out.shape, "mean visible share", round(out.vis_share.mean(), 3))
