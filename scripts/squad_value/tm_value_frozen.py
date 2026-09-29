"""Weekly squad-value snapshots per canonical team: each player's latest valuation strictly
before the snapshot, assigned to the club of his latest event (transfer or valuation) before it.
Squad value = sum of the top-N player values at the club."""
import numpy as np
import pandas as pd

ROOT = "C:/Users/Vicente/Desktop/BetBot/mundialytics_betting_engine/"
SP = "data/external/transfermarkt/work/"
TM = ROOT + "data/external/transfermarkt/"
cmap = pd.read_csv(SP + "tm_club_map.csv")
club2team = dict(zip(cmap.club_id, cmap.team))
v = pd.read_csv(TM + "player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur", "current_club_id"])
v["date"] = pd.to_datetime(v.date)
t = pd.read_csv(TM + "transfers.csv", usecols=["player_id", "transfer_date", "to_club_id"])
t["date"] = pd.to_datetime(t.transfer_date, errors="coerce")
t = t[t.date <= "2026-09-29"]
keep = set(v.loc[v.current_club_id.isin(club2team), "player_id"]) | set(t.loc[t.to_club_id.isin(club2team), "player_id"])
v = v[v.player_id.isin(keep) & (v.date >= "2011-01-01")]
t = t[t.player_id.isin(keep) & (t.date >= "2011-01-01")]
ev = pd.concat([v[["player_id", "date", "current_club_id"]].rename(columns={"current_club_id": "club"}),
                t[["player_id", "date", "to_club_id"]].rename(columns={"to_club_id": "club"})]).sort_values(["date"], kind="stable")
vv = v[["player_id", "date", "market_value_in_eur"]].sort_values("date")
print("players", len(keep), "valuations", len(vv), "club events", len(ev))

snaps = pd.date_range("2019-07-01", "2026-06-30", freq="W-MON")
TOPN = (11, 18, 25)
rows = []
for s in snaps:
    vc = pd.Timestamp(year=(s.year if s.month >= 7 else s.year - 1), month=6, day=12)
    vc = min(vc, s)
    lv = vv[vv.date < vc].groupby("player_id").tail(1)
    lv = lv[lv.date >= vc - pd.Timedelta(days=550)]  # drop stale (retired / out of top football)
    lc = ev[ev.date < s].groupby("player_id").tail(1)[["player_id", "club"]]
    x = lv.merge(lc, on="player_id")
    x = x[x.club.isin(club2team)]
    x["team"] = x.club.map(club2team)
    x = x.sort_values("market_value_in_eur", ascending=False)
    for team, grp in x.groupby("team"):
        vals = grp.market_value_in_eur.values
        r = {"snap": s, "team": team, "n_players": len(vals)}
        for n in TOPN:
            r[f"v{n}"] = float(vals[:n].sum())
        rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(SP + "tm_squad_value_frozen.csv", index=False)
print(out.shape)
