"""Current ESPN rosters x Transfermarkt player values.

Value = the player's latest valuation or transfer value in the (frozen) dump, falling back to
players.csv market_value_in_eur. Matching, most to least trusted:
  1. within the players whose TM club history includes this team: exact name, or a shared
     surname-ish token (len>=4) with birth year within +-3 (ESPN ages are sometimes wrong);
  2. global exact name, birth year +-2;
  3. global token subset, birth year +-1;
  4. global fuzzy ratio >= 0.88, birth year +-1.
"""
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

ROOT = "C:/Users/Vicente/Desktop/BetBot/mundialytics_betting_engine/"
SP = "data/external/transfermarkt/work/"
TM = ROOT + "data/external/transfermarkt/"


def norm(s):
    s = unicodedata.normalize("NFKD", str(s).replace("ß", "ss").replace("ı", "i").replace("ł", "l").replace("ø", "o")).encode("ascii", "ignore").decode().lower()
    s = s.replace("-", " ").replace("'", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", s)).strip()


r = pd.read_csv(ROOT + "data/external/advanced/espn/espn_team_rosters_current.csv")
p = pd.read_csv(TM + "players.csv", usecols=["player_id", "name", "date_of_birth", "market_value_in_eur"])
v = pd.read_csv(TM + "player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur", "current_club_id"])
t = pd.read_csv(TM + "transfers.csv", usecols=["player_id", "transfer_date", "to_club_id", "market_value_in_eur"])
t = t.rename(columns={"transfer_date": "date", "to_club_id": "club"})
t["date"] = pd.to_datetime(t.date, errors="coerce")
t = t[t.date <= "2026-09-29"]
v["date"] = pd.to_datetime(v.date)
val = pd.concat([v.rename(columns={"current_club_id": "club"}), t]).dropna(subset=["market_value_in_eur"])
val = val[val.market_value_in_eur > 0].sort_values("date")
last = val.groupby("player_id").tail(1).set_index("player_id")
p["value"] = p.player_id.map(last.market_value_in_eur).fillna(p.market_value_in_eur)
p["value_date"] = p.player_id.map(last.date)
p = p[p.value > 0].copy()
p["by"] = pd.to_datetime(p.date_of_birth, errors="coerce").dt.year
p["k"] = p.name.map(norm)
p["toks"] = p.k.str.split().map(set)
P = p.set_index("player_id")

cmap = pd.read_csv(SP + "tm_club_map.csv")
club2team = dict(zip(cmap.club_id, cmap.team))
ev = pd.concat([v[["player_id", "current_club_id"]].rename(columns={"current_club_id": "club"}), t[["player_id", "club"]]])
ev["team"] = ev.club.map(club2team)
team_pool = ev.dropna(subset=["team"]).groupby("team").player_id.agg(lambda s: set(s) & set(P.index))
by_name = defaultdict(list)
for pid, k in zip(p.player_id, p.k):
    by_name[k].append(pid)
by_year = {y: g for y, g in p.groupby("by")}


def yr_ok(pid, by, tol):
    b = P.at[pid, "by"]
    return not (np.isfinite(by) and np.isfinite(b)) or abs(b - by) <= tol


def match(row):
    toks = {x for x in row.k.split() if len(x) >= 4} or set(row.k.split())
    # 1. own club history
    best = []
    for pid in team_pool.get(row.team, ()):
        ck, ct = P.at[pid, "k"], P.at[pid, "toks"]
        et = row.k.split()
        cl = ck.split()
        if ck == row.k:
            best.append((3, pid))
        elif not yr_ok(pid, row.by, 3):
            continue
        elif set(et) == ct or len(set(et) & ct) >= 2:
            best.append((2, pid))   # same tokens, any order (Kim Min-Jae / Min-jae Kim)
        elif ((len(et) > 1 and len(et[-1]) >= 3 and et[-1] in ct) or (len(cl) > 1 and len(cl[-1]) >= 3 and cl[-1] in et))                 and et[0][0] == cl[0][0]:
            best.append((2, pid))   # surname match + same first initial
        elif (len(cl) == 1 and cl[0] in et and yr_ok(pid, row.by, 1)) or (len(et) == 1 and et[0] in ct and yr_ok(pid, row.by, 1)):
            best.append((2, pid))   # mononym (Alisson, Gabriel, Rodri)
        elif SequenceMatcher(None, row.k, ck).ratio() >= 0.8:
            best.append((1, pid))
    if best:
        best.sort(key=lambda x: (x[0], P.at[x[1], "value_date"] if pd.notna(P.at[x[1], "value_date"]) else pd.Timestamp(0)))
        return best[-1][1], "club"
    # 2. global exact
    c = [pid for pid in by_name.get(row.k, []) if yr_ok(pid, row.by, 2)]
    if len(c) == 1:
        return c[0], "exact"
    if not np.isfinite(row.by):
        return None, "none"
    cands = pd.concat([by_year[y] for y in (row.by - 1, row.by, row.by + 1) if y in by_year])
    allt = set(row.k.split())
    sub = cands[cands.toks.map(lambda s: s <= allt or allt <= s)]
    if len(sub) == 1:
        return sub.player_id.iloc[0], "subset"
    rat = cands.k.map(lambda k: SequenceMatcher(None, row.k, k).ratio())
    if len(rat) and rat.max() >= 0.88 and (rat >= 0.88).sum() == 1:
        return cands.player_id[rat.idxmax()], "fuzzy"
    return None, "none"


r["k"] = r.player.map(norm)
r["by"] = 2026 - r.age
res = [match(row) for row in r.itertuples()]
r["player_id"] = [x[0] for x in res]
r["how"] = [x[1] for x in res]
r["value"] = r.player_id.map(P.value)
r["value_date"] = r.player_id.map(P.value_date)
r["tm_name"] = r.player_id.map(P.name)
r.to_csv(SP + "roster_values.csv", index=False)
r["ok"] = r.player_id.notna()
print(r.how.value_counts().to_dict())
cov = r.groupby(["competition", "team"]).agg(n=("ok", "size"), matched=("ok", "mean"),
                                            v18=("value", lambda s: s.nlargest(18).sum())).reset_index()
print(cov.groupby("competition").matched.mean().round(3).to_dict(), "overall", round(r.ok.mean(), 3))
promo = ['coventry', 'elversberg', 'frosinone', 'hull', 'ipswich', 'la coruna', 'le mans', 'malaga', 'monza',
         'paderborn', 'santander', 'schalke 04', 'troyes', 'venezia']
print(cov[cov.team.isin(promo)].sort_values("matched").round(2).to_string())
# sanity: suspicious matches (fuzzy/subset) sample
print(r[r.how.isin(["fuzzy", "subset"])][["team", "player", "tm_name", "age"]].sample(12, random_state=1).to_string())
z = r[r.how == "club"].loc[lambda x: x.k != x.tm_name.map(norm)]
print(len(z), "non-exact club matches"); print(z[["team", "player", "tm_name", "age"]].sample(min(40, len(z)), random_state=0).to_string())
