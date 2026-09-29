"""Suspensions predicted from league cards: the part of team news known in the MORNING.

The lineup pass only sees absences an hour before kick-off. Bans are deterministic from the
card record, so they are known days ahead. This replays each league's rules over the
Transfermarkt card events (league matches only, which is what the league bans count):

    red card or second yellow   -> next league match
    GB1  5 yellows by club match 19 -> 1, 10 by match 32 -> 2, 15 -> 3
    ES1  every 5 yellows -> 1          L1  5, 10, 15 -> 1 each
    IT1  5, 10, 14, 17, then each -> 1  FR1  3 yellows within the last 10 matches -> 1

The yellow that preceded a second yellow does not count. Cup and European cards are ignored
(they carry separate bans in most leagues), so this is a lower bound on who is banned.

Checked against what happened: a banned player should not be in the matchday squad.
Output (one row per date x team with a regular XI): susp_n, susp_cnt (share of regulars
banned), susp_val (their value share); inj_cnt / inj_val = regulars missing from the previous
match's squad without a ban there (probably still injured, also known in the morning);
morning_val = susp_val + inj_val.

    python scripts/lineup_pass/build_suspensions.py
"""
import sys
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from mundialytics.features.lineup_absence import regular_xi  # noqa: E402

TM = "data/external/transfermarkt/"
OUT = TM + "work/suspensions.csv"
LEAGUES = ["GB1", "ES1", "L1", "IT1", "FR1"]

cmap = pd.read_csv(TM + "work/tm_club_map.csv")
club2team = dict(zip(cmap.club_id, cmap.team))
g = pd.read_csv(TM + "games.csv", usecols=["game_id", "competition_id", "season", "date",
                                            "home_club_id", "away_club_id"])
g = g[g.competition_id.isin(LEAGUES) & (g.date >= "2020-07-01")].copy()
g["date"] = pd.to_datetime(g.date)
z = zipfile.ZipFile(TM + "archive.zip")
ev = pd.read_csv(z.open("game_events.csv"), usecols=["game_id", "type", "club_id", "player_id", "description"])
ev = ev[(ev.type == "Cards") & ev.game_id.isin(set(g.game_id))].copy()
desc = ev.description.fillna("")
ev["kind"] = np.where(desc.str.contains("Second yellow"), "2y",
                      np.where(desc.str.contains("Red card"), "red",
                               np.where(desc.str.contains("Yellow card"), "y", "other")))
second = set(zip(ev.loc[ev.kind == "2y", "game_id"], ev.loc[ev.kind == "2y", "player_id"]))
ev = ev[~((ev.kind == "y").values & np.array([k in second for k in zip(ev.game_id, ev.player_id)]))]
print(ev.kind.value_counts().to_string(), flush=True)

lu = pd.read_csv(TM + "game_lineups.csv", usecols=["game_id", "player_id", "club_id", "type"])
lu = lu[lu.game_id.isin(set(g.game_id))]
squad = lu.groupby(["game_id", "club_id"]).player_id.apply(set).to_dict()
xi = lu[lu.type == "starting_lineup"].groupby(["game_id", "club_id"]).player_id.apply(set).to_dict()

# team match sequence per season
tg = pd.concat([g.assign(club=g.home_club_id), g.assign(club=g.away_club_id)])
tg = tg.sort_values("date")
cards = ev.groupby(["game_id", "club_id"])


def yellow_bans(comp, n_yellow, club_match_no, recent):
    """Matches banned on receiving yellow number n_yellow."""
    if comp == "GB1":
        if n_yellow == 5 and club_match_no <= 19:
            return 1
        if n_yellow == 10 and club_match_no <= 32:
            return 2
        return 3 if n_yellow == 15 else 0
    if comp == "ES1":
        return 1 if n_yellow % 5 == 0 else 0
    if comp == "L1":
        return 1 if n_yellow in (5, 10, 15) else 0
    if comp == "IT1":
        return 1 if n_yellow in (5, 10, 14, 17) or n_yellow >= 19 else 0
    if comp == "FR1":
        return 1 if recent >= 3 else 0
    return 0


banned = {}          # (game_id, club) -> set of players banned for that game
checks = []
for (season, club), seq in tg.groupby(["season", "club"]):
    comp = seq.competition_id.iloc[0]
    yel = defaultdict(int)
    fr_window = defaultdict(list)     # FR1: match numbers of counted yellows
    owed = defaultdict(int)           # player -> matches still to serve
    for i, r in enumerate(seq.itertuples(), start=1):
        today = {p for p, k in owed.items() if k > 0}
        banned[(r.game_id, club)] = today
        sq = squad.get((r.game_id, club))
        if sq:
            checks += [p not in sq for p in today]
        for p in today:
            owed[p] -= 1
        try:
            c = cards.get_group((r.game_id, club))
        except KeyError:
            continue
        for p, k in zip(c.player_id, c.kind):
            if k in ("red", "2y"):
                owed[p] += 1
            elif k == "y":
                yel[p] += 1
                fr_window[p] = [m for m in fr_window[p] if m > i - 10] + [i]
                n = yellow_bans(comp, yel[p], i, len(fr_window[p]))
                if n:
                    owed[p] += n
                    fr_window[p] = []
checks = np.array(checks)
print(f"predicted bans {len(checks):,}; banned player NOT in the matchday squad: {checks.mean():.1%}", flush=True)

# per team-match share of the regular XI banned
lu2 = lu.merge(g[["game_id", "date", "season"]], on="game_id")
lu2["team"] = lu2.club_id.map(club2team)
lu2 = lu2.dropna(subset=["team"]).rename(columns={"player_id": "player"})
lu2["starter"] = lu2.type == "starting_lineup"
v = pd.read_csv(TM + "player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur"])
v = v.rename(columns={"player_id": "player", "market_value_in_eur": "val"})
v["date"] = pd.to_datetime(v.date)
rows = []
for (season, club), grp in lu2.groupby(["season", "club_id"]):
    hist = grp[["date", "player", "starter"]]
    prev = None                      # previous league match: (game_id, squad)
    prev2 = None                     # the one before
    for gid, dt in grp[["game_id", "date"]].drop_duplicates().sort_values("date").itertuples(index=False):
        regs = regular_xi(hist, before=dt)
        b = banned.get((gid, club), set())
        if regs is not None:
            # out of last match's squad and not serving a ban there: most likely still injured
            pb = banned.get((prev[0], club), set()) if prev else set()
            inj = [p for p in regs if prev and p not in prev[1] and p not in pb and p not in b]
            inj_nb = [p for p in regs if prev and p not in prev[1]]
            inj2 = [p for p in regs if prev and prev2 and p not in prev[1] and p not in prev2[1]]
            rows.append({"date": dt, "team": grp.team.iloc[0], "regs": regs,
                         "susp": [p for p in regs if p in b], "inj": inj, "inj_nb": inj_nb, "inj2": inj2,
                         "unav": [p for p in regs if p not in squad.get((gid, club), set())]})
        prev2, prev = prev, (gid, squad.get((gid, club), set()))
out = pd.DataFrame(rows)
q = out[["date", "regs"]].explode("regs").rename(columns={"regs": "player"}).drop_duplicates()
q["player"] = q.player.astype(int)
q = pd.merge_asof(q.sort_values("date"), v.sort_values("date"), on="date", by="player",
                  direction="backward", allow_exact_matches=False)
val = {(a, b): (c if np.isfinite(c) else 1e6) for a, b, c in zip(q.date, q.player, q.val)}
out["susp_n"] = out.susp.str.len()
out["susp_cnt"] = out.susp_n / out.regs.str.len()
out["susp_val"] = [sum(val[(dt, p)] for p in s) / sum(val[(dt, p)] for p in r)
                   for dt, r, s in zip(out.date, out.regs, out.susp)]
out["inj_cnt"] = out.inj.str.len() / out.regs.str.len()
out["inj_val"] = [sum(val[(dt, p)] for p in s) / sum(val[(dt, p)] for p in r)
                  for dt, r, s in zip(out.date, out.regs, out.inj)]
out["morning_val"] = out.susp_val + out.inj_val
out["inj_nb_cnt"] = out.inj_nb.str.len() / out.regs.str.len()
out["inj2_cnt"] = out.inj2.str.len() / out.regs.str.len()
hit = [len(set(i) & set(u)) for i, u in zip(out.inj, out.unav)]
print(f"'still injured' regulars {out.inj.str.len().sum():,}: absent from today's squad {sum(hit) / max(1, out.inj.str.len().sum()):.1%}; "
      f"they cover {sum(hit) / max(1, out.unav.str.len().sum()):.1%} of today's unavailable regulars")
out = out.drop(columns=["regs", "susp", "inj", "inj_nb", "inj2", "unav"])
out.to_csv(OUT, index=False)
print(f"team-matches {len(out):,}; with >=1 regular banned {(out.susp_n > 0).mean():.1%}; -> {OUT}")
