"""Map Transfermarkt club_id -> canonical team name by joining TM games to our matches
on (date, competition, score)."""
import sys

import pandas as pd

ROOT = "C:/Users/Vicente/Desktop/BetBot/mundialytics_betting_engine/"
sys.path.insert(0, ROOT + "src")
from mundialytics.statistical_core.schemas import canonical_name  # noqa

SP = "data/external/transfermarkt/work/"
TM = ROOT + "data/external/transfermarkt/"
COMP = {"GB1": "Premier League", "ES1": "LaLiga", "L1": "Bundesliga", "IT1": "Serie A", "FR1": "Ligue 1"}
g = pd.read_csv(TM + "games.csv", usecols=["game_id", "competition_id", "date", "home_club_id", "away_club_id",
                                            "home_club_goals", "away_club_goals", "home_club_name", "away_club_name"])
g = g[g.competition_id.isin(COMP)].copy()
g["competition"] = g.competition_id.map(COMP)
g["date"] = pd.to_datetime(g.date)
m = pd.read_csv(ROOT + "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv", low_memory=False,
                usecols=["date", "home_team", "away_team", "home_goals", "away_goals", "competition"])
m["date"] = pd.to_datetime(m.date)
m["home_team"] = m.home_team.map(canonical_name)
m["away_team"] = m.away_team.map(canonical_name)
j = g.merge(m, left_on=["date", "competition", "home_club_goals", "away_club_goals"],
            right_on=["date", "competition", "home_goals", "away_goals"])
pairs = pd.concat([j[["home_club_id", "home_team", "home_club_name"]].set_axis(["club_id", "team", "tm_name"], axis=1),
                   j[["away_club_id", "away_team", "away_club_name"]].set_axis(["club_id", "team", "tm_name"], axis=1)])
cnt = pairs.groupby(["club_id", "team"]).size().rename("n").reset_index()
tot = cnt.groupby("club_id").n.transform("sum")
cnt["share"] = cnt.n / tot
best = cnt.sort_values("n", ascending=False).drop_duplicates("club_id")
best = best[(best.share > 0.5) & (best.n >= 10)]
names = pairs.drop_duplicates("club_id").set_index("club_id").tm_name
best["tm_name"] = best.club_id.map(names)
# a team must map to one club
best = best.sort_values("n", ascending=False).drop_duplicates("team")
best.to_csv(SP + "tm_club_map.csv", index=False)
ours = set(m[m.date >= "2014-08-01"].home_team)
print("mapped clubs", len(best), "| our teams since 2014:", len(ours), "| unmapped:", sorted(ours - set(best.team)))
