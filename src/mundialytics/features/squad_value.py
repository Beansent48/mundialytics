"""Squad market value per club, dated, for the match engine.

Why: the engine rates teams from results and xG only, so it cannot tell a promoted club
that spent the summer rebuilding (Como 24/25, Paris FC 25/26) from one that did not, and it
learns about summer signings only after they play. Against Bet365 our gap is 50% larger in
matches with a promoted side, and there it is differentiation, not a mean bias. Squad value
is the missing signal (scripts/squad_value/README.md).

Source: the Kaggle Transfermarkt dump (davidcariboo/player-scores, CC0) in
data/external/transfermarkt/. It is FROZEN (valuations end 2026-06-12) and scrapes
top-division clubs only. A player's value does not change when he moves, so the current
squad is the ESPN roster and each player carries his latest dump value. Players the dump
never saw (second-division lifers) get DEFAULT_EUR.

Output: data/processed/squad_values.csv with columns snap, team, v18, n, known_share. v18
is the sum of the 18 most valuable players. Rows are dated; read them with
`squad_values_asof` so a fit at a past cutoff never sees later squads.

History snapshots (weekly, 2019-07 to the dump's end) are built the same way the backtest
validated. Values are frozen at June 12 of the season's start year, and a player is known
only if, before that date, he was at a club playing a covered top flight THAT season. That
is point-in-time from games.csv: the league column on the valuations is the club's current
league, not the league at valuation time.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from mundialytics.statistical_core.schemas import canonical_name

ROOT = Path(__file__).resolve().parents[3]
TM_DIR = ROOT / "data/external/transfermarkt"
SQUAD_VALUES_PATH = ROOT / "data/processed/squad_values.csv"
CLUB_MAP_PATH = ROOT / "data/processed/tm_club_map.csv"
MATCH_CACHE_PATH = ROOT / "data/processed/squad_value_player_matches.csv"
ROSTERS_PATH = ROOT / "data/external/advanced/espn/espn_team_rosters_current.csv"
FOUNDATION_PATH = ROOT / "data/processed/foundation_big5_multi_season.csv"

TOP_N = 18
DEFAULT_EUR = 1.0e6
MAX_AGE_DAYS = 120      # a team's latest snapshot older than this is ignored
TM_LEAGUES = {"GB1": "Premier League", "ES1": "LaLiga", "L1": "Bundesliga",
              "IT1": "Serie A", "FR1": "Ligue 1"}
# top flights the dump scrapes: being at one of these clubs makes a player "known"
COVERED = {"GB1", "ES1", "L1", "IT1", "FR1", "NL1", "PO1", "BE1", "TR1", "SC1",
           "GR1", "RU1", "UKR1", "DK1"}


# ── reading ────────────────────────────────────────────────────────────────────

def load_squad_values(path: str | Path | None = None) -> pd.DataFrame | None:
    p = Path(path) if path else SQUAD_VALUES_PATH
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["snap"])
    return df if not df.empty else None


def squad_values_asof(df: pd.DataFrame | None, asof=None,
                      max_age_days: int = MAX_AGE_DAYS) -> dict[str, float]:
    """{team: v18} from each team's latest snapshot on or before `asof` (None = latest)."""
    if df is None or df.empty:
        return {}
    d = df
    if asof is not None:
        d = d[d["snap"] <= pd.Timestamp(asof)]
    if d.empty:
        return {}
    last = d.sort_values("snap").groupby("team").tail(1)
    ref = pd.Timestamp(asof) if asof is not None else d["snap"].max()
    last = last[last["snap"] >= ref - pd.Timedelta(days=max_age_days)]
    return {canonical_name(t): float(v) for t, v in zip(last["team"], last["v18"]) if v > 0}


# ── building ───────────────────────────────────────────────────────────────────

def _season_start(dates: pd.Series) -> np.ndarray:
    return np.where(dates.dt.month >= 7, dates.dt.year, dates.dt.year - 1)


def build_club_map(tm_dir: Path = TM_DIR, foundation: Path = FOUNDATION_PATH) -> pd.DataFrame:
    """Transfermarkt club_id -> canonical team, by joining TM games to ours on
    (date, league, score). Only unambiguous pairs (>50% of a club's games, >=10 games)."""
    g = pd.read_csv(tm_dir / "games.csv", usecols=[
        "competition_id", "date", "home_club_id", "away_club_id", "home_club_goals",
        "away_club_goals", "home_club_name", "away_club_name"])
    g = g[g["competition_id"].isin(TM_LEAGUES)].copy()
    g["competition"] = g["competition_id"].map(TM_LEAGUES)
    g["date"] = pd.to_datetime(g["date"])
    m = pd.read_csv(foundation, low_memory=False,
                    usecols=["date", "home_team", "away_team", "home_goals", "away_goals", "competition"])
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m["home_team"] = m["home_team"].map(canonical_name)
    m["away_team"] = m["away_team"].map(canonical_name)
    j = g.merge(m, left_on=["date", "competition", "home_club_goals", "away_club_goals"],
                right_on=["date", "competition", "home_goals", "away_goals"])
    pairs = pd.concat([
        j[["home_club_id", "home_team", "home_club_name"]].set_axis(["club_id", "team", "tm_name"], axis=1),
        j[["away_club_id", "away_team", "away_club_name"]].set_axis(["club_id", "team", "tm_name"], axis=1)])
    cnt = pairs.groupby(["club_id", "team"]).size().rename("n").reset_index()
    cnt["share"] = cnt["n"] / cnt.groupby("club_id")["n"].transform("sum")
    best = cnt.sort_values("n", ascending=False).drop_duplicates("club_id")
    best = best[(best["share"] > 0.5) & (best["n"] >= 10)]
    best["tm_name"] = best["club_id"].map(pairs.drop_duplicates("club_id").set_index("club_id")["tm_name"])
    return best.sort_values("n", ascending=False).drop_duplicates("team").reset_index(drop=True)


def _load_dump(tm_dir: Path):
    v = pd.read_csv(tm_dir / "player_valuations.csv",
                    usecols=["player_id", "date", "market_value_in_eur", "current_club_id"])
    v["date"] = pd.to_datetime(v["date"])
    t = pd.read_csv(tm_dir / "transfers.csv",
                    usecols=["player_id", "transfer_date", "to_club_id", "market_value_in_eur"])
    t["date"] = pd.to_datetime(t["transfer_date"], errors="coerce")
    # the dump lists some contract-end dates years ahead as "transfers"
    t = t[t["date"] <= pd.Timestamp.now()].dropna(subset=["date"])
    return v, t


def build_history(tm_dir: Path = TM_DIR, club_map: pd.DataFrame | None = None,
                  start: str = "2019-07-01", end: str | None = None,
                  default_eur: float = DEFAULT_EUR, freeze_june: bool = False) -> pd.DataFrame:
    """Weekly snapshots, production-faithful (see module docstring).

    freeze_june=False (the default since the live Transfermarkt source): each snapshot
    reads valuations up to its own date, as a weekly live refresh would. True rebuilds
    the June-frozen history of the first deployment (values as of June 12 all season).
    """
    club_map = club_map if club_map is not None else build_club_map(tm_dir)
    club2team = dict(zip(club_map["club_id"], club_map["team"]))
    g = pd.read_csv(tm_dir / "games.csv", usecols=["competition_id", "season", "home_club_id", "away_club_id"])
    g = g[g["competition_id"].isin(COVERED)]
    covered = set(zip(g["home_club_id"], g["season"])) | set(zip(g["away_club_id"], g["season"]))
    v, t = _load_dump(tm_dir)
    keep = (set(v.loc[v["current_club_id"].isin(club2team), "player_id"])
            | set(t.loc[t["to_club_id"].isin(club2team), "player_id"]))
    v = v[v["player_id"].isin(keep)].copy()
    t = t[t["player_id"].isin(keep)].copy()
    v["cov"] = [(c, s) in covered for c, s in zip(v["current_club_id"], _season_start(v["date"]))]
    t["cov"] = [(c, s) in covered for c, s in zip(t["to_club_id"], _season_start(t["date"]))]
    known_from = pd.concat([v.loc[v["cov"], ["player_id", "date"]],
                            t.loc[t["cov"], ["player_id", "date"]]]).groupby("player_id")["date"].min()
    ev = pd.concat([
        v[["player_id", "date", "current_club_id"]].rename(columns={"current_club_id": "club"}),
        t[["player_id", "date", "to_club_id"]].rename(columns={"to_club_id": "club"}),
    ]).sort_values("date", kind="stable")
    vv = v[["player_id", "date", "market_value_in_eur"]].sort_values("date")
    end = pd.Timestamp(end) if end else max(ev["date"].max(), vv["date"].max())
    rows = []
    for s in pd.date_range(start, end, freq="W-MON"):
        vc = (min(pd.Timestamp(year=(s.year if s.month >= 7 else s.year - 1), month=6, day=12), s)
              if freeze_june else s)
        lv = vv[vv["date"] < vc].groupby("player_id").tail(1).set_index("player_id")["market_value_in_eur"]
        le = ev[ev["date"] < s].groupby("player_id").tail(1)
        le = le[(le["date"] >= s - pd.Timedelta(days=550)) & le["club"].isin(club2team)]
        known = (le["player_id"].map(known_from) < vc).to_numpy()
        val = le["player_id"].map(lv).to_numpy()
        vals = np.where(known & ~np.isnan(val), val, default_eur)
        teams = le["club"].map(club2team).to_numpy()
        for team in np.unique(teams):
            m = teams == team
            rows.append({"snap": s, "team": team, "v18": float(np.sort(vals[m])[::-1][:TOP_N].sum()),
                         "n": int(m.sum()), "known_share": float(known[m].mean())})
    return pd.DataFrame(rows)


# ── current squads: ESPN roster x dump value ───────────────────────────────────

def _norm(s) -> str:
    s = str(s).replace("ß", "ss").replace("ı", "i").replace("ł", "l").replace("ø", "o")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("-", " ").replace("'", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", s)).strip()


class _PlayerIndex:
    """Transfermarkt players with a value, indexed for roster matching."""

    def __init__(self, tm_dir: Path, club_map: pd.DataFrame):
        p = pd.read_csv(tm_dir / "players.csv",
                        usecols=["player_id", "name", "date_of_birth", "market_value_in_eur"])
        v, t = _load_dump(tm_dir)
        val = pd.concat([v.rename(columns={"current_club_id": "club"}),
                         t.rename(columns={"to_club_id": "club"})[["player_id", "date", "club", "market_value_in_eur"]]])
        val = val.dropna(subset=["market_value_in_eur"])
        val = val[val["market_value_in_eur"] > 0].sort_values("date")
        last = val.groupby("player_id").tail(1).set_index("player_id")
        p["value"] = p["player_id"].map(last["market_value_in_eur"]).fillna(p["market_value_in_eur"])
        p["value_date"] = p["player_id"].map(last["date"])
        p = p[p["value"] > 0].copy()
        p["by"] = pd.to_datetime(p["date_of_birth"], errors="coerce").dt.year
        p["k"] = p["name"].map(_norm)
        p["toks"] = p["k"].str.split().map(set)
        self.p = p
        self.P = p.set_index("player_id")
        club2team = dict(zip(club_map["club_id"], club_map["team"]))
        ev = pd.concat([v[["player_id", "current_club_id"]].rename(columns={"current_club_id": "club"}),
                        t[["player_id", "to_club_id"]].rename(columns={"to_club_id": "club"})])
        ev["team"] = ev["club"].map(club2team)
        valued = set(self.P.index)
        self.team_pool = ev.dropna(subset=["team"]).groupby("team")["player_id"].agg(lambda s: set(s) & valued)
        self.by_name = defaultdict(list)
        for pid, k in zip(p["player_id"], p["k"]):
            self.by_name[k].append(pid)
        self.by_year = {y: g for y, g in p.groupby("by")}

    def _yr_ok(self, pid, by, tol) -> bool:
        b = self.P.at[pid, "by"]
        return not (np.isfinite(by) and np.isfinite(b)) or abs(b - by) <= tol

    def match(self, name: str, team: str, by: float):
        """(player_id, how) — most to least trusted; None when nothing is safe."""
        k = _norm(name)
        et = k.split()
        if not et:
            return None, "none"
        # 1. players who have been at this club (ESPN ages are sometimes off: +-3)
        best = []
        for pid in self.team_pool.get(team, ()):
            ck, ct = self.P.at[pid, "k"], self.P.at[pid, "toks"]
            cl = ck.split()
            if ck == k:
                best.append((3, pid))
            elif not self._yr_ok(pid, by, 3):
                continue
            elif set(et) == ct or len(set(et) & ct) >= 2:
                best.append((2, pid))           # same tokens, any order
            elif (((len(et) > 1 and len(et[-1]) >= 3 and et[-1] in ct)
                   or (len(cl) > 1 and len(cl[-1]) >= 3 and cl[-1] in et)) and et[0][0] == cl[0][0]):
                best.append((2, pid))           # surname + first initial
            elif ((len(cl) == 1 and cl[0] in et) or (len(et) == 1 and et[0] in ct)) and self._yr_ok(pid, by, 1):
                best.append((2, pid))           # mononym (Alisson, Rodri)
            elif SequenceMatcher(None, k, ck).ratio() >= 0.8:
                best.append((1, pid))
        if best:
            def recency(pid):
                d = self.P.at[pid, "value_date"]
                return d if pd.notna(d) else pd.Timestamp(0)
            best.sort(key=lambda x: (x[0], recency(x[1])))
            return best[-1][1], "club"
        # 2. anywhere, exact name
        c = [pid for pid in self.by_name.get(k, []) if self._yr_ok(pid, by, 2)]
        if len(c) == 1:
            return c[0], "exact"
        if not np.isfinite(by):
            return None, "none"
        cands = pd.concat([self.by_year[y] for y in (by - 1, by, by + 1) if y in self.by_year])
        allt = set(et)
        sub = cands[cands["toks"].map(lambda s: s <= allt or allt <= s)]
        if len(sub) == 1:
            return sub["player_id"].iloc[0], "subset"
        close = [pid for pid, ck in zip(cands["player_id"], cands["k"])
                 if SequenceMatcher(None, k, ck).quick_ratio() >= 0.88
                 and SequenceMatcher(None, k, ck).ratio() >= 0.88]
        if len(close) == 1:
            return close[0], "fuzzy"
        return None, "none"


def build_current(rosters: pd.DataFrame, tm_dir: Path = TM_DIR, club_map: pd.DataFrame | None = None,
                  cache: pd.DataFrame | None = None, snap=None,
                  default_eur: float = DEFAULT_EUR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(team snapshot rows, player match table) for today's ESPN rosters.

    `cache` is the previous player match table: players already matched (by ESPN athlete
    id and team) are not re-matched, so the daily run only pays for new signings.
    """
    club_map = club_map if club_map is not None else build_club_map(tm_dir)
    snap = pd.Timestamp(snap).normalize() if snap is not None else pd.Timestamp.now().normalize()
    r = rosters.copy()
    r["team"] = r["team"].map(canonical_name)
    r["by"] = snap.year - pd.to_numeric(r["age"], errors="coerce")
    known = {}
    if cache is not None and not cache.empty:
        known = {(int(a), t): (pid, how) for a, t, pid, how in
                 zip(cache["espn_athlete_id"], cache["team"], cache["player_id"], cache["how"])}
    idx = None
    pids, hows = [], []
    for row in r.itertuples(index=False):
        key = (int(row.espn_athlete_id), row.team)
        if key in known:
            pid, how = known[key]
        else:
            idx = idx or _PlayerIndex(tm_dir, club_map)
            pid, how = idx.match(row.player, row.team, row.by)
        pids.append(pid if pd.notna(pid) else None)
        hows.append(how)
    r["player_id"] = pids
    r["how"] = hows
    idx = idx or _PlayerIndex(tm_dir, club_map)
    r["value"] = r["player_id"].map(idx.P["value"])
    r["known"] = r["value"].notna()
    r["value_eur"] = r["value"].fillna(default_eur)
    teams = (r.groupby("team")
              .agg(v18=("value_eur", lambda s: float(s.nlargest(TOP_N).sum())),
                   n=("value_eur", "size"), known_share=("known", "mean"))
              .reset_index())
    teams.insert(0, "snap", snap)
    matches = r[["espn_athlete_id", "team", "player", "player_id", "how", "value"]]
    return teams, matches


def upsert_snapshots(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Replace any rows for the snapshot dates in `new`, keep everything else."""
    if existing is None or existing.empty:
        return new.sort_values(["snap", "team"]).reset_index(drop=True)
    keep = existing[~existing["snap"].isin(set(new["snap"]))]
    return pd.concat([keep, new], ignore_index=True).sort_values(["snap", "team"]).reset_index(drop=True)


# ── live: Transfermarkt squads ─────────────────────────────────────────────────

_CLUB_NOISE = re.compile(r"\b(fc|cf|ac|as|ss|ssc|us|uc|sv|vfb|vfl|tsg|sc|rc|rcd|ca|cd|ud|sd|ogc|afc|"
                         r"calcio|club|de|du|football|futbol|1|1899|1846|1904|1907|1909|04|05|07|09)\b")


def _club_key(name: str) -> str:
    return re.sub(r"\s+", " ", _CLUB_NOISE.sub(" ", _norm(name))).strip()


def map_live_clubs(tm: pd.DataFrame, club_map: pd.DataFrame, current: pd.DataFrame) -> dict[int, str]:
    """TM club_id -> canonical team for this season's clubs.

    Known ids come from the history club map. A club new to the Big Five (not in the map)
    is matched by name to the still-unmatched current team of the same competition
    (`current`: competition, team -- from the ESPN rosters).
    """
    known = dict(zip(club_map["club_id"], club_map["team"]))
    out = {int(c): known[c] for c in tm["club_id"].unique() if c in known}
    for comp, grp in tm.drop_duplicates("club_id").groupby("competition"):
        free = sorted(set(current.loc[current["competition"] == comp, "team"]) - set(out.values()))
        for r in grp.itertuples():
            if r.club_id in out or not free:
                continue
            k = _club_key(r.club_name)
            scored = sorted(((SequenceMatcher(None, k, _club_key(t)).ratio(), t) for t in free), reverse=True)
            if scored and scored[0][0] >= 0.6:
                out[int(r.club_id)] = scored[0][1]
                free.remove(scored[0][1])
    return out


def build_live(tm: pd.DataFrame, club_map: pd.DataFrame, current: pd.DataFrame,
               snap=None) -> tuple[pd.DataFrame, list[str]]:
    """(team snapshot rows, unmapped TM clubs) from a Transfermarkt squad scrape."""
    snap = pd.Timestamp(snap).normalize() if snap is not None else pd.Timestamp.now().normalize()
    ids = map_live_clubs(tm, club_map, current)
    t = tm.assign(team=tm["club_id"].map(ids))
    unmapped = sorted(t.loc[t["team"].isna(), "club_name"].unique())
    t = t.dropna(subset=["team"])
    teams = (t.groupby("team")
              .agg(v18=("value_eur", lambda s: float(s.dropna().nlargest(TOP_N).sum())),
                   n=("player_id", "size"), known_share=("value_eur", lambda s: float(s.notna().mean())))
              .reset_index())
    teams.insert(0, "snap", snap)
    return teams[teams["v18"] > 0], unmapped
