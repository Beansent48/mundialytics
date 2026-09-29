from __future__ import annotations

"""Player props: anytime scorer, 2+ goals, shots O1.5/O2.5, assist, yellow card.

Recipe (validated in scripts/backtest_player_props.py, folds 2021/22-2025/26,
ALL props 5/5 folds vs the strongest baseline, ECE 0.001-0.014):
  rates      career per-90 credibility-shrunk toward position-group priors
             (K = 900 minutes); goals use 0.7*xG-rate + 0.3*goal-rate,
             assists 0.7*xA-rate + 0.3*assist-rate.
  minutes    E[min | plays]: last-10-played average shrunk toward the
             position-group mean (n/(n+3)), clipped to [20, 95].
  context    (team match lambda / team baseline lambda)^0.7 for attacking props
             (team uplift does not transfer 1:1 to a player); cards instead use
             minutes^0.7 (sub-linear: late-game refs, subs get carded).
  dists      Poisson for goals/assists/cards; Negative Binomial (disp 1.3,
             train-picked) for shots.

Probabilities are CONDITIONED ON PLAYING (bookmaker "must play" convention).
Teams are addressed by FOUNDATION names (Understat names mapped internally).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from mundialytics.enrichment.understat_team_aliases import to_foundation_name

K_MIN = 900.0
K_RECENT = 450.0
# How stale the appearance-derived roster may be before a team is served no
# players at all. That roster is "whoever featured in this club's last ten
# games", which carries no notion of WHEN those games were: a club promoted
# after years away came back with the squad it last had in the top flight —
# Hull's 2026/27 attack was its 2016/17 one, Harry Maguire included. Just over
# a season keeps a club that spent last year in the second tier and drops one
# gone longer; an empty shortlist is the honest answer for a squad we cannot see.
ROSTER_MAX_AGE_DAYS = 400.0

# Competitions whose squads this model may install. It is fitted on big-five
# history, and the squad table now also carries the 108 clubs drawn into Europe
# -- two thirds of which play in leagues never measured here. Those clubs are
# excluded by WHERE THEY PLAY rather than by how many of their names we happen
# to recognise, because the two cannot be told apart by counting: a promoted
# Ligue 1 side matched 5 players, and European clubs matched up to 10. Judging
# by the count would have stripped real big-five clubs of their squads.
UEFA_COMPETITIONS = frozenset({
    "Champions League", "Europa League", "Conference League",
})
SHOTS_DISP = 1.3
PEN_CONV = 0.78  # measured penalty conversion in our shots data
STATS = ["xg", "goals", "shots", "xa", "assists", "yellow_cards", "npxg", "npgoals"]
# per-stat recency blend (A/B tested): last-15-appearance form helps shots/goals,
# hurts cards (noisy small-sample) and is neutral for assists -> career-only there
RECENT_W = {"xg": 0.5, "goals": 0.5, "shots": 0.5, "xa": 0.0, "assists": 0.0, "yellow_cards": 0.0,
            "npxg": 0.5, "npgoals": 0.5}
# Pricing once the XI is out (predict_lineup). E[min | plays] averages a
# player's starts and cameos; with the published lineup each player is priced on
# the minutes of the role he has tonight — as a starter or from the bench — and
# mu = scale * rate * (min/90)^exponent, because a per-90 career rate that mixes
# both roles overshoots starters and undershoots subs when multiplied by raw
# minutes. Exponent and scale were picked on pre-2021 seasons only; on the
# 2021/22-2025/26 folds every prop beat the morning price 5/5 (anytime -0.0047,
# shots 1.5 -0.025, assist -0.0031, yellow -0.0040 log-loss) and top-1 scorer
# picks went 37.6% -> 39.7%. scripts/experiment_player_lineup_minutes.py
LINEUP_MIN_FIT = {"goal": (0.7, 0.895), "shots": (0.8, 0.934), "ass": (0.7, 0.900), "yc": (0.6, 0.927)}


def _pos_group(p: str) -> str:
    p = str(p)
    if p == "GK":
        return "GK"
    if p.startswith("D") and not p.startswith("DM"):
        return "DEF"
    if p.startswith("DM") or p.startswith("M"):
        return "MID"
    if p.startswith("AM"):
        return "ATT"
    if p.startswith("F"):
        return "FW"
    return "SUB"


def _espn_pos_group(p: str) -> str | None:
    """ESPN's lineup slot (G, CD-L, LB, DM, CM-R, AM, LM, CF-L, ...) -> position
    group; None for SUB, which says nothing about where he plays."""
    p = str(p).upper()
    if p == "G":
        return "GK"
    if p.startswith("CD") or p in {"LB", "RB", "SW", "CB", "LWB", "RWB"}:
        return "DEF"
    if p.startswith("AM"):
        return "ATT"
    if p.startswith("CF") or p in {"F", "LF", "RF", "RCF", "LCF", "ST"}:
        return "FW"
    if p.startswith("CM") or p in {"DM", "LM", "RM", "M"}:
        return "MID"
    return None


def likely_xi(players: pd.DataFrame, n: int = 11) -> pd.DataFrame:
    """The n players most likely to start, from one team's predict output.

    exp_min is minutes WHEN FEATURING, so it cannot say who features: ranked on
    it alone, 42% of the published 2026/27 shortlist did not play (38% were not
    even in the squad). Starts in the team's last five matches decide first,
    exp_min breaks ties -- 58% -> 83% of shortlisted players playing, replayed
    on 2021/22-2025/26 (scripts/experiment_player_fresh_form.py)."""
    if players is None or players.empty:
        return players
    if "start_share" not in players.columns:
        return players.nlargest(n, "exp_min")
    return players.sort_values(["start_share", "exp_min"], ascending=False).head(n)


def _p_ge(mu: np.ndarray, k: int, disp: float = 1.0) -> np.ndarray:
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, 10)
    if disp > 1.05:
        r = mu / (disp - 1.0)
        return 1 - nbinom.cdf(k - 1, r, 1.0 / disp)
    return 1 - poisson.cdf(k - 1, mu)


@dataclass
class PlayerPropsModel:
    """Fit on understat_player_match rows; predict per-player prop probabilities."""

    _players: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _rosters: dict = field(default_factory=dict, init=False, repr=False)
    _pri: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _glob: dict = field(default_factory=dict, init=False, repr=False)
    _pos_min: dict = field(default_factory=dict, init=False, repr=False)
    _team_last_date: dict = field(default_factory=dict, init=False, repr=False)
    _stale_rosters: dict = field(default_factory=dict, init=False, repr=False)
    _current_squad_teams: set = field(default_factory=set, init=False, repr=False)
    _squad_unmatched: dict = field(default_factory=dict, init=False, repr=False)
    _squads_fp: str = field(default="", init=False, repr=False)
    _data_max_date: object = field(default=None, init=False, repr=False)

    def fit(self, pm: pd.DataFrame, shots: pd.DataFrame | None = None,
            shots_path: "str | Path | None" = None,
            current_squads: "pd.DataFrame | str | Path | None" = None,
            roster_max_age_days: float = ROSTER_MAX_AGE_DAYS,
            current: "pd.DataFrame | str | Path | None" = None) -> "PlayerPropsModel":
        """`pm`: understat player-match rows (player_id, player, team, game_id, date,
        position, minutes + base stats). All history is training; state = as of
        last game. `shots`/`shots_path`: understat shot events — penalties carry
        situation=NaN (soccerdata quirk) and power the pen-taker split of the
        goal mu (anytime 5/5 folds). Without them the model falls back exactly
        to the xG-based mu.

        `current_squads`: today's squad lists (see
        mundialytics.identity.current_squads). Pass it when SERVING — it decides
        who is in each roster, and nothing else. Rates, minutes and the whole
        probability recipe are untouched, so a backtest that omits it reproduces
        the validated model exactly. `roster_max_age_days` caps how old the
        fallback roster may be; see _guard_stale_rosters.

        `current`: ESPN player-match rows for the season Understat has not
        published (data/external/advanced/espn/espn_player_match_current.csv).
        See _append_current; without it the state freezes at Understat's last
        match, which is what production did until v0.57.0."""
        pm = pm.copy()
        pm["date"] = pd.to_datetime(pm["date"], errors="coerce")
        pm = pm.dropna(subset=["date"])
        for c in ["minutes", "xg", "goals", "shots", "xa", "assists", "yellow_cards"]:
            pm[c] = pd.to_numeric(pm[c], errors="coerce").fillna(0.0)
        self._n_current_rows = 0
        if current is not None:
            pm = self._append_current(pm, current)
            self._n_current_rows = int(pm.attrs.get("n_current_rows", 0))

        # penalties per (game, player) -> npxg/npgoals + taker-share ingredients
        if shots is None and shots_path is not None and Path(shots_path).exists():
            shots = pd.read_csv(shots_path, usecols=["game_id", "player_id", "situation", "result"])
        if shots is not None:
            pen = shots[shots["situation"].isna()].copy()
            pen["pen_goal"] = (pen["result"] == "Goal").astype(float)
            pg = pen.groupby(["game_id", "player_id"]).agg(
                pen_att=("result", "size"), pen_goal=("pen_goal", "sum")).reset_index()
            pm = pm.merge(pg, on=["game_id", "player_id"], how="left")
        for c in ["pen_att", "pen_goal"]:
            if c not in pm.columns:
                pm[c] = 0.0
        pm[["pen_att", "pen_goal"]] = pm[["pen_att", "pen_goal"]].fillna(0.0)
        pm["npxg"] = (pm["xg"] - 0.76 * pm["pen_att"]).clip(lower=0)
        pm["npgoals"] = (pm["goals"] - pm["pen_goal"]).clip(lower=0)
        tp = pm.groupby(["team", "game_id"])["pen_att"].sum().rename("team_pen_att").reset_index()
        pm = pm.merge(tp, on=["team", "game_id"], how="left")

        mode_pos = (pm[pm["position"] != "Sub"].groupby("player_id")["position"]
                    .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "MC"))
        pm["pgroup"] = pm["player_id"].map(mode_pos).map(_pos_group).fillna("MID")

        played = pm[pm["minutes"] > 0]
        self._pri = played.groupby("pgroup").apply(
            lambda gr: pd.Series({c: gr[c].sum() / max(gr["minutes"].sum(), 1) * 90.0 for c in STATS}),
            include_groups=False)
        self._glob = {c: played[c].sum() / max(played["minutes"].sum(), 1) * 90.0 for c in STATS}
        self._pos_min = played.groupby("pgroup")["minutes"].mean().to_dict()
        self._pos_min["_all"] = float(played["minutes"].mean())

        pm = pm.sort_values(["player_id", "date", "game_id"])
        agg = pm.groupby("player_id").agg(
            player=("player", "last"), team=("team", "last"), pgroup=("pgroup", "last"),
            last_date=("date", "max"), cmin=("minutes", "sum"),
            **{f"c_{c}": (c, "sum") for c in STATS})
        tail15 = (pm.groupby("player_id").tail(15).groupby("player_id")
                  .agg(rmin15=("minutes", "sum"), **{f"rr_{c}": (c, "sum") for c in STATS}))
        agg = agg.join(tail15)
        # pen-taker state: player vs team pens over the player's last 60 squad rows
        tail60 = (pm.groupby("player_id").tail(60).groupby("player_id")
                  .agg(p_pen60=("pen_att", "sum"), t_pen60=("team_pen_att", "sum")))
        agg = agg.join(tail60)
        # team attacking-pen rate per game (last 38 team games)
        tg = (tp.merge(pm[["team", "game_id", "date"]].drop_duplicates(), on=["team", "game_id"])
              .sort_values(["team", "date"]))
        self._team_pen_rate = tg.groupby("team")["team_pen_att"].apply(
            lambda s: float(s.tail(38).mean())).to_dict()
        mp = pm[pm["minutes"] > 0].groupby("player_id")["minutes"]
        agg["avg_minp10"] = mp.apply(lambda s: s.tail(10).mean())
        agg["nplayed10"] = mp.apply(lambda s: min(len(s), 10))
        # role minutes for predict_lineup: over the player's last 10 squad rows,
        # minutes in the games he started / came on in (harness semantics)
        is_sub = pm["position"].astype(str) == "Sub"
        last10 = pm.assign(
            min_start=pm["minutes"].where(~is_sub & (pm["minutes"] > 0)),
            min_sub=pm["minutes"].where(is_sub & (pm["minutes"] > 0)),
        ).groupby("player_id").tail(10).groupby("player_id")
        agg["avg_min_start10"] = last10["min_start"].mean()
        agg["n_start10"] = last10["min_start"].count()
        agg["avg_min_sub10"] = last10["min_sub"].mean()
        agg["n_sub10"] = last10["min_sub"].count()
        self._pos_min_start = played[played["position"] != "Sub"].groupby("pgroup")["minutes"].mean().to_dict()
        self._pos_min_sub = played[played["position"] == "Sub"].groupby("pgroup")["minutes"].mean().to_dict()
        self._name_idx = None
        self._players = agg

        # roster: players seen in each team's last 10 games
        tg = pm[["team", "game_id", "date"]].drop_duplicates().sort_values(["team", "date"])
        tg["tgn"] = tg.groupby("team").cumcount()
        last_tgn = tg.groupby("team")["tgn"].max()
        pm2 = pm.merge(tg[["team", "game_id", "tgn"]], on=["team", "game_id"])
        recent = pm2[pm2["tgn"] > pm2["team"].map(last_tgn) - 10]
        self._rosters = recent.groupby("team")["player_id"].agg(lambda s: sorted(set(s))).to_dict()
        # starts in each club's last five matches -> likely_xi
        last5 = pm2[pm2["tgn"] > pm2["team"].map(last_tgn) - 5]
        n5 = (last5.groupby("team")["tgn"].nunique()).to_dict()
        st5 = last5[(last5["position"] != "Sub") & (last5["minutes"] > 0)]
        share = st5.groupby(["player_id", "team"]).size().reset_index(name="n")
        share["start_share"] = share["n"] / share["team"].map(n5)
        # a player's current club is his most recent one
        cur_team = pm.groupby("player_id")["team"].last()
        share = share[share["team"] == share["player_id"].map(cur_team)]
        self._players["start_share"] = share.set_index("player_id")["start_share"].reindex(
            self._players.index).fillna(0.0)
        self._data_max_date = pm["date"].max()
        self._team_last_date = tg.groupby("team")["date"].max().to_dict()
        # foundation-name lookup for the Understat teams we know
        self._fd_to_us = {to_foundation_name(t): t for t in self._rosters}
        # per-team attacking baseline: mean team xG over its last 19 games (players summed)
        txg = pm.groupby(["team", "game_id"], sort=False).agg(xg=("xg", "sum"), date=("date", "first"))
        txg = txg.reset_index().sort_values(["team", "date"])
        self._team_xg_base = txg.groupby("team")["xg"].apply(lambda s: float(s.tail(19).mean())).to_dict()
        self._glob_xg = float(txg["xg"].mean())

        self._guard_stale_rosters(roster_max_age_days)
        self._apply_current_squads(current_squads)
        return self

    @staticmethod
    def _append_current(pm: pd.DataFrame, current) -> pd.DataFrame:
        """ESPN rows for the current season, in Understat's shape, appended to pm.

        Understat stopped on 2026-05-24, so without these every 2026/27 price
        came from each player's state at the end of 2025/26 -- form, minutes and
        the published "likely XI" alike (live: 38% of shortlisted players were
        not even in the matchday squad). ESPN has every squad with goals, shots,
        assists, cards and starter/sub, but no minutes and no xG, so:
          minutes  a starter's / sub's own mean in that role before, else his
                   position's (83 / 24 as a last resort); unused bench = 0
          xG       shots x his xG per shot, shrunk to the league's with 20 shots
          xA       his xA per 90 carries on (ESPN has no chance creation)
        Replayed on 2021/22-2025/26 this recovered ~95% of what full data gives
        over the frozen state, every prop 5/5 (shots 1.5 -0.019, anytime -0.0049)
        and lifted the shortlist from 58% to 83% of players who played.
        scripts/experiment_player_fresh_form.py. Players are matched by name;
        anyone ESPN names that Understat never measured starts a new record."""
        from mundialytics.identity.current_squads import NameIndex

        cur = pd.read_csv(current) if not isinstance(current, pd.DataFrame) else current.copy()
        need = {"event_id", "date", "team", "player", "position", "starter", "appearances"}
        if cur.empty or not need <= set(cur.columns):
            return pm
        cur["date"] = pd.to_datetime(cur["date"], errors="coerce")
        cur = cur[cur["date"] > pm["date"].max()].dropna(subset=["date", "player"])
        if cur.empty:
            return pm
        for c in ["goals", "shots", "assists", "yellow_cards", "appearances"]:
            cur[c] = pd.to_numeric(cur.get(c), errors="coerce").fillna(0.0)
        started = cur["starter"].astype(str).str.lower().isin(["true", "1"])

        # who is who: Understat's players by name, most recent first
        last = pm.groupby("player_id").agg(player=("player", "last"), date=("date", "max"))
        mode_pos = (pm[pm["position"] != "Sub"].groupby("player_id")["position"]
                    .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "MC"))
        pgroup = mode_pos.map(_pos_group)
        idx = NameIndex()
        for pid, r in last.iterrows():
            idx.add(r["player"], pid, rank=r["date"].timestamp())
        new_ids: dict[str, int] = {}
        next_id = int(min(pm["player_id"].min(), 0)) - 1
        pids = []
        for who, pos in zip(cur["player"].astype(str), cur["position"].astype(str)):
            pid, kind = idx.lookup_detail(who)
            # a loose (short-key) match must not hand a keeper a striker's history
            if pid is not None and kind != "full" and pos != "SUB":
                if (pgroup.get(pid) == "GK") != (pos == "G"):
                    pid = None
            if pid is None:
                if who not in new_ids:
                    new_ids[who] = next_id
                    next_id -= 1
                pid = new_ids[who]
            pids.append(pid)
        cur["player_id"] = pids

        # role minutes and xG per shot from each player's Understat history
        hist = pm[pm["minutes"] > 0]
        is_sub = hist["position"] == "Sub"
        own_st = hist[~is_sub].groupby("player_id")["minutes"].mean()
        own_sb = hist[is_sub].groupby("player_id")["minutes"].mean()
        grp = pd.Series(cur["player_id"].map(pgroup).to_numpy(), index=cur.index)
        grp = grp.fillna(cur["position"].map(_espn_pos_group)).fillna("MID")
        # a new player who only came off the bench so far: his group is unknown
        # from this row; take the one from any start he has made this season
        first = (cur[cur["position"] != "SUB"].assign(g=cur["position"].map(_espn_pos_group))
                 .groupby("player_id")["g"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "MID"))
        new = cur["player_id"] < 0
        grp[new] = cur.loc[new, "player_id"].map(first).fillna(grp[new])
        hist_g = hist["player_id"].map(pgroup)
        pos_st = hist[~is_sub].groupby(hist_g[~is_sub])["minutes"].mean()
        pos_sb = hist[is_sub].groupby(hist_g[is_sub])["minutes"].mean()
        est_st = cur["player_id"].map(own_st).fillna(grp.map(pos_st)).fillna(83.0)
        est_sb = cur["player_id"].map(own_sb).fillna(grp.map(pos_sb)).fillna(24.0)
        mins = np.where(cur["appearances"] > 0, np.where(started, est_st, est_sb), 0.0)
        glob_xps = float(hist["xg"].sum() / max(hist["shots"].sum(), 1.0))
        n_sh = hist.groupby("player_id")["shots"].sum()
        raw_xps = (hist.groupby("player_id")["xg"].sum() / n_sh.replace(0, np.nan)).clip(0.02, 0.5)
        xps = ((raw_xps * n_sh + glob_xps * 20) / (n_sh + 20)).fillna(glob_xps)
        xa90 = hist.groupby("player_id")["xa"].sum() / hist.groupby("player_id")["minutes"].sum() * 90.0
        pos_xa90 = (hist.groupby(hist_g)["xa"].sum() / hist.groupby(hist_g)["minutes"].sum() * 90.0)

        # teams under the name the Understat rows use, where the club is known
        fd_to_us = {to_foundation_name(t): t for t in pm["team"].dropna().unique()}
        code = {"GK": "GK", "DEF": "DC", "MID": "MC", "ATT": "AMC", "FW": "FW"}
        rows = pd.DataFrame({
            "player_id": cur["player_id"].to_numpy(),
            "player": cur["player"].to_numpy(),
            "team": [fd_to_us.get(t, t) for t in cur["team"].astype(str)],
            "game_id": pd.to_numeric(cur["event_id"], errors="coerce").to_numpy(),
            "date": cur["date"].to_numpy(),
            "position": np.where(started, grp.map(code).fillna("MC"), "Sub"),
            "minutes": mins,
            "goals": cur["goals"].to_numpy(), "shots": cur["shots"].to_numpy(),
            "assists": cur["assists"].to_numpy(), "yellow_cards": cur["yellow_cards"].to_numpy(),
            "xg": (cur["shots"] * cur["player_id"].map(xps).fillna(glob_xps)).to_numpy(),
            "xa": (cur["player_id"].map(xa90).fillna(grp.map(pos_xa90)).fillna(0.0) * mins / 90.0).to_numpy(),
        })
        out = pd.concat([pm, rows], ignore_index=True)
        out.attrs["n_current_rows"] = len(rows)
        return out

    def _guard_stale_rosters(self, max_age_days: float) -> None:
        """Drop rosters whose last game is too far behind the rest of the data.

        Measured against the data's own end, not today's date, so a backtest
        fold judges staleness by its own clock. A team still playing sits at
        zero days and is never touched."""
        self._stale_rosters = {}
        if not max_age_days or self._data_max_date is None:
            return
        cutoff = pd.Timestamp(self._data_max_date) - pd.Timedelta(days=float(max_age_days))
        for team, last in list(self._team_last_date.items()):
            if pd.notna(last) and pd.Timestamp(last) < cutoff and team in self._rosters:
                self._stale_rosters[team] = pd.Timestamp(last)
                del self._rosters[team]

    def _apply_current_squads(self, current_squads) -> None:
        """Replace the appearance-derived rosters with today's actual squads.

        Each squad member is matched back to his own history by name, so a
        player carries his rates to the club he now plays for — something the
        appearance roster could not express at all, since it only ever knew him
        at the club he was at when the data stopped. A squad member with no
        big-five history is left out rather than handed invented rates: the
        shortlist gets shorter, which is the truthful reading of "this player
        has never been measured here"."""
        from mundialytics.identity.current_squads import (
            NameIndex, load_current_squads, squads_fingerprint,
        )

        self._current_squad_teams = set()
        self._squad_unmatched = {}
        self._unmodelled_squads = []
        self._squads_fp = ""
        if current_squads is None:
            return
        cs = (current_squads if isinstance(current_squads, pd.DataFrame)
              else load_current_squads(current_squads))
        if cs is None or cs.empty or self._players is None:
            return
        # stamped on the model so a consumer reading a cached fit can tell
        # whether the squads have moved on since it was built
        self._squads_fp = squads_fingerprint(cs)

        idx = NameIndex()
        for pid, row in self._players.iterrows():
            last = row.get("last_date")
            rank = pd.Timestamp(last).timestamp() if pd.notna(last) else 0.0
            idx.add(row["player"], pid, rank=rank)

        if "pos_group" not in cs.columns:
            cs = cs.assign(pos_group="Unknown")
        # One or two coincidental name matches is enough to install a "roster",
        # and predict_fixture would then serve it as an authoritative shortlist
        # of one. A club we do not model gets what it got before the squad table
        # reached it: nothing.
        if "competition" in cs.columns:
            unmodelled = cs["competition"].isin(UEFA_COMPETITIONS)
            self._unmodelled_squads = sorted(set(cs.loc[unmodelled, "team"].astype(str)))
            cs = cs[~unmodelled]

        for team, grp in cs.groupby("team"):
            ids, missing = [], []
            for who, squad_pos in zip(grp["player"], grp["pos_group"]):
                pid, kind = idx.lookup_detail(who)
                if pid is None:
                    missing.append(str(who))
                    continue
                # A short-key or containment match got here by dropping name
                # parts, which is how a goalkeeper ends up wearing a striker's
                # history. The position vocabularies only line up on the keeper,
                # so that is the one contradiction worth refusing.
                if kind != "full":
                    was_gk = str(self._players.at[pid, "pgroup"]) == "GK"
                    if was_gk != (str(squad_pos) == "Goalkeeper"):
                        missing.append(str(who))
                        continue
                ids.append(pid)
            if not ids:
                continue
            # keep the Understat spelling when we know the club, so the pen rate
            # and the attacking baseline keyed on it still resolve
            key = self._fd_to_us.get(team, team)
            self._rosters[key] = sorted(set(ids))
            self._fd_to_us[team] = key
            self._current_squad_teams.add(key)
            self._squad_unmatched[key] = missing
            self._stale_rosters.pop(key, None)

    def _resolve_team(self, team: str) -> str | None:
        if team in self._rosters:
            return team
        hit = self._fd_to_us.get(team) or self._fd_to_us.get(team.lower())
        if hit:
            return hit
        # case-insensitive scan over Understat names ('Real Madrid' vs 'real madrid')
        low = team.lower()
        return next((t for t in self._rosters if t.lower() == low), None)

    def predict_team_players(self, team: str, atk_factor: float = 1.0) -> pd.DataFrame:
        """Prop probabilities for every rostered player of `team` (foundation or
        Understat name). `atk_factor` = engine match lambda / team baseline lambda."""
        us_team = self._resolve_team(team)
        if us_team is None or self._players is None:
            return pd.DataFrame()
        ids = self._rosters.get(us_team, [])
        P = self._players.loc[[i for i in ids if i in self._players.index]].copy()
        if P.empty:
            return P
        return self._price(P, us_team, atk_factor)

    def _price(self, P: pd.DataFrame, us_team: str, atk_factor: float,
               started: pd.Series | None = None) -> pd.DataFrame:
        """Prop probabilities for the players in P. `started` (bool per row) switches
        minutes to the confirmed role — see LINEUP_MIN_FIT; None is the morning price."""
        for c in STATS:
            prior = P["pgroup"].map(self._pri[c]).fillna(self._glob[c])
            raw = np.where(P["cmin"] > 0, P[f"c_{c}"] / P["cmin"].clip(lower=1e-9) * 90.0, prior)
            cred = P["cmin"] / (P["cmin"] + K_MIN)
            r_car = cred * raw + (1 - cred) * prior
            w = RECENT_W[c]
            if w > 0:
                rmin = P["rmin15"].fillna(0.0)
                raw_r = np.where(rmin > 0, P[f"rr_{c}"].fillna(0.0) / rmin.clip(lower=1e-9) * 90.0, r_car)
                cred_r = rmin / (rmin + K_RECENT)
                r_rec = cred_r * raw_r + (1 - cred_r) * r_car
                P[f"r_{c}"] = w * r_rec + (1 - w) * r_car
            else:
                P[f"r_{c}"] = r_car

        prior_min = P["pgroup"].map(self._pos_min).fillna(self._pos_min["_all"])
        cred_m = P["nplayed10"].fillna(0) / (P["nplayed10"].fillna(0) + 3.0)
        P["exp_min"] = (cred_m * P["avg_minp10"].fillna(prior_min) + (1 - cred_m) * prior_min).clip(20, 95)
        emins = P["exp_min"] / 90.0
        af = float(np.clip(atk_factor, 0.4, 2.5)) ** 0.7
        # per-mu minutes multiplier: the morning recipe, or the confirmed role's
        if started is None:
            m_goal = m_shots = m_ass = emins
            m_yc = emins ** 0.7
        else:
            st = started.reindex(P.index).fillna(False).astype(bool)
            role_min = np.where(
                st,
                self._role_minutes(P, "start", getattr(self, "_pos_min_start", {}), 83.0),
                self._role_minutes(P, "sub", getattr(self, "_pos_min_sub", {}), 24.0))
            P["exp_min"] = role_min
            em = pd.Series(np.clip(role_min, 5, 95) / 90.0, index=P.index)
            (g_g, a_g), (g_s, a_s), (g_a, a_a), (g_y, a_y) = (
                LINEUP_MIN_FIT[k] for k in ("goal", "shots", "ass", "yc"))
            m_goal, m_shots, m_ass = a_g * em ** g_g, a_s * em ** g_s, a_a * em ** g_a
            m_yc = a_y * em ** g_y

        # goal mu = non-pen component + pen-taker component (5/5 folds; falls
        # back to the xG mu exactly when pen data was absent at fit)
        t_pen = P["t_pen60"].fillna(0.0)
        taker_share = (t_pen / (t_pen + 4.0)) * (P["p_pen60"].fillna(0.0) / t_pen.clip(lower=1e-9))
        pen_rate = getattr(self, "_team_pen_rate", {}).get(us_team, 0.22)
        mu_pen = taker_share * pen_rate * PEN_CONV * m_goal * af
        mu_goal = (0.7 * P["r_npxg"] + 0.3 * P["r_npgoals"]) * m_goal * af + mu_pen
        mu_shots = P["r_shots"] * m_shots * af
        mu_ass = (0.7 * P["r_xa"] + 0.3 * P["r_assists"]) * m_ass * af
        mu_yc = P["r_yellow_cards"] * m_yc

        out = pd.DataFrame({
            "player": P["player"], "pgroup": P["pgroup"], "team": us_team,
            "exp_min": P["exp_min"].round(0).astype(int),
            "start_share": P["start_share"] if "start_share" in P.columns else np.nan,
            "p_anytime_scorer": _p_ge(mu_goal, 1),
            "p_2plus_goals": _p_ge(mu_goal, 2),
            "p_shots_over_1_5": _p_ge(mu_shots, 2, SHOTS_DISP),
            "p_shots_over_2_5": _p_ge(mu_shots, 3, SHOTS_DISP),
            "p_assist": _p_ge(mu_ass, 1),
            "p_yellow": _p_ge(mu_yc, 1),
            "mu_goals": mu_goal.round(3), "mu_shots": mu_shots.round(2),
        }, index=P.index)
        return out.sort_values("p_anytime_scorer", ascending=False).round(4)

    @staticmethod
    def _role_minutes(P: pd.DataFrame, role: str, pos_prior: dict, default: float) -> np.ndarray:
        """Last-10 minutes in this role, shrunk n/(n+3) to the position's role mean."""
        prior = P["pgroup"].map(pos_prior).fillna(default)
        n = P.get(f"n_{role}10", pd.Series(0.0, index=P.index)).fillna(0.0)
        avg = P.get(f"avg_min_{role}10", pd.Series(np.nan, index=P.index)).fillna(prior)
        cred = n / (n + 3.0)
        return (cred * avg + (1 - cred) * prior).to_numpy(dtype=float)

    def predict_lineup(self, team: str, starters: list[str], bench: list[str] = (),
                       atk_factor: float = 1.0) -> tuple[pd.DataFrame, list[str]]:
        """Prop probabilities for a CONFIRMED matchday squad (ESPN names), each
        player priced on the minutes of his role tonight (LINEUP_MIN_FIT).

        Returns (props with a `started` column, names that matched no player
        with big-five history). Players are matched by name over everyone the
        model has measured, not the club roster, so a new signing still counts."""
        if self._players is None:
            return pd.DataFrame(), list(starters) + list(bench)
        from mundialytics.identity.current_squads import NameIndex

        if getattr(self, "_name_idx", None) is None:
            idx = NameIndex()
            for pid, row in self._players.iterrows():
                last = row.get("last_date")
                idx.add(row["player"], pid, rank=pd.Timestamp(last).timestamp() if pd.notna(last) else 0.0)
            self._name_idx = idx
        ids, started, missing = [], [], []
        for names, role in ((starters, True), (bench, False)):
            for who in names:
                pid = self._name_idx.lookup(who) if who else None
                if pid is None or pid in ids:
                    missing.append(str(who))
                    continue
                ids.append(pid)
                started.append(role)
        if not ids:
            return pd.DataFrame(), missing
        P = self._players.loc[ids].copy()
        us_team = self._resolve_team(team) or team
        out = self._price(P, us_team, atk_factor, started=pd.Series(started, index=P.index))
        out["started"] = pd.Series(started, index=P.index).reindex(out.index)
        return out, missing

    LEAGUE_MEAN_LAMBDA = 1.40  # engine's league-average side lambda (goals scale)

    def _atk_factor(self, team: str, lam: float | None, base: float | None) -> float:
        """Match uplift vs the team's OWN baseline, scale-normalized so a strong
        team doesn't get a permanent af > 1 (harness semantics)."""
        if lam is None:
            return 1.0
        if base is None:
            us = self._resolve_team(team)
            xg_base = self._team_xg_base.get(us) if us else None
            if not xg_base or xg_base <= 0:
                return 1.0
            return (lam / self.LEAGUE_MEAN_LAMBDA) / (xg_base / self._glob_xg)
        return lam / base

    def team_players_for_lambda(self, team: str, lam: float | None) -> pd.DataFrame:
        """One team's player props given only that side's match lambda (e.g. a
        European tie where the opponent isn't Understat-covered)."""
        return self.predict_team_players(team, self._atk_factor(team, lam, None))

    def predict_fixture(self, home_team: str, away_team: str,
                        lam_home: float | None = None, lam_away: float | None = None,
                        base_home: float | None = None, base_away: float | None = None) -> pd.DataFrame:
        """Both teams' players. Attack factors from engine lambdas when provided;
        baselines default to each team's own recent xG level."""
        h = self.predict_team_players(home_team, self._atk_factor(home_team, lam_home, base_home))
        a = self.predict_team_players(away_team, self._atk_factor(away_team, lam_away, base_away))
        if not h.empty:
            h = h.assign(side="home")
        if not a.empty:
            a = a.assign(side="away")
        return pd.concat([h, a])
