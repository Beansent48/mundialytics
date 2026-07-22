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
SHOTS_DISP = 1.3
PEN_CONV = 0.78  # measured penalty conversion in our shots data
STATS = ["xg", "goals", "shots", "xa", "assists", "yellow_cards", "npxg", "npgoals"]
# per-stat recency blend (A/B tested): last-15-appearance form helps shots/goals,
# hurts cards (noisy small-sample) and is neutral for assists -> career-only there
RECENT_W = {"xg": 0.5, "goals": 0.5, "shots": 0.5, "xa": 0.0, "assists": 0.0, "yellow_cards": 0.0,
            "npxg": 0.5, "npgoals": 0.5}


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

    def fit(self, pm: pd.DataFrame, shots: pd.DataFrame | None = None,
            shots_path: "str | Path | None" = None) -> "PlayerPropsModel":
        """`pm`: understat player-match rows (player_id, player, team, game_id, date,
        position, minutes + base stats). All history is training; state = as of
        last game. `shots`/`shots_path`: understat shot events — penalties carry
        situation=NaN (soccerdata quirk) and power the pen-taker split of the
        goal mu (anytime 5/5 folds). Without them the model falls back exactly
        to the xG-based mu."""
        pm = pm.copy()
        pm["date"] = pd.to_datetime(pm["date"], errors="coerce")
        pm = pm.dropna(subset=["date"])
        for c in ["minutes", "xg", "goals", "shots", "xa", "assists", "yellow_cards"]:
            pm[c] = pd.to_numeric(pm[c], errors="coerce").fillna(0.0)

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
        self._players = agg

        # roster: players seen in each team's last 10 games
        tg = pm[["team", "game_id", "date"]].drop_duplicates().sort_values(["team", "date"])
        tg["tgn"] = tg.groupby("team").cumcount()
        last_tgn = tg.groupby("team")["tgn"].max()
        pm2 = pm.merge(tg[["team", "game_id", "tgn"]], on=["team", "game_id"])
        recent = pm2[pm2["tgn"] > pm2["team"].map(last_tgn) - 10]
        self._rosters = recent.groupby("team")["player_id"].agg(lambda s: sorted(set(s))).to_dict()
        # foundation-name lookup for the Understat teams we know
        self._fd_to_us = {to_foundation_name(t): t for t in self._rosters}
        # per-team attacking baseline: mean team xG over its last 19 games (players summed)
        txg = pm.groupby(["team", "game_id"], sort=False).agg(xg=("xg", "sum"), date=("date", "first"))
        txg = txg.reset_index().sort_values(["team", "date"])
        self._team_xg_base = txg.groupby("team")["xg"].apply(lambda s: float(s.tail(19).mean())).to_dict()
        self._glob_xg = float(txg["xg"].mean())
        return self

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

        # goal mu = non-pen component + pen-taker component (5/5 folds; falls
        # back to the xG mu exactly when pen data was absent at fit)
        t_pen = P["t_pen60"].fillna(0.0)
        taker_share = (t_pen / (t_pen + 4.0)) * (P["p_pen60"].fillna(0.0) / t_pen.clip(lower=1e-9))
        pen_rate = getattr(self, "_team_pen_rate", {}).get(us_team, 0.22)
        mu_pen = taker_share * pen_rate * PEN_CONV * emins * af
        mu_goal = (0.7 * P["r_npxg"] + 0.3 * P["r_npgoals"]) * emins * af + mu_pen
        mu_shots = P["r_shots"] * emins * af
        mu_ass = (0.7 * P["r_xa"] + 0.3 * P["r_assists"]) * emins * af
        mu_yc = P["r_yellow_cards"] * emins ** 0.7

        out = pd.DataFrame({
            "player": P["player"], "pgroup": P["pgroup"], "team": us_team,
            "exp_min": P["exp_min"].round(0).astype(int),
            "p_anytime_scorer": _p_ge(mu_goal, 1),
            "p_2plus_goals": _p_ge(mu_goal, 2),
            "p_shots_over_1_5": _p_ge(mu_shots, 2, SHOTS_DISP),
            "p_shots_over_2_5": _p_ge(mu_shots, 3, SHOTS_DISP),
            "p_assist": _p_ge(mu_ass, 1),
            "p_yellow": _p_ge(mu_yc, 1),
            "mu_goals": mu_goal.round(3), "mu_shots": mu_shots.round(2),
        }, index=P.index)
        return out.sort_values("p_anytime_scorer", ascending=False).round(4)

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
