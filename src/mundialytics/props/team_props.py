from __future__ import annotations

"""Team-event props: corners, yellow cards, fouls, shots, shots on target.

Per market and side, a PoissonRegressor on the proven rate-feature recipe
(team rolling event-for windows 5/10/19 + EWMA hl5, opponent event-against
rolling, is_home). Over-dispersion is measured on training totals and O/U
probabilities use Negative Binomial when var/mean > 1.1 (all five markets in
practice: corners 1.18, yellows 1.13, fouls 1.59, shots 1.39, SOT 1.19).

Validation (scripts/backtest_team_props.py, folds 2021/22-2025/26 vs
league-mean baseline, binary log-loss on standard lines): fouls -0.047/-0.056
(5/5 folds), shots -0.025 (5/5), SOT -0.018 (5/5), yellows -0.015/-0.018
(5/5, 4/5), corners -0.003/-0.005 (4/5, weakest as expected — corners are the
noisiest market).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor

MARKETS: dict[str, tuple[str, str, list[float]]] = {
    "corners": ("home_corners", "away_corners", [7.5, 8.5, 9.5, 10.5, 11.5]),
    "yellows": ("home_yellow_cards", "away_yellow_cards", [2.5, 3.5, 4.5, 5.5, 6.5]),
    "fouls":   ("home_fouls", "away_fouls", [19.5, 21.5, 23.5, 25.5]),
    "shots":   ("home_shots", "away_shots", [20.5, 22.5, 24.5, 26.5]),
    "sot":     ("home_sot", "away_sot", [6.5, 7.5, 8.5, 9.5]),
}
WINDOWS = (5, 10, 19)


def _prob_over(total_lam: float | np.ndarray, line: float, disp: float) -> float | np.ndarray:
    """P(total > line); NB with var = disp*mean when over-dispersed, else Poisson."""
    k = int(np.floor(line))
    lam = np.clip(total_lam, 0.2, 40.0)
    if disp > 1.1:
        r = lam / (disp - 1.0)
        return 1.0 - nbinom.cdf(k, r, 1.0 / disp)
    return 1.0 - poisson.cdf(k, lam)


@dataclass
class TeamPropsModel:
    """Fit on the events foundation; predict O/U probabilities for a fixture."""

    seasons_from: str = "2014-2015"
    _models: dict = field(default_factory=dict, init=False, repr=False)
    _disp: dict = field(default_factory=dict, init=False, repr=False)
    _league_lam: dict = field(default_factory=dict, init=False, repr=False)
    _team_feats: dict = field(default_factory=dict, init=False, repr=False)
    _feat_names: list = field(default_factory=list, init=False, repr=False)

    def fit(self, matches: pd.DataFrame) -> "TeamPropsModel":
        """`matches`: foundation rows with date/season/home_team/away_team + event columns."""
        df = matches.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["season"] >= self.seasons_from]
        for market, (hc, ac, _) in MARKETS.items():
            m = df.dropna(subset=[hc, ac, "date"]).copy()
            m[hc] = pd.to_numeric(m[hc], errors="coerce")
            m[ac] = pd.to_numeric(m[ac], errors="coerce")
            m = m.dropna(subset=[hc, ac])
            if len(m) < 3000:
                continue
            lr = self._long_rows(m, hc, ac)
            feats = self._feature_names()
            tr = lr.dropna(subset=feats + ["ev_for"])
            reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
            self._models[market] = reg
            tt = (m[hc] + m[ac]).astype(float)
            self._disp[market] = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
            if "competition" in m.columns:
                self._league_lam[market] = m.assign(tot=tt).groupby("competition")["tot"].mean().to_dict()
            # latest per-team feature state = rolling stats over each team's most recent games
            self._team_feats[market] = self._latest_team_state(lr)
        self._feat_names = self._feature_names()
        return self

    @staticmethod
    def _long_rows(m: pd.DataFrame, hc: str, ac: str) -> pd.DataFrame:
        rows = []
        for r in m.itertuples(index=False):
            rows.append(dict(match_id=r.match_id, date=r.date, team=r.home_team, opp=r.away_team,
                             is_home=1, ev_for=getattr(r, hc), ev_against=getattr(r, ac)))
            rows.append(dict(match_id=r.match_id, date=r.date, team=r.away_team, opp=r.home_team,
                             is_home=0, ev_for=getattr(r, ac), ev_against=getattr(r, hc)))
        lr = pd.DataFrame(rows).sort_values(["team", "date", "match_id"])
        for col in ["ev_for", "ev_against"]:
            for w in WINDOWS:
                lr[f"{col}_r{w}"] = (lr.groupby("team", group_keys=False)[col]
                                     .apply(lambda s: s.shift(1).rolling(w, min_periods=3).mean()))
            lr[f"{col}_ewm"] = (lr.groupby("team", group_keys=False)[col]
                                .apply(lambda s: s.shift(1).ewm(halflife=5, min_periods=3).mean()))
        opp_src = [f"ev_against_r{w}" for w in WINDOWS] + ["ev_against_ewm"]
        opp = lr[["match_id", "team"] + opp_src].rename(
            columns={"team": "opp", **{c: f"opp_{c}" for c in opp_src}})
        return lr.merge(opp, on=["match_id", "opp"], how="left")

    @staticmethod
    def _feature_names() -> list[str]:
        return ([f"ev_for_r{w}" for w in WINDOWS] + ["ev_for_ewm"]
                + [f"opp_ev_against_r{w}" for w in WINDOWS] + ["opp_ev_against_ewm"] + ["is_home"])

    @staticmethod
    def _latest_team_state(lr: pd.DataFrame) -> dict:
        """Per team: for/against rolling stats INCLUDING its last played game (for the next fixture)."""
        out: dict = {}
        for team, g in lr.groupby("team"):
            s_for, s_against = g["ev_for"], g["ev_against"]
            st = {}
            for w in WINDOWS:
                st[f"for_r{w}"] = float(s_for.tail(w).mean()) if len(s_for) >= 3 else np.nan
                st[f"against_r{w}"] = float(s_against.tail(w).mean()) if len(s_against) >= 3 else np.nan
            st["for_ewm"] = float(s_for.ewm(halflife=5).mean().iloc[-1]) if len(s_for) >= 3 else np.nan
            st["against_ewm"] = float(s_against.ewm(halflife=5).mean().iloc[-1]) if len(s_against) >= 3 else np.nan
            out[team] = st
        return out

    def _side_lambda(self, market: str, team: str, opp: str, is_home: int) -> float | None:
        mf = self._team_feats.get(market, {})
        st_t = mf.get(team) or mf.get(team.lower())      # foundation team names are lowercase
        st_o = mf.get(opp) or mf.get(opp.lower())
        if st_t is None or st_o is None:
            return None
        row = ([st_t[f"for_r{w}"] for w in WINDOWS] + [st_t["for_ewm"]]
               + [st_o[f"against_r{w}"] for w in WINDOWS] + [st_o["against_ewm"]] + [float(is_home)])
        if any(pd.isna(v) for v in row):
            return None
        x = pd.DataFrame([row], columns=self._feat_names)
        return float(np.clip(self._models[market].predict(x)[0], 0.1, 25))

    def predict_fixture(self, home_team: str, away_team: str) -> dict:
        """{market: {"lambda_home", "lambda_away", "lambda_total", "dispersion",
        "over": {line: p}}} for every market where both teams have history."""
        out: dict = {}
        for market in MARKETS:
            if market not in self._models:
                continue
            lh = self._side_lambda(market, home_team, away_team, 1)
            la = self._side_lambda(market, away_team, home_team, 0)
            if lh is None or la is None:
                continue
            disp = self._disp[market]
            lines = MARKETS[market][2]
            out[market] = {
                "lambda_home": round(lh, 2), "lambda_away": round(la, 2),
                "lambda_total": round(lh + la, 2), "dispersion": round(disp, 2),
                "over": {ln: round(float(_prob_over(lh + la, ln, disp)), 4) for ln in lines},
            }
        return out
