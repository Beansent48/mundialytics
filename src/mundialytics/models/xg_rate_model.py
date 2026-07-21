from __future__ import annotations

"""xG-rate lambda predictor — a first-class goal-lambda source built on rolling
pre-match xG form.

An 8-fold + 5-fold temporal backtest (2020-2025) found that a dedicated predictor
of match xG from rolling, opponent-adjusted xG rates beats the goals-form
GoalLambdaModel for 1X2 (RPS ~0.2027 vs ~0.2066 for the blend it replaces), and
that the RPS-optimal club blend is 0.60*xG-rate + 0.40*goals-AttackDefense — the
goals GoalLambdaModel gets 0 weight once this is present. See
[[project_xg_modeling_findings]].

For each team-row the model predicts expected xG-for from: the team's own trailing
xG-for rates (windows 5/10/19), the opponent's trailing xG-against rates, and a
home flag. Rolling features are shifted by one match (leakage-safe). Requires
home_xg/away_xg on the training matches; where xG is absent the engine falls back
to the goals path.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

WINDOWS = (5, 10, 19)


@dataclass
class XGRateModel:
    windows: Sequence[int] = WINDOWS
    alpha: float = 0.1
    lambda_floor: float = 0.05
    lambda_cap: float = 6.0
    _model: PoissonRegressor | None = field(default=None, init=False)
    _feats: list[str] = field(default_factory=list, init=False)
    # Walk-forward form state: per team, the recent xG-for / xG-against series
    # (most recent last, capped at max window). predict() reads current form and
    # update_form() appends results as they happen, so a mid-season prediction uses
    # form AS OF that date instead of a value frozen at training end.
    _hist_for: dict[str, list[float]] = field(default_factory=dict, init=False)
    _hist_against: dict[str, list[float]] = field(default_factory=dict, init=False)
    _global: dict[str, float] = field(default_factory=dict, init=False)

    # ── feature engineering ────────────────────────────────────────────────────

    def _long_rows(self, matches: pd.DataFrame) -> pd.DataFrame:
        m = matches.dropna(subset=["home_xg", "away_xg"]).copy()
        rows = []
        for _, r in m.iterrows():
            rows.append({"match_id": r.get("match_id"), "date": r.get("date"), "team": r["home_team"],
                         "opp": r["away_team"], "is_home": 1, "xg_for": r["home_xg"], "xg_against": r["away_xg"]})
            rows.append({"match_id": r.get("match_id"), "date": r.get("date"), "team": r["away_team"],
                         "opp": r["home_team"], "is_home": 0, "xg_for": r["away_xg"], "xg_against": r["home_xg"]})
        lr = pd.DataFrame(rows)
        lr["date"] = pd.to_datetime(lr["date"], errors="coerce")
        return lr

    def _add_rolling(self, lr: pd.DataFrame) -> pd.DataFrame:
        lr = lr.sort_values(["team", "date", "match_id"]).copy()
        for col in ["xg_for", "xg_against"]:
            lr[col] = pd.to_numeric(lr[col], errors="coerce")
            for w in self.windows:
                lr[f"{col}_r{w}"] = (lr.groupby("team", group_keys=False)[col]
                                     .apply(lambda s: s.shift(1).rolling(w, min_periods=3).mean()))
        opp_cols = {f"xg_against_r{w}": f"opp_xg_against_r{w}" for w in self.windows}
        opp = lr[["match_id", "team"] + list(opp_cols)].rename(columns={"team": "opp", **opp_cols})
        return lr.merge(opp, on=["match_id", "opp"], how="left")

    # ── fit / predict ──────────────────────────────────────────────────────────

    def fit(self, matches: pd.DataFrame) -> "XGRateModel":
        if not {"home_xg", "away_xg"}.issubset(matches.columns):
            return self
        # Guard the common case where the xG columns exist but are all NaN (pre-2014
        # rows, uncovered leagues) — the model stays not-ready and the engine falls back.
        if len(matches.dropna(subset=["home_xg", "away_xg"])) < 300:
            return self
        lr = self._add_rolling(self._long_rows(matches))
        self._feats = ([f"xg_for_r{w}" for w in self.windows]
                       + [f"opp_xg_against_r{w}" for w in self.windows] + ["is_home"])
        train = lr.dropna(subset=self._feats + ["xg_for"])
        if len(train) < 300:
            self._model = None
            return self
        self._model = PoissonRegressor(alpha=self.alpha, max_iter=1000).fit(
            train[self._feats], train["xg_for"].clip(lower=0))
        # Global fallback rate (median) for teams/windows with too little history.
        self._global = {f"xg_for_r{w}": float(pd.to_numeric(lr["xg_for"], errors="coerce").median()) for w in self.windows}
        self._global.update({f"xg_against_r{w}": float(pd.to_numeric(lr["xg_against"], errors="coerce").median()) for w in self.windows})
        # Seed the walk-forward form state with each team's training-history series.
        cap = max(self.windows)
        for team, g in lr.sort_values("date").groupby("team"):
            self._hist_for[str(team)] = pd.to_numeric(g["xg_for"], errors="coerce").dropna().tolist()[-cap:]
            self._hist_against[str(team)] = pd.to_numeric(g["xg_against"], errors="coerce").dropna().tolist()[-cap:]
        return self

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def update_form(self, home_team: str, away_team: str, home_xg: float, away_xg: float) -> None:
        """Append a played match to the walk-forward form state (call after each match
        in chronological order so subsequent predictions use up-to-date form)."""
        cap = max(self.windows)
        for team, xf, xa in [(str(home_team), home_xg, away_xg), (str(away_team), away_xg, home_xg)]:
            self._hist_for.setdefault(team, []).append(float(xf))
            self._hist_against.setdefault(team, []).append(float(xa))
            self._hist_for[team] = self._hist_for[team][-cap:]
            self._hist_against[team] = self._hist_against[team][-cap:]

    def _rates(self, team: str) -> dict[str, float]:
        hf, ha = self._hist_for.get(str(team), []), self._hist_against.get(str(team), [])
        rates = {}
        for w in self.windows:
            rates[f"xg_for_r{w}"] = float(np.mean(hf[-w:])) if len(hf) >= 3 else self._global[f"xg_for_r{w}"]
            rates[f"xg_against_r{w}"] = float(np.mean(ha[-w:])) if len(ha) >= 3 else self._global[f"xg_against_r{w}"]
        return rates

    def predict_lambda(self, home_team: str, away_team: str, neutral: bool = False) -> tuple[float, float]:
        """Return (home_xg_lambda, away_xg_lambda). Falls back to global mean rates
        for teams unseen in training."""
        if self._model is None:
            return 1.5, 1.2
        h, a = self._rates(home_team), self._rates(away_team)
        is_home = 0 if neutral else 1

        def feat_row(own: dict, opp: dict, home_flag: int) -> list[float]:
            row = [own[f"xg_for_r{w}"] for w in self.windows]
            row += [opp[f"xg_against_r{w}"] for w in self.windows]
            row += [home_flag]
            return row

        X = np.array([feat_row(h, a, is_home), feat_row(a, h, 0)], dtype=float)
        pred = np.clip(self._model.predict(X), self.lambda_floor, self.lambda_cap)
        return float(pred[0]), float(pred[1])
