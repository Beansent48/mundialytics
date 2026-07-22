from __future__ import annotations

"""xG-rate lambda predictor — a first-class goal-lambda source built on rolling
pre-match xG form, with walk-forward form state and an optional open-play /
set-piece decomposition.

Backtests (see [[project_xg_modeling_findings]]):
- Base features (rolling xG-for windows 5/10/19 + opponent xG-against + is_home)
  beat the goals-form GoalLambdaModel; the RPS-optimal club blend is
  0.60*xG-rate + 0.40*goals-AttackDefense.
- WALK-FORWARD form (update_form after each played match) is worth ~0.004 RPS
  over freezing form at training end.
- SETPIECE decomposition (own open-play/set-piece xG rates r10 + the opponent's
  own op/sp attack profile) adds ~-0.0002 RPS and -0.0007 BTTS log-loss. It is
  enabled automatically when home_xg_op/home_xg_sp/away_xg_op/away_xg_sp are
  present on the training matches; absent columns reproduce the base behavior.

Rolling features are shifted by one match (leakage-safe). Requires home_xg/away_xg
on the training matches; where xG is absent the engine falls back to the goals path.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

WINDOWS = (5, 10, 19)
SP_WINDOW = 10
SP_COLS = ("home_xg_op", "away_xg_op", "home_xg_sp", "away_xg_sp")


@dataclass
class XGRateModel:
    windows: Sequence[int] = WINDOWS
    alpha: float = 0.1
    lambda_floor: float = 0.05
    lambda_cap: float = 6.0
    # SETPIECE decomposition is GATED OFF by default: the end-to-end validation
    # (validate_setpiece_engine.py) showed only -0.00010 RPS in 4/6 folds with O/U
    # slightly WORSE — below the deploy bar (clear 5-6/6 OOS wins). The batch-level
    # promise (-0.00018/-0.00067) halved once inside the full engine. Flip to True
    # to re-enable; the op/sp data plumbing (foundation columns, shots aggregation)
    # remains in place.
    use_setpiece: bool = False
    _model: PoissonRegressor | None = field(default=None, init=False)
    _feats: list[str] = field(default_factory=list, init=False)
    _has_sp: bool = field(default=False, init=False)
    # Walk-forward form state: per team, recent series (most recent last, capped at
    # max window). predict() reads current form; update_form() appends played
    # matches so mid-season predictions use form AS OF that date.
    _hist_for: dict[str, list[float]] = field(default_factory=dict, init=False)
    _hist_against: dict[str, list[float]] = field(default_factory=dict, init=False)
    _hist_op: dict[str, list[float]] = field(default_factory=dict, init=False)
    _hist_sp: dict[str, list[float]] = field(default_factory=dict, init=False)
    _global: dict[str, float] = field(default_factory=dict, init=False)

    # ── feature engineering ────────────────────────────────────────────────────

    def _long_rows(self, matches: pd.DataFrame) -> pd.DataFrame:
        m = matches.dropna(subset=["home_xg", "away_xg"]).copy()
        self._has_sp = (self.use_setpiece and all(c in m.columns for c in SP_COLS)
                        and m[list(SP_COLS)].notna().all(axis=1).sum() >= 300)
        rows = []
        for _, r in m.iterrows():
            base_h = {"match_id": r.get("match_id"), "date": r.get("date"), "team": r["home_team"],
                      "opp": r["away_team"], "is_home": 1, "xg_for": r["home_xg"], "xg_against": r["away_xg"]}
            base_a = {"match_id": r.get("match_id"), "date": r.get("date"), "team": r["away_team"],
                      "opp": r["home_team"], "is_home": 0, "xg_for": r["away_xg"], "xg_against": r["home_xg"]}
            if self._has_sp:
                base_h.update(xg_op=r.get("home_xg_op"), xg_sp=r.get("home_xg_sp"))
                base_a.update(xg_op=r.get("away_xg_op"), xg_sp=r.get("away_xg_sp"))
            rows.append(base_h); rows.append(base_a)
        lr = pd.DataFrame(rows)
        lr["date"] = pd.to_datetime(lr["date"], errors="coerce")
        return lr

    def _add_rolling(self, lr: pd.DataFrame) -> pd.DataFrame:
        lr = lr.sort_values(["team", "date", "match_id"]).copy()
        cols = {"xg_for": self.windows, "xg_against": self.windows}
        if self._has_sp:
            cols.update({"xg_op": (SP_WINDOW,), "xg_sp": (SP_WINDOW,)})
        for col, wins in cols.items():
            lr[col] = pd.to_numeric(lr[col], errors="coerce")
            for w in wins:
                lr[f"{col}_r{w}"] = (lr.groupby("team", group_keys=False)[col]
                                     .apply(lambda s: s.shift(1).rolling(w, min_periods=3).mean()))
        opp_src = [f"xg_against_r{w}" for w in self.windows]
        if self._has_sp:
            opp_src += [f"xg_op_r{SP_WINDOW}", f"xg_sp_r{SP_WINDOW}"]
        opp = lr[["match_id", "team"] + opp_src].rename(
            columns={"team": "opp", **{c: f"opp_{c}" for c in opp_src}})
        return lr.merge(opp, on=["match_id", "opp"], how="left")

    def _feature_list(self) -> list[str]:
        feats = [f"xg_for_r{w}" for w in self.windows]
        if self._has_sp:
            feats += [f"xg_op_r{SP_WINDOW}", f"xg_sp_r{SP_WINDOW}"]
        feats += [f"opp_xg_against_r{w}" for w in self.windows]
        if self._has_sp:
            feats += [f"opp_xg_op_r{SP_WINDOW}", f"opp_xg_sp_r{SP_WINDOW}"]
        return feats + ["is_home"]

    # ── fit / predict ──────────────────────────────────────────────────────────

    def fit(self, matches: pd.DataFrame) -> "XGRateModel":
        if not {"home_xg", "away_xg"}.issubset(matches.columns):
            return self
        # Guard the common case where the xG columns exist but are all NaN (pre-2014
        # rows, uncovered leagues) — the model stays not-ready and the engine falls back.
        if len(matches.dropna(subset=["home_xg", "away_xg"])) < 300:
            return self
        lr = self._add_rolling(self._long_rows(matches))
        self._feats = self._feature_list()
        train = lr.dropna(subset=self._feats + ["xg_for"])
        if len(train) < 300:
            self._model = None
            return self
        self._model = PoissonRegressor(alpha=self.alpha, max_iter=1000).fit(
            train[self._feats], train["xg_for"].clip(lower=0))
        # Global fallback rates + walk-forward form state seeded from training.
        self._global = {"xg_for": float(pd.to_numeric(lr["xg_for"], errors="coerce").median()),
                        "xg_against": float(pd.to_numeric(lr["xg_against"], errors="coerce").median())}
        if self._has_sp:
            self._global["xg_op"] = float(pd.to_numeric(lr["xg_op"], errors="coerce").median())
            self._global["xg_sp"] = float(pd.to_numeric(lr["xg_sp"], errors="coerce").median())
        cap = max(self.windows)
        for team, g in lr.sort_values("date").groupby("team"):
            t = str(team)
            self._hist_for[t] = pd.to_numeric(g["xg_for"], errors="coerce").dropna().tolist()[-cap:]
            self._hist_against[t] = pd.to_numeric(g["xg_against"], errors="coerce").dropna().tolist()[-cap:]
            if self._has_sp:
                self._hist_op[t] = pd.to_numeric(g["xg_op"], errors="coerce").dropna().tolist()[-cap:]
                self._hist_sp[t] = pd.to_numeric(g["xg_sp"], errors="coerce").dropna().tolist()[-cap:]
        return self

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def update_form(self, home_team: str, away_team: str, home_xg: float, away_xg: float,
                    home_xg_op: float | None = None, away_xg_op: float | None = None,
                    home_xg_sp: float | None = None, away_xg_sp: float | None = None) -> None:
        """Append a played match to the walk-forward form state (call after each match
        in chronological order). op/sp values are optional; when omitted, the op/sp
        state simply keeps its last-known window (slight staleness, no error)."""
        cap = max(self.windows)
        for team, xf, xa, xo, xs in [(str(home_team), home_xg, away_xg, home_xg_op, home_xg_sp),
                                     (str(away_team), away_xg, home_xg, away_xg_op, away_xg_sp)]:
            self._hist_for.setdefault(team, []).append(float(xf))
            self._hist_against.setdefault(team, []).append(float(xa))
            self._hist_for[team] = self._hist_for[team][-cap:]
            self._hist_against[team] = self._hist_against[team][-cap:]
            if self._has_sp and xo is not None and xs is not None:
                self._hist_op.setdefault(team, []).append(float(xo))
                self._hist_sp.setdefault(team, []).append(float(xs))
                self._hist_op[team] = self._hist_op[team][-cap:]
                self._hist_sp[team] = self._hist_sp[team][-cap:]

    def _rate(self, hist: list[float], w: int, fallback: float) -> float:
        return float(np.mean(hist[-w:])) if len(hist) >= 3 else fallback

    def predict_lambda(self, home_team: str, away_team: str, neutral: bool = False) -> tuple[float, float]:
        """Return (home_xg_lambda, away_xg_lambda) from current walk-forward form."""
        if self._model is None:
            return 1.5, 1.2
        h, a = str(home_team), str(away_team)
        is_home = 0 if neutral else 1

        def feat_row(own: str, opp: str, home_flag: int) -> list[float]:
            row = [self._rate(self._hist_for.get(own, []), w, self._global["xg_for"]) for w in self.windows]
            if self._has_sp:
                row += [self._rate(self._hist_op.get(own, []), SP_WINDOW, self._global["xg_op"]),
                        self._rate(self._hist_sp.get(own, []), SP_WINDOW, self._global["xg_sp"])]
            row += [self._rate(self._hist_against.get(opp, []), w, self._global["xg_against"]) for w in self.windows]
            if self._has_sp:
                row += [self._rate(self._hist_op.get(opp, []), SP_WINDOW, self._global["xg_op"]),
                        self._rate(self._hist_sp.get(opp, []), SP_WINDOW, self._global["xg_sp"])]
            return row + [home_flag]

        X = np.array([feat_row(h, a, is_home), feat_row(a, h, 0)], dtype=float)
        pred = np.clip(self._model.predict(X), self.lambda_floor, self.lambda_cap)
        return float(pred[0]), float(pred[1])
