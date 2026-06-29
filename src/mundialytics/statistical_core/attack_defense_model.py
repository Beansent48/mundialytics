"""
Maher (1982) / Dixon-Coles (1997) double-Poisson model with MLE-fitted
attack and defense parameters per team.

Model:
    log(λ_home) = μ + home_adv + attack[home] - defense[away]
    log(λ_away) = μ           + attack[away]  - defense[home]

Identifiability constraint: attack of the reference team is fixed to 0.

Extensions vs the heuristic MatchOutcomeModel:
  - Parameters are fitted jointly via MLE, not computed as simple ratios.
  - Time-decay weights down-weight old matches exponentially.
  - Dixon-Coles rho corrects the 0-0 / 1-0 / 0-1 / 1-1 cells.
  - Teams with no history fall back to global mean (attack=0, defense=0).
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from mundialytics.statistical_core.calibration import recency_weights
from mundialytics.statistical_core.distributions import (
    outcome_probabilities,
    scoreline_distribution,
)
from mundialytics.statistical_core.schemas import canonical_name, standardize_fixtures


def _dc_tau(hg: int, ag: int, lh: float, la: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor."""
    if abs(rho) < 1e-12:
        return 1.0
    if hg == 0 and ag == 0:
        return 1.0 - rho * lh * la
    if hg == 0 and ag == 1:
        return 1.0 + rho * lh
    if hg == 1 and ag == 0:
        return 1.0 + rho * la
    if hg == 1 and ag == 1:
        return 1.0 - rho
    return 1.0


class AttackDefenseModel:
    """MLE double-Poisson model (Maher 1982 / Dixon-Coles 1997).

    Produces the same ``predict_fixtures`` interface as ``MatchOutcomeModel``
    so the two models are interchangeable in downstream pipelines.

    Parameters
    ----------
    dixon_coles_rho : float
        Low-score correction (negative boosts draws, default -0.07).
    time_decay_half_life : float
        Exponential decay half-life in days. None = no decay.
    max_goals : int
        Grid size for the scoreline matrix.
    goal_floor, goal_cap : float
        Safety bounds for predicted lambdas.
    home_advantage_init : float
        Starting value for the home advantage in log-space (exp(0.1)≈1.10).
    min_matches : int
        Teams with fewer than this number of matches will be flagged.
    """

    def __init__(
        self,
        dixon_coles_rho: float = -0.07,
        time_decay_half_life: float | None = 365.0,
        max_goals: int = 10,
        goal_floor: float = 0.05,
        goal_cap: float = 6.0,
        home_advantage_init: float = 0.10,
        min_matches: int = 5,
    ):
        self.dixon_coles_rho = float(dixon_coles_rho)
        self.time_decay_half_life = time_decay_half_life
        self.max_goals = int(max_goals)
        self.goal_floor = float(goal_floor)
        self.goal_cap = float(goal_cap)
        self.home_advantage_init = float(home_advantage_init)
        self.min_matches = int(min_matches)

        self.teams_: list[str] = []
        self.team_index_: dict[str, int] = {}
        self.n_teams_: int = 0

        # Fitted parameters (log-scale)
        self.mu_: float = 0.0          # global intercept
        self.home_adv_: float = 0.0    # log home advantage
        self.attack_: np.ndarray = np.array([])   # per-team, shape (n,)
        self.defense_: np.ndarray = np.array([])  # per-team, shape (n,)

        self.global_goal_mean_: float = 1.25
        self.match_counts_: dict[str, int] = {}
        self.fit_result_: dict[str, Any] = {}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _pack(self, mu: float, home_adv: float, attacks: np.ndarray, defenses: np.ndarray) -> np.ndarray:
        # attacks[0] is the reference team (fixed to 0); we only optimise [1:]
        return np.concatenate([[mu, home_adv], attacks[1:], defenses])

    def _unpack(self, x: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        mu = x[0]
        home_adv = x[1]
        attacks = np.concatenate([[0.0], x[2:2 + self.n_teams_ - 1]])
        defenses = x[2 + self.n_teams_ - 1:]
        return mu, home_adv, attacks, defenses

    def _neg_log_likelihood(
        self,
        x: np.ndarray,
        hg: np.ndarray,
        ag: np.ndarray,
        hi: np.ndarray,
        ai: np.ndarray,
        w: np.ndarray,
    ) -> float:
        mu, home_adv, attacks, defenses = self._unpack(x)
        lh = np.exp(mu + home_adv + attacks[hi] - defenses[ai])
        la = np.exp(mu + attacks[ai] - defenses[hi])
        lh = np.clip(lh, 1e-6, 20.0)
        la = np.clip(la, 1e-6, 20.0)

        ll = np.sum(w * (hg * np.log(lh) - lh + ag * np.log(la) - la))

        # Dixon-Coles correction
        rho = self.dixon_coles_rho
        if abs(rho) > 1e-12:
            for k in range(len(hg)):
                h, a = int(hg[k]), int(ag[k])
                if h <= 1 and a <= 1:
                    tau = _dc_tau(h, a, lh[k], la[k], rho)
                    tau = max(tau, 1e-10)
                    ll += w[k] * math.log(tau)

        return -ll

    def _lam(self, home_idx: int, away_idx: int, neutral: bool) -> tuple[float, float]:
        ha = 0.0 if neutral else self.home_adv_
        lh = math.exp(self.mu_ + ha + self.attack_[home_idx] - self.defense_[away_idx])
        la = math.exp(self.mu_ + self.attack_[away_idx] - self.defense_[home_idx])
        return (
            float(np.clip(lh, self.goal_floor, self.goal_cap)),
            float(np.clip(la, self.goal_floor, self.goal_cap)),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, matches: pd.DataFrame) -> "AttackDefenseModel":
        """Fit on a match-level DataFrame with home_team, away_team, home_goals, away_goals, date."""
        df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        df["home_team"] = df["home_team"].map(canonical_name)
        df["away_team"] = df["away_team"].map(canonical_name)
        df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").fillna(0).astype(int)
        df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").fillna(0).astype(int)
        df["date"] = pd.to_datetime(df.get("date"), errors="coerce")

        all_teams = sorted(set(df["home_team"].tolist()) | set(df["away_team"].tolist()))
        self.teams_ = all_teams
        self.team_index_ = {t: i for i, t in enumerate(all_teams)}
        self.n_teams_ = len(all_teams)

        # Match counts for flagging low-sample teams
        for team in all_teams:
            cnt = ((df["home_team"] == team) | (df["away_team"] == team)).sum()
            self.match_counts_[team] = int(cnt)

        self.global_goal_mean_ = float(
            (df["home_goals"].mean() + df["away_goals"].mean()) / 2
        )

        # Arrays for optimisation
        hi = df["home_team"].map(self.team_index_).values.astype(int)
        ai = df["away_team"].map(self.team_index_).values.astype(int)
        hg = df["home_goals"].values.astype(float)
        ag = df["away_goals"].values.astype(float)

        # Time-decay weights
        if self.time_decay_half_life and "date" in df.columns:
            w = recency_weights(df["date"], half_life_days=float(self.time_decay_half_life))
        else:
            w = np.ones(len(df), dtype=float)
        w = w / w.mean()  # normalise so scale doesn't change with dataset size

        # Initial parameters
        mu0 = math.log(max(self.global_goal_mean_, 0.1))
        x0 = np.zeros(1 + 1 + (self.n_teams_ - 1) + self.n_teams_)
        x0[0] = mu0
        x0[1] = self.home_advantage_init

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(hg, ag, hi, ai, w),
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        self.mu_, self.home_adv_, self.attack_, self.defense_ = self._unpack(result.x)
        self.fit_result_ = {
            "success": bool(result.success),
            "nit": int(result.nit),
            "nll": float(result.fun),
            "n_teams": self.n_teams_,
            "n_matches": len(df),
            "global_goal_mean": round(self.global_goal_mean_, 4),
            "home_adv_log": round(float(self.home_adv_), 4),
            "home_adv_factor": round(math.exp(float(self.home_adv_)), 4),
            "model": "attack_defense_mle_v1",
        }
        return self

    def team_params(self) -> pd.DataFrame:
        """Return a sorted DataFrame with attack, defense, and net strength per team."""
        rows = []
        for team in self.teams_:
            i = self.team_index_[team]
            rows.append({
                "team": team,
                "attack": round(float(self.attack_[i]), 4),
                "defense": round(float(self.defense_[i]), 4),
                "net_strength": round(float(self.attack_[i] - self.defense_[i]), 4),
                "matches": self.match_counts_.get(team, 0),
            })
        return pd.DataFrame(rows).sort_values("net_strength", ascending=False).reset_index(drop=True)

    def expected_goals(self, home_team: str, away_team: str, neutral: int | bool = 0) -> tuple[float, float, list[str]]:
        ht = canonical_name(home_team)
        at = canonical_name(away_team)
        warnings: list[str] = []
        hi = self.team_index_.get(ht)
        ai = self.team_index_.get(at)
        if hi is None:
            warnings.append(f"unknown_home_team:{ht}")
            hi = 0
        if ai is None:
            warnings.append(f"unknown_away_team:{at}")
            ai = 0
        for team, idx in [(ht, hi), (at, ai)]:
            if self.match_counts_.get(team, 0) < self.min_matches:
                warnings.append(f"low_sample:{team}({self.match_counts_.get(team,0)})")
        lh, la = self._lam(hi, ai, bool(int(neutral)))
        return lh, la, warnings

    def predict_fixtures(self, fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Same interface as MatchOutcomeModel.predict_fixtures."""
        f = standardize_fixtures(fixtures)
        rows: list[dict[str, Any]] = []
        score_rows: list[pd.DataFrame] = []
        for _, r in f.iterrows():
            lh, la, warnings = self.expected_goals(r["home_team"], r["away_team"], r.get("neutral", 0))
            probs = outcome_probabilities(lh, la, max_goals=self.max_goals, dixon_coles_rho=self.dixon_coles_rho)
            dist = scoreline_distribution(lh, la, max_goals=self.max_goals, normalize=True, dixon_coles_rho=self.dixon_coles_rho)
            top = dist.top_scorelines(8)
            score_frame = dist.to_long_frame(match_id=str(r["match_id"]))
            score_rows.append(score_frame)
            row = r.to_dict()
            row.update({
                "lambda_home": lh,
                "lambda_away": la,
                "expected_home_goals": lh,
                "expected_away_goals": la,
                "home_attack": round(float(self.attack_[self.team_index_.get(canonical_name(r["home_team"]), 0)]), 4),
                "away_attack": round(float(self.attack_[self.team_index_.get(canonical_name(r["away_team"]), 0)]), 4),
                "home_defense": round(float(self.defense_[self.team_index_.get(canonical_name(r["home_team"]), 0)]), 4),
                "away_defense": round(float(self.defense_[self.team_index_.get(canonical_name(r["away_team"]), 0)]), 4),
                "model_type": "attack_defense_mle_v1",
                "scoreline_distribution_json": json.dumps(top, ensure_ascii=False),
                "warnings": ";".join(warnings),
            })
            row.update(probs)
            rows.append(row)
        score_df = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
        return pd.DataFrame(rows), score_df
