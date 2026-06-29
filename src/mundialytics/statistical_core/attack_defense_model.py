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
        l2_reg: float = 0.01,
    ):
        self.dixon_coles_rho = float(dixon_coles_rho)
        self.time_decay_half_life = time_decay_half_life
        self.max_goals = int(max_goals)
        self.goal_floor = float(goal_floor)
        self.goal_cap = float(goal_cap)
        self.home_advantage_init = float(home_advantage_init)
        self.min_matches = int(min_matches)
        self.l2_reg = float(l2_reg)

        self.teams_: list[str] = []
        self.team_index_: dict[str, int] = {}
        self.n_teams_: int = 0

        self.leagues_: list[str] = []
        self.league_index_: dict[str, int] = {}
        self.n_leagues_: int = 0

        # Fitted parameters (log-scale)
        self.mu_: float = 0.0             # global intercept
        self.home_adv_: float = 0.0       # log home advantage
        self.league_effect_: np.ndarray = np.array([])  # per-league, shape (L,); [0]=0
        self.attack_: np.ndarray = np.array([])          # per-team, shape (N,); [0]=0
        self.defense_: np.ndarray = np.array([])         # per-team, shape (N,)

        self.global_goal_mean_: float = 1.25
        self.match_counts_: dict[str, int] = {}
        self.fit_result_: dict[str, Any] = {}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _unpack(self, x: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        mu = x[0]
        home_adv = x[1]
        nL = self.n_leagues_
        nN = self.n_teams_
        league_effects = np.concatenate([[0.0], x[2: 2 + nL - 1]])
        attacks = np.concatenate([[0.0], x[2 + nL - 1: 2 + nL - 1 + nN - 1]])
        defenses = x[2 + nL - 1 + nN - 1:]
        return mu, home_adv, league_effects, attacks, defenses

    def _neg_log_likelihood(
        self,
        x: np.ndarray,
        hg: np.ndarray,
        ag: np.ndarray,
        hi: np.ndarray,
        ai: np.ndarray,
        li: np.ndarray,
        w: np.ndarray,
    ) -> float:
        mu, home_adv, league_effects, attacks, defenses = self._unpack(x)
        leff = league_effects[li]
        lh = np.exp(mu + home_adv + leff + attacks[hi] - defenses[ai])
        la = np.exp(mu + leff + attacks[ai] - defenses[hi])
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

        # L2 regularization — shrinks attack, defense, league effects toward zero
        if self.l2_reg > 0:
            ll -= self.l2_reg * (
                np.sum(attacks[1:] ** 2)
                + np.sum(defenses ** 2)
                + np.sum(league_effects[1:] ** 2)
            )

        return -ll

    def _lam(self, home_idx: int, away_idx: int, league_idx: int, neutral: bool) -> tuple[float, float]:
        ha = 0.0 if neutral else self.home_adv_
        # league_effect_ stores per-league mu (intercept) from per-league fit
        league_mu = float(self.league_effect_[league_idx]) if league_idx < len(self.league_effect_) else self.mu_
        lh = math.exp(league_mu + ha + self.attack_[home_idx] - self.defense_[away_idx])
        la = math.exp(league_mu + self.attack_[away_idx] - self.defense_[home_idx])
        return (
            float(np.clip(lh, self.goal_floor, self.goal_cap)),
            float(np.clip(la, self.goal_floor, self.goal_cap)),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def _fit_single(self, df: pd.DataFrame) -> None:
        """Fit on one competition's matches. Assumes df is already cleaned."""
        all_teams = sorted(set(df["home_team"].tolist()) | set(df["away_team"].tolist()))
        self.teams_ = all_teams
        self.team_index_ = {t: i for i, t in enumerate(all_teams)}
        self.n_teams_ = len(all_teams)
        self.leagues_ = []
        self.league_index_ = {}
        self.n_leagues_ = 1  # single competition, no league effects
        self.league_effect_ = np.zeros(1)

        for team in all_teams:
            cnt = ((df["home_team"] == team) | (df["away_team"] == team)).sum()
            self.match_counts_[team] = int(cnt)

        self.global_goal_mean_ = float((df["home_goals"].mean() + df["away_goals"].mean()) / 2)

        hi = df["home_team"].map(self.team_index_).values.astype(int)
        ai = df["away_team"].map(self.team_index_).values.astype(int)
        li = np.zeros(len(df), dtype=int)   # single league → index 0
        hg = df["home_goals"].values.astype(float)
        ag = df["away_goals"].values.astype(float)

        if self.time_decay_half_life and "date" in df.columns:
            w = recency_weights(df["date"], half_life_days=float(self.time_decay_half_life))
        else:
            w = np.ones(len(df), dtype=float)
        w = w / w.mean()

        # x = [μ, home_adv, attacks[1..N-1], defenses[0..N-1]]
        # (no league effects for single-league fit)
        n_params = 1 + 1 + (self.n_teams_ - 1) + self.n_teams_
        x0 = np.zeros(n_params)
        x0[0] = math.log(max(self.global_goal_mean_, 0.1))
        x0[1] = self.home_advantage_init

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(hg, ag, hi, ai, li, w),
            method="L-BFGS-B",
            options={"maxiter": 5000, "maxfun": 100_000, "ftol": 1e-9, "gtol": 1e-4},
        )

        self.mu_, self.home_adv_, self.league_effect_, self.attack_, self.defense_ = self._unpack(result.x)
        return result

    def fit(self, matches: pd.DataFrame) -> "AttackDefenseModel":
        """Fit per-competition models and merge into a single parameter store.

        When competition column is present, each league is fit independently so
        that attack/defense parameters are calibrated within their own competition.
        This avoids the identifiability problem in multi-league joint estimation
        (without cross-league matches there is no anchor for inter-league scaling).

        Produces a unified team_index_ and parameter arrays where parameters from
        each league are placed without interference.
        """
        df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        df["home_team"] = df["home_team"].map(canonical_name)
        df["away_team"] = df["away_team"].map(canonical_name)
        df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").fillna(0).astype(int)
        df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").fillna(0).astype(int)
        df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
        if "competition" not in df.columns:
            df["competition"] = "unknown"
        df["competition"] = df["competition"].fillna("unknown")

        competitions = sorted(df["competition"].unique())
        self.leagues_ = competitions
        self.league_index_ = {c: i for i, c in enumerate(competitions)}
        self.n_leagues_ = len(competitions)

        # Collect all teams and build a global index
        all_teams = sorted(set(df["home_team"].tolist()) | set(df["away_team"].tolist()))
        self.teams_ = all_teams
        self.team_index_ = {t: i for i, t in enumerate(all_teams)}
        self.n_teams_ = len(all_teams)
        self.attack_ = np.zeros(self.n_teams_)
        self.defense_ = np.zeros(self.n_teams_)
        self.league_effect_ = np.zeros(self.n_leagues_)
        self.global_goal_mean_ = float((df["home_goals"].mean() + df["away_goals"].mean()) / 2)

        for team in all_teams:
            cnt = ((df["home_team"] == team) | (df["away_team"] == team)).sum()
            self.match_counts_[team] = int(cnt)

        per_league_results: dict[str, Any] = {}
        for comp in competitions:
            sub = df[df["competition"] == comp].copy()
            if len(sub) < 10:
                continue
            m_sub = AttackDefenseModel(
                dixon_coles_rho=self.dixon_coles_rho,
                time_decay_half_life=self.time_decay_half_life,
                goal_floor=self.goal_floor,
                goal_cap=self.goal_cap,
                home_advantage_init=self.home_advantage_init,
                min_matches=self.min_matches,
                l2_reg=self.l2_reg,
            )
            result_sub = m_sub._fit_single(sub)
            # Copy league-level params
            league_mu = m_sub.mu_
            league_home_adv = m_sub.home_adv_
            league_idx = self.league_index_[comp]
            # Store per-league intercept as league_effect (relative to global mean)
            self.league_effect_[league_idx] = league_mu
            # Copy team attack/defense into global arrays
            for team, local_idx in m_sub.team_index_.items():
                global_idx = self.team_index_.get(team)
                if global_idx is not None:
                    self.attack_[global_idx] = m_sub.attack_[local_idx]
                    self.defense_[global_idx] = m_sub.defense_[local_idx]
            per_league_results[comp] = {
                "success": bool(result_sub.success),
                "n_teams": m_sub.n_teams_,
                "n_matches": len(sub),
                "home_adv_factor": round(math.exp(league_home_adv), 3),
                "goal_mean": round(m_sub.global_goal_mean_, 3),
            }

        # Use the mean home advantage across leagues
        home_advs = [v["home_adv_factor"] for v in per_league_results.values()]
        mean_ha_factor = float(np.mean(home_advs)) if home_advs else math.exp(self.home_advantage_init)
        self.home_adv_ = math.log(max(mean_ha_factor, 0.5))
        self.mu_ = math.log(max(self.global_goal_mean_, 0.1))

        self.fit_result_ = {
            "success": all(v["success"] for v in per_league_results.values()),
            "n_teams": self.n_teams_,
            "n_leagues": self.n_leagues_,
            "n_matches": len(df),
            "global_goal_mean": round(self.global_goal_mean_, 4),
            "home_adv_factor": round(mean_ha_factor, 4),
            "per_league": per_league_results,
            "model": "attack_defense_mle_v3_per_league",
        }
        return self

    def team_params(self) -> pd.DataFrame:
        """Return a sorted DataFrame of per-team parameters.

        ``strength`` = attack + defense (log-scale sum): both positive means you
        score AND prevent goals above average — the best overall quality metric.
        ``attack_balance`` = attack - defense: positive means offensive profile
        (score more than you defend), negative means defensive profile.
        """
        rows = []
        for team in self.teams_:
            i = self.team_index_[team]
            a, d = float(self.attack_[i]), float(self.defense_[i])
            rows.append({
                "team": team,
                "attack": round(a, 4),
                "defense": round(d, 4),
                "strength": round(a + d, 4),       # overall quality
                "attack_balance": round(a - d, 4), # offensive vs defensive profile
                "matches": self.match_counts_.get(team, 0),
            })
        return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)

    def expected_goals(
        self,
        home_team: str,
        away_team: str,
        neutral: int | bool = 0,
        competition: str | None = None,
    ) -> tuple[float, float, list[str]]:
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
        for team in [ht, at]:
            if self.match_counts_.get(team, 0) < self.min_matches:
                warnings.append(f"low_sample:{team}({self.match_counts_.get(team,0)})")
        league_idx = self.league_index_.get(competition or "", 0)
        lh, la = self._lam(hi, ai, league_idx, bool(int(neutral)))
        return lh, la, warnings

    def predict_fixtures(self, fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Same interface as MatchOutcomeModel.predict_fixtures."""
        f = standardize_fixtures(fixtures)
        rows: list[dict[str, Any]] = []
        score_rows: list[pd.DataFrame] = []
        for _, r in f.iterrows():
            lh, la, warnings = self.expected_goals(
                r["home_team"], r["away_team"], r.get("neutral", 0), competition=r.get("competition")
            )
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
