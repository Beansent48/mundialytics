from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mundialytics.statistical_core.schemas import canonical_name


POSITION_GOAL_CONVERSION = {
    "ST": 0.145,
    "CF": 0.135,
    "LW": 0.105,
    "RW": 0.105,
    "AM": 0.095,
    "CAM": 0.095,
    "CM": 0.065,
    "DM": 0.045,
    "LB": 0.035,
    "RB": 0.035,
    "CB": 0.030,
    "GK": 0.001,
    "UNK": 0.085,
}


@dataclass(frozen=True)
class ScorerForecastConfig:
    n_simulations: int = 5000
    seed: int = 42
    default_shot_conversion: float = 0.10
    survival_match_weight: float = 0.65


class CompetitionForecastEngine:
    """Football-meets-data style narrative outputs from the statistical core.

    This module does not pretend to know awards that require unavailable data
    (keeper shot-stopping, age/breakout criteria, assists). It produces
    transparent approximations and marks unsupported awards as not_available.
    """

    def __init__(self, config: ScorerForecastConfig | None = None):
        self.config = config or ScorerForecastConfig()
        self.audit: dict[str, Any] = {}

    def build_outputs(
        self,
        player_event_predictions: pd.DataFrame | None,
        tournament_simulation: pd.DataFrame | None,
        match_predictions: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        top = self.top_scorer_forecast(player_event_predictions, tournament_simulation)
        awards = self.award_forecast(top, player_event_predictions, tournament_simulation)
        competition = self.competition_summary(tournament_simulation, match_predictions)
        self.audit = {
            "version": "v0.22_competition_forecast_engine",
            "top_scorer_rows": int(len(top)),
            "award_rows": int(len(awards)),
            "competition_rows": int(len(competition)),
            "experimental": [
                "Top scorer probabilities use player shot volume converted to approximate goals, not a dedicated player xG model yet.",
                "Best player is an approximate attacking-impact ranking, not an official award model.",
                "Best goalkeeper and breakout predictions are not available without goalkeeper/age-specific data.",
            ],
        }
        return top, awards, competition

    def top_scorer_forecast(self, player_event_predictions: pd.DataFrame | None, tournament_simulation: pd.DataFrame | None) -> pd.DataFrame:
        if player_event_predictions is None or player_event_predictions.empty:
            return pd.DataFrame()
        p = player_event_predictions.copy()
        p = p[p["market"].astype(str).eq("player_shots")].copy()
        if p.empty:
            return pd.DataFrame()
        for col in ["player", "team", "position", "expected_count", "expected_minutes", "sample_size_minutes", "confidence_flag", "warnings"]:
            if col not in p.columns:
                p[col] = "" if col in {"player", "team", "position", "confidence_flag", "warnings"} else 0.0
        p["team"] = p["team"].map(canonical_name)
        p["player"] = p["player"].map(canonical_name)
        p["shots_expected_current_fixtures"] = pd.to_numeric(p["expected_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
        p["position_key"] = p["position"].astype(str).str.upper().str.strip().replace({"": "UNK"})
        p["shot_conversion_assumption"] = p["position_key"].map(POSITION_GOAL_CONVERSION).fillna(self.config.default_shot_conversion).astype(float)
        p["expected_goals_current_fixtures"] = p["shots_expected_current_fixtures"] * p["shot_conversion_assumption"]
        agg_cols = {
            "shots_expected_current_fixtures": "sum",
            "expected_goals_current_fixtures": "sum",
            "sample_size_minutes": "max",
            "expected_minutes": "sum",
            "shot_conversion_assumption": "mean",
        }
        by = p.groupby(["player", "team"], dropna=False).agg(agg_cols).reset_index()
        by["team_survival_bonus"] = by["team"].map(_team_survival_bonus(tournament_simulation, self.config.survival_match_weight)).fillna(1.0)
        by["expected_tournament_goals_approx"] = by["expected_goals_current_fixtures"] * by["team_survival_bonus"]
        by["top_scorer_probability"] = _simulate_top_scorer_probability(
            by["expected_tournament_goals_approx"].to_numpy(dtype=float),
            n_simulations=self.config.n_simulations,
            seed=self.config.seed,
        )
        by["confidence"] = np.where(pd.to_numeric(by["sample_size_minutes"], errors="coerce").fillna(0) >= 270, "medium", "low")
        by["model_type"] = "v022_shots_to_goals_top_scorer_monte_carlo"
        by["warnings"] = np.where(by["sample_size_minutes"].astype(float) < 270, "low_player_sample_for_top_scorer", "goals_approximated_from_shots")
        return by.sort_values(["top_scorer_probability", "expected_tournament_goals_approx"], ascending=False).reset_index(drop=True)

    def award_forecast(
        self,
        top_scorers: pd.DataFrame | None,
        player_event_predictions: pd.DataFrame | None,
        tournament_simulation: pd.DataFrame | None,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        if top_scorers is not None and not top_scorers.empty:
            for _, r in top_scorers.head(20).iterrows():
                rows.append(
                    {
                        "award": "golden_boot",
                        "candidate": r["player"],
                        "team": r["team"],
                        "probability_approx": float(r["top_scorer_probability"]),
                        "score": float(r["expected_tournament_goals_approx"]),
                        "availability": "approx_available",
                        "method": "monte_carlo_poisson_from_shots",
                        "warnings": "experimental_goals_from_shots",
                    }
                )
        impact = _attacking_impact_table(player_event_predictions, tournament_simulation)
        if not impact.empty:
            total = float(impact["impact_score"].sum())
            impact["probability_approx"] = impact["impact_score"] / total if total > 0 else 0.0
            for _, r in impact.head(20).iterrows():
                rows.append(
                    {
                        "award": "best_player_approx",
                        "candidate": r["player"],
                        "team": r["team"],
                        "probability_approx": float(r["probability_approx"]),
                        "score": float(r["impact_score"]),
                        "availability": "approx_available",
                        "method": "attacking_volume_plus_team_progression",
                        "warnings": "not_official_award_model_no_assists_or_defensive_value",
                    }
                )
        rows.extend(
            [
                {
                    "award": "best_goalkeeper_clean_sheets",
                    "candidate": "not_available",
                    "team": "not_available",
                    "probability_approx": np.nan,
                    "score": np.nan,
                    "availability": "not_available",
                    "method": "requires_goalkeeper_and_clean_sheet_data",
                    "warnings": "not_enough_goalkeeper_specific_data",
                },
                {
                    "award": "breakout_player",
                    "candidate": "not_available",
                    "team": "not_available",
                    "probability_approx": np.nan,
                    "score": np.nan,
                    "availability": "not_available",
                    "method": "requires_age_or_prior_minutes_baseline",
                    "warnings": "no_age_or_breakout_criteria_available",
                },
            ]
        )
        return pd.DataFrame(rows)

    def competition_summary(self, tournament_simulation: pd.DataFrame | None, match_predictions: pd.DataFrame | None) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        if tournament_simulation is not None and not tournament_simulation.empty:
            ts = tournament_simulation.copy()
            for _, r in ts.sort_values("champion_probability", ascending=False).iterrows():
                rows.append(
                    {
                        "record_type": "team_power_ranking",
                        "team": r.get("team"),
                        "ranking_score": float(r.get("champion_probability", 0)) + 0.35 * float(r.get("qualify_group_probability", 0)),
                        "champion_probability": float(r.get("champion_probability", 0)),
                        "qualify_group_probability": float(r.get("qualify_group_probability", 0)),
                        "expected_points": float(r.get("expected_points", 0)),
                        "expected_goals_for": float(r.get("expected_goals_for", 0)),
                        "headline": _team_headline(r),
                    }
                )
        if match_predictions is not None and not match_predictions.empty:
            mp = match_predictions.copy()
            mp["total_expected_goals"] = pd.to_numeric(mp["lambda_home"], errors="coerce").fillna(0) + pd.to_numeric(mp["lambda_away"], errors="coerce").fillna(0)
            for _, r in mp.sort_values("total_expected_goals", ascending=False).head(20).iterrows():
                rows.append(
                    {
                        "record_type": "match_goal_environment",
                        "team": f"{r.get('home_team')} vs {r.get('away_team')}",
                        "ranking_score": float(r.get("total_expected_goals", 0)),
                        "champion_probability": np.nan,
                        "qualify_group_probability": np.nan,
                        "expected_points": np.nan,
                        "expected_goals_for": float(r.get("total_expected_goals", 0)),
                        "headline": f"{r.get('home_team')} vs {r.get('away_team')}: {float(r.get('total_expected_goals', 0)):.2f} expected goals, most likely {r.get('most_likely_score')}",
                    }
                )
        return pd.DataFrame(rows)


def _team_survival_bonus(tournament_simulation: pd.DataFrame | None, survival_weight: float) -> dict[str, float]:
    if tournament_simulation is None or tournament_simulation.empty:
        return {}
    out = {}
    for _, r in tournament_simulation.iterrows():
        team = canonical_name(r.get("team"))
        # Current fixtures are already represented in player props. This adds a
        # transparent expected future-match bonus from progression probabilities.
        future = sum(float(r.get(col, 0.0)) for col in ["qf_probability", "sf_probability", "final_probability", "champion_probability"])
        out[team] = 1.0 + float(survival_weight) * future
    return out


def _simulate_top_scorer_probability(lambdas: np.ndarray, n_simulations: int, seed: int) -> np.ndarray:
    if lambdas.size == 0:
        return np.array([], dtype=float)
    lam = np.clip(np.asarray(lambdas, dtype=float), 0.0, 12.0)
    if float(lam.sum()) <= 0:
        return np.zeros(len(lam), dtype=float)
    rng = np.random.default_rng(seed)
    wins = np.zeros(len(lam), dtype=float)
    sims = max(int(n_simulations), 100)
    for _ in range(sims):
        goals = rng.poisson(lam)
        top = np.flatnonzero(goals == goals.max())
        wins[top] += 1.0 / len(top)
    probs = wins / sims
    return probs / probs.sum() if probs.sum() > 0 else probs


def _attacking_impact_table(player_event_predictions: pd.DataFrame | None, tournament_simulation: pd.DataFrame | None) -> pd.DataFrame:
    if player_event_predictions is None or player_event_predictions.empty:
        return pd.DataFrame()
    p = player_event_predictions.copy()
    p["team"] = p["team"].map(canonical_name)
    p["player"] = p["player"].map(canonical_name)
    pivot = p.pivot_table(index=["player", "team"], columns="market", values="expected_count", aggfunc="sum", fill_value=0).reset_index()
    for c in ["player_shots", "player_shots_on_target", "player_fouls_committed"]:
        if c not in pivot.columns:
            pivot[c] = 0.0
    bonus = _team_survival_bonus(tournament_simulation, survival_weight=0.5)
    pivot["team_bonus"] = pivot["team"].map(bonus).fillna(1.0)
    pivot["impact_score"] = pivot["team_bonus"] * (0.55 * pivot["player_shots"] + 0.35 * pivot["player_shots_on_target"] + 0.10 * pivot["player_fouls_committed"])
    return pivot.sort_values("impact_score", ascending=False).reset_index(drop=True)


def _team_headline(row: pd.Series) -> str:
    team = row.get("team")
    return (
        f"{team}: champion {float(row.get('champion_probability', 0)):.1%}, "
        f"qualify {float(row.get('qualify_group_probability', 0)):.1%}, "
        f"expected points {float(row.get('expected_points', 0)):.2f}."
    )
