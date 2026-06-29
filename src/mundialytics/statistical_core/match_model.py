from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mundialytics.statistical_core.calibration import recency_weights, safe_count, weighted_mean
from mundialytics.statistical_core.distributions import outcome_probabilities, scoreline_distribution
from mundialytics.statistical_core.schemas import canonical_name, standardize_fixtures


@dataclass
class TeamStrengthProfile:
    team: str
    matches: int
    goals_for: float
    goals_against: float
    recent_goals_for: float
    recent_goals_against: float
    rating: float
    attack_strength: float
    defense_weakness: float
    warnings: str = ""


class MatchOutcomeModel:
    """Auditable independent-Poisson match model.

    This is intentionally simple and transparent for v0.20: it builds team
    strengths from historical event/player data, then creates a scoreline
    distribution for each manual fixture. It is not yet a fully trained
    Dixon-Coles model, but its inputs/outputs are stable and testable.
    """

    def __init__(
        self,
        max_goals: int = 10,
        default_goal_mean: float = 1.25,
        goal_floor: float = 0.05,
        goal_cap: float = 6.0,
        attack_cap: float = 2.5,
        defense_cap: float = 2.5,
        profile_shrinkage_k: float = 8.0,
        low_sample_blend_k: float = 10.0,
        recency_half_life_days: float = 365.0,
        rating_coefficient: float = 85.0,
        rating_divisor: float = 900.0,
        rating_clip: float = 0.35,
        home_advantage: float = 1.08,
        away_disadvantage: float = 0.96,
        draw_lambda_blend: float = 0.0,
        dixon_coles_rho: float = -0.07,
    ):
        self.max_goals = int(max_goals)
        self.default_goal_mean = float(default_goal_mean)
        self.global_goal_mean = float(default_goal_mean)
        self.goal_floor = float(goal_floor)
        self.goal_cap = float(goal_cap)
        self.attack_cap = float(attack_cap)
        self.defense_cap = float(defense_cap)
        self.profile_shrinkage_k = float(profile_shrinkage_k)
        self.low_sample_blend_k = float(low_sample_blend_k)
        self.recency_half_life_days = float(recency_half_life_days)
        self.rating_coefficient = float(rating_coefficient)
        self.rating_divisor = float(rating_divisor)
        self.rating_clip = float(rating_clip)
        self.home_advantage = float(home_advantage)
        self.away_disadvantage = float(away_disadvantage)
        self.draw_lambda_blend = float(draw_lambda_blend)
        self.dixon_coles_rho = float(dixon_coles_rho)
        self.profiles: dict[str, TeamStrengthProfile] = {}
        self.audit: dict[str, Any] = {}

    @property
    def model_config(self) -> dict[str, float | int]:
        return {
            "max_goals": self.max_goals,
            "default_goal_mean": self.default_goal_mean,
            "goal_floor": self.goal_floor,
            "goal_cap": self.goal_cap,
            "attack_cap": self.attack_cap,
            "defense_cap": self.defense_cap,
            "profile_shrinkage_k": self.profile_shrinkage_k,
            "low_sample_blend_k": self.low_sample_blend_k,
            "recency_half_life_days": self.recency_half_life_days,
            "rating_coefficient": self.rating_coefficient,
            "rating_divisor": self.rating_divisor,
            "rating_clip": self.rating_clip,
            "home_advantage": self.home_advantage,
            "away_disadvantage": self.away_disadvantage,
            "draw_lambda_blend": self.draw_lambda_blend,
            "dixon_coles_rho": self.dixon_coles_rho,
        }

    def _safe_goal_lambda(self, value: float) -> float:
        return float(min(max(safe_count(value, "goals", self.default_goal_mean), self.goal_floor), self.goal_cap))

    def fit(self, historical_events: pd.DataFrame | None) -> "MatchOutcomeModel":
        events = _team_match_goal_frame(historical_events)
        return self.fit_team_goal_frame(events)

    def fit_team_goal_frame(self, events: pd.DataFrame | None) -> "MatchOutcomeModel":
        """Fit from a precomputed team/match goal frame.

        This is equivalent to ``fit(historical_events)`` after event-to-team
        aggregation, but is much faster for automated model-lab sweeps. The
        expected columns are: team, date, goals_for, goals_against.
        """
        if events is None or events.empty:
            self.global_goal_mean = self.default_goal_mean
            self.profiles = {}
            self.audit = {"match_model_fit": "fallback_no_historical_goal_events", "teams": 0, "model_config": self.model_config}
            return self
        events = events.copy()
        events["team"] = events["team"].map(canonical_name)
        events["goals_for"] = pd.to_numeric(events["goals_for"], errors="coerce").fillna(0.0)
        events["goals_against"] = pd.to_numeric(events["goals_against"], errors="coerce").fillna(events["goals_for"].mean())
        events["date"] = pd.to_datetime(events.get("date"), errors="coerce")
        self.global_goal_mean = safe_count(events["goals_for"].mean(), "goals", default=self.default_goal_mean)
        profiles: dict[str, TeamStrengthProfile] = {}
        for team, g in events.groupby("team"):
            weights = recency_weights(g["date"], half_life_days=self.recency_half_life_days) if "date" in g.columns else None
            gf_raw = weighted_mean(g["goals_for"], weights, self.global_goal_mean)
            ga_raw = weighted_mean(g["goals_against"], weights, self.global_goal_mean)
            sample_weight = 1.0
            if self.profile_shrinkage_k > 0:
                sample_weight = float(len(g) / (len(g) + self.profile_shrinkage_k))
            gf = sample_weight * gf_raw + (1.0 - sample_weight) * self.global_goal_mean
            ga = sample_weight * ga_raw + (1.0 - sample_weight) * self.global_goal_mean
            recent = g.sort_values("date").tail(5) if "date" in g.columns else g.tail(5)
            rgf = float(pd.to_numeric(recent["goals_for"], errors="coerce").mean()) if len(recent) else gf
            rga = float(pd.to_numeric(recent["goals_against"], errors="coerce").mean()) if len(recent) else ga
            if self.profile_shrinkage_k > 0:
                rgf = sample_weight * rgf + (1.0 - sample_weight) * self.global_goal_mean
                rga = sample_weight * rga + (1.0 - sample_weight) * self.global_goal_mean
            goal_diff_signal = (0.7 * (gf - ga)) + (0.3 * (rgf - rga))
            rating = float(np.clip(1500 + self.rating_coefficient * goal_diff_signal, 1200, 1850))
            attack = float(np.clip(gf / max(self.global_goal_mean, 0.1), 0.35, self.attack_cap))
            defense = float(np.clip(ga / max(self.global_goal_mean, 0.1), 0.35, self.defense_cap))
            warning = "" if len(g) >= 5 else "low_team_match_sample"
            profiles[str(team)] = TeamStrengthProfile(
                team=str(team),
                matches=int(len(g)),
                goals_for=float(gf),
                goals_against=float(ga),
                recent_goals_for=float(rgf),
                recent_goals_against=float(rga),
                rating=rating,
                attack_strength=attack,
                defense_weakness=defense,
                warnings=warning,
            )
        self.profiles = profiles
        self.audit = {"match_model_fit": "historical_event_team_goals", "teams": len(profiles), "global_goal_mean": self.global_goal_mean, "model_config": self.model_config}
        return self

    def inject_external_ratings(self, elo_ratings: dict[str, float]) -> "MatchOutcomeModel":
        """Override internal goal-diff ratings with externally computed ELO.

        Call after fit(). The ELO values replace the heuristic
        ``1500 + 85 * goal_diff_signal`` rating in each TeamStrengthProfile.
        Teams not present in elo_ratings keep their internal rating.

        Parameters
        ----------
        elo_ratings : dict mapping canonical team name -> ELO float
        """
        for team, profile in self.profiles.items():
            ext = elo_ratings.get(team) or elo_ratings.get(canonical_name(team))
            if ext is not None:
                self.profiles[team] = TeamStrengthProfile(
                    team=profile.team,
                    matches=profile.matches,
                    goals_for=profile.goals_for,
                    goals_against=profile.goals_against,
                    recent_goals_for=profile.recent_goals_for,
                    recent_goals_against=profile.recent_goals_against,
                    rating=float(ext),
                    attack_strength=profile.attack_strength,
                    defense_weakness=profile.defense_weakness,
                    warnings=profile.warnings,
                )
        self.audit["external_elo_teams_injected"] = sum(
            1 for t in self.profiles if (elo_ratings.get(t) or elo_ratings.get(canonical_name(t))) is not None
        )
        return self

    def profile_for(self, team: str) -> TeamStrengthProfile:
        key = canonical_name(team)
        if key in self.profiles:
            return self.profiles[key]
        return TeamStrengthProfile(
            team=key,
            matches=0,
            goals_for=self.global_goal_mean,
            goals_against=self.global_goal_mean,
            recent_goals_for=self.global_goal_mean,
            recent_goals_against=self.global_goal_mean,
            rating=1500.0,
            attack_strength=1.0,
            defense_weakness=1.0,
            warnings="fallback_team_profile_no_history",
        )

    def expected_goals(self, home_team: str, away_team: str, neutral: int | bool = 1) -> tuple[float, float, list[str]]:
        home = self.profile_for(home_team)
        away = self.profile_for(away_team)
        warnings = [w for w in [home.warnings, away.warnings] if w]
        rating_diff = home.rating - away.rating
        rating_factor_home = math.exp(np.clip(rating_diff / max(self.rating_divisor, 1.0), -self.rating_clip, self.rating_clip))
        rating_factor_away = math.exp(np.clip(-rating_diff / max(self.rating_divisor, 1.0), -self.rating_clip, self.rating_clip))
        home_adv = self.home_advantage if not bool(int(neutral)) else 1.0
        away_adv = self.away_disadvantage if not bool(int(neutral)) else 1.0
        h_lam = self.global_goal_mean * home.attack_strength * away.defense_weakness * rating_factor_home * home_adv
        a_lam = self.global_goal_mean * away.attack_strength * home.defense_weakness * rating_factor_away * away_adv
        if self.low_sample_blend_k > 0:
            home_conf = home.matches / (home.matches + self.low_sample_blend_k) if home.matches > 0 else 0.0
            away_conf = away.matches / (away.matches + self.low_sample_blend_k) if away.matches > 0 else 0.0
            pair_conf = float(math.sqrt(max(home_conf, 0.0) * max(away_conf, 0.0)))
            h_lam = pair_conf * h_lam + (1.0 - pair_conf) * self.global_goal_mean
            a_lam = pair_conf * a_lam + (1.0 - pair_conf) * self.global_goal_mean
            if pair_conf < 0.60:
                warnings.append(f"lambda_shrunk_low_sample_pair_conf={pair_conf:.2f}")
        if self.draw_lambda_blend > 0:
            b = float(np.clip(self.draw_lambda_blend, 0.0, 1.0))
            avg = 0.5 * (h_lam + a_lam)
            h_lam = (1.0 - b) * h_lam + b * avg
            a_lam = (1.0 - b) * a_lam + b * avg
        return self._safe_goal_lambda(h_lam), self._safe_goal_lambda(a_lam), warnings

    def predict_fixtures(self, fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        f = standardize_fixtures(fixtures)
        rows: list[dict[str, Any]] = []
        score_rows: list[pd.DataFrame] = []
        for _, r in f.iterrows():
            lh, la, warnings = self.expected_goals(r["home_team"], r["away_team"], r.get("neutral", 1))
            probs = outcome_probabilities(lh, la, max_goals=self.max_goals, dixon_coles_rho=self.dixon_coles_rho)
            dist = scoreline_distribution(lh, la, max_goals=self.max_goals, normalize=True, dixon_coles_rho=self.dixon_coles_rho)
            top_scorelines = dist.top_scorelines(8)
            score_frame = dist.to_long_frame(match_id=str(r["match_id"]))
            score_rows.append(score_frame)
            row = r.to_dict()
            row.update(
                {
                    "lambda_home": lh,
                    "lambda_away": la,
                    "expected_home_goals": lh,
                    "expected_away_goals": la,
                    "home_rating": self.profile_for(r["home_team"]).rating,
                    "away_rating": self.profile_for(r["away_team"]).rating,
                    "model_type": "v026_poisson_profile_dixon_coles_optional",
                    "scoreline_distribution_json": json.dumps(top_scorelines, ensure_ascii=False),
                    "warnings": ";".join(warnings),
                }
            )
            row.update(probs)
            # Numerical guard to make audit easy.
            total = row["p_home_win"] + row["p_draw"] + row["p_away_win"]
            if abs(total - 1.0) > 1e-6:
                row["warnings"] = ";".join([w for w in [row["warnings"], f"outcome_probs_sum={total:.8f}"] if w])
            rows.append(row)
        score_df = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
        return pd.DataFrame(rows), score_df


def _team_match_goal_frame(historical_events: pd.DataFrame | None) -> pd.DataFrame:
    if historical_events is None or historical_events.empty:
        return pd.DataFrame()
    df = historical_events.copy()
    if "team" not in df.columns or "match_id" not in df.columns:
        return pd.DataFrame()
    df["team"] = df["team"].map(canonical_name)
    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].map(canonical_name)
    else:
        df["opponent"] = "unknown"
    if "date" not in df.columns:
        df["date"] = pd.NaT
    if "goals_for" in df.columns:
        goal_col = "goals_for"
    elif "goals" in df.columns:
        goal_col = "goals"
    else:
        return pd.DataFrame()
    team_goals = (
        df.assign(_goals=pd.to_numeric(df[goal_col], errors="coerce").fillna(0))
        .groupby(["match_id", "date", "team", "opponent"], dropna=False)["_goals"]
        .sum()
        .reset_index(name="goals_for")
    )
    opp = team_goals[["match_id", "team", "goals_for"]].rename(columns={"team": "opponent", "goals_for": "goals_against"})
    out = team_goals.merge(opp, on=["match_id", "opponent"], how="left")
    out["goals_against"] = pd.to_numeric(out["goals_against"], errors="coerce").fillna(out["goals_for"].mean())
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out
