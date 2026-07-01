"""Resolves match/event lambdas for a fixture, regardless of whether either
side is a real historical team (looked up via the existing PredictionEngine)
or a fictional SquadLab squad (bridged via SquadLambdaModel).

The squad is NEVER registered into PredictionEngine/AttackDefenseModel's
fit-time-only caches (_team_row_cache, ad_model_.team_index_) — those
structures were never designed for post-fit mutation. Instead,
SeasonLambdaSource hand-reimplements AttackDefenseModel._lam's own formula
(lh = exp(mu + ha + attack[home] - defense[away])) using the squad's
bridged attack/defense params plus the real opponent's already-fitted
attack_/defense_ arrays, read-only.
"""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from mundialytics.statistical_core.attack_defense_model import AttackDefenseModel
from mundialytics.statistical_core.player_strength import PlayerStrengthProfile
from mundialytics.statistical_core.prediction_engine import PredictionEngine
from mundialytics.statistical_core.schemas import canonical_name
from mundialytics.statistical_core.squadlab.squad_lambda_model import (
    SquadAttackDefenseParams,
    SquadLambdaModel,
)

# EventLambdaModel markets are suffixed "_for" (shots_for, sot_for, ...);
# squad-side calibration keys (calibration_constants.EVENT_CALIBRATION,
# SquadLambdaModel.event_lambdas) are unsuffixed. Translate between them.
MARKET_KEY_MAP = {
    "shots_for": "shots", "sot_for": "sot", "corners_for": "corners",
    "fouls_for": "fouls", "yellow_cards_for": "yellow_cards",
}


class LambdaSource(Protocol):
    """Resolves (lambda_home, lambda_away) for one fixture, regardless of
    whether either side is a real historical team or a fictional squad."""

    def match_lambdas(
        self, home: str, away: str, *, competition: str, neutral: bool
    ) -> tuple[float, float]: ...

    def event_lambdas(
        self, home: str, away: str, *, competition: str
    ) -> dict[str, tuple[float, float]]:
        """market -> (home_lambda, away_lambda) for shots/sot/corners/fouls/yellows."""
        ...


class RealTeamLambdaSource:
    """Thin adapter around the existing, unmodified PredictionEngine."""

    def __init__(self, engine: PredictionEngine):
        self.engine = engine

    def match_lambdas(self, home: str, away: str, *, competition: str, neutral: bool) -> tuple[float, float]:
        pred = self.engine.predict_match(home, away, competition=competition, neutral=neutral)
        return pred.lambda_home, pred.lambda_away

    def event_lambdas(self, home: str, away: str, *, competition: str) -> dict[str, tuple[float, float]]:
        pred = self.engine.predict_match(home, away, competition=competition, neutral=False)
        return {
            "shots_for": (pred.expected_shots_home, pred.expected_shots_away),
            "sot_for": (pred.expected_sot_home, pred.expected_sot_away),
            "corners_for": (pred.expected_corners_home, pred.expected_corners_away),
            "fouls_for": (pred.expected_fouls_home, pred.expected_fouls_away),
            "yellow_cards_for": (pred.expected_yellows_home, pred.expected_yellows_away),
        }


class SeasonLambdaSource:
    """Dispatcher used by a season: real-vs-real fixtures delegate to
    RealTeamLambdaSource unchanged; fixtures involving the squad resolve
    through a hand-reimplementation of AttackDefenseModel._lam's formula."""

    def __init__(
        self,
        squad_team_name: str,
        squad: list[PlayerStrengthProfile],
        squad_bridge: SquadLambdaModel,
        real_source: RealTeamLambdaSource,
        ad_model: AttackDefenseModel,
    ):
        self.squad_team_name = squad_team_name
        self.real_source = real_source
        self.ad_model = ad_model
        # Computed once at season-build time, not per-fixture.
        self.squad_params: SquadAttackDefenseParams = squad_bridge.fit_squad(squad)
        self.squad_event_lambdas: dict[str, float] = squad_bridge.event_lambdas(squad)

    def _league_context(self, competition: str, neutral: bool) -> tuple[float, float]:
        ad = self.ad_model
        league_idx = ad.league_index_.get(competition, 0)
        if neutral:
            ha = 0.0
        elif league_idx < len(ad.league_home_adv_):
            ha = float(ad.league_home_adv_[league_idx])
        else:
            ha = ad.home_adv_
        league_mu = float(ad.league_effect_[league_idx]) if league_idx < len(ad.league_effect_) else ad.mu_
        return league_mu, ha

    def _opponent_params(self, opponent: str) -> tuple[float, float]:
        ad = self.ad_model
        idx = ad.team_index_.get(canonical_name(opponent))
        if idx is None:
            return 0.0, 0.0   # AttackDefenseModel's own fallback for unknown teams
        return float(ad.attack_[idx]), float(ad.defense_[idx])

    def match_lambdas(self, home: str, away: str, *, competition: str, neutral: bool) -> tuple[float, float]:
        if home != self.squad_team_name and away != self.squad_team_name:
            return self.real_source.match_lambdas(home, away, competition=competition, neutral=neutral)

        squad_is_home = home == self.squad_team_name
        opponent = away if squad_is_home else home
        opp_attack, opp_defense = self._opponent_params(opponent)
        league_mu, ha = self._league_context(competition, neutral)
        sq_attack, sq_defense = self.squad_params.attack_param, self.squad_params.defense_param

        if squad_is_home:
            lh = math.exp(league_mu + ha + sq_attack - opp_defense)
            la = math.exp(league_mu + opp_attack - sq_defense)
        else:
            lh = math.exp(league_mu + ha + opp_attack - sq_defense)
            la = math.exp(league_mu + sq_attack - opp_defense)

        ad = self.ad_model
        lh = float(np.clip(lh, ad.goal_floor, ad.goal_cap))
        la = float(np.clip(la, ad.goal_floor, ad.goal_cap))
        return lh, la

    def event_lambdas(self, home: str, away: str, *, competition: str) -> dict[str, tuple[float, float]]:
        if home != self.squad_team_name and away != self.squad_team_name:
            return self.real_source.event_lambdas(home, away, competition=competition)

        squad_is_home = home == self.squad_team_name
        opponent = away if squad_is_home else home
        engine = self.real_source.engine
        opp_canon = canonical_name(opponent)
        opp_is_home_flag = 0 if squad_is_home else 1

        out: dict[str, tuple[float, float]] = {}
        for full_market, em in engine.event_models_.items():
            opp_row = engine._get_team_row(opp_canon, opp_is_home_flag, competition)
            opp_lam = float(em.predict_lambda(opp_row)[0]) if not opp_row.empty else em.mean_
            squad_lam = self.squad_event_lambdas.get(MARKET_KEY_MAP.get(full_market, full_market), em.mean_)
            out[full_market] = (squad_lam, opp_lam) if squad_is_home else (opp_lam, squad_lam)
        return out
