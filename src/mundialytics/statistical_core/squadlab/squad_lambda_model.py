"""Bridges a SquadLab squad's PlayerStrengthModel.team_strength() output onto
AttackDefenseModel's log-scale parameters and real per-market event rates.

See calibration_constants.py's module docstring for why this is a
range-based map rather than a precise per-club fit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mundialytics.statistical_core.player_strength import PlayerStrengthModel, PlayerStrengthProfile
from mundialytics.statistical_core.squadlab.calibration_constants import (
    ATTACK_PARAM_CLIP,
    DEFENSE_PARAM_CLIP,
    EVENT_CALIBRATION,
    GOAL_ATTACK_INTERCEPT,
    GOAL_ATTACK_SLOPE,
    GOAL_DEFENSE_INTERCEPT,
    GOAL_DEFENSE_SLOPE,
)


@dataclass
class SquadAttackDefenseParams:
    attack_param: float   # on AttackDefenseModel's log-scale
    defense_param: float


class SquadLambdaModel:
    def __init__(self, strength_model: PlayerStrengthModel):
        self.strength_model = strength_model

    def fit_squad(self, squad: list[PlayerStrengthProfile]) -> SquadAttackDefenseParams:
        strength = self.strength_model.team_strength(squad)
        attack_param = GOAL_ATTACK_SLOPE * strength["attack_index"] + GOAL_ATTACK_INTERCEPT
        defense_param = GOAL_DEFENSE_SLOPE * strength["defense_index"] + GOAL_DEFENSE_INTERCEPT
        attack_param = float(np.clip(attack_param, *ATTACK_PARAM_CLIP))
        defense_param = float(np.clip(defense_param, *DEFENSE_PARAM_CLIP))
        return SquadAttackDefenseParams(attack_param=attack_param, defense_param=defense_param)

    def event_lambdas(self, squad: list[PlayerStrengthProfile]) -> dict[str, float]:
        """One lambda per market (shots/sot/corners/fouls/yellow_cards), the
        squad's own (not opponent-adjusted) rate. Opponent adjustment happens
        in the LambdaSource dispatcher the same way goal lambda does."""
        strength = self.strength_model.team_strength(squad)
        out: dict[str, float] = {}
        for market, c in EVENT_CALIBRATION.items():
            basis_value = strength["attack_index"] if c["basis"] == "attack" else strength["defense_index"]
            lam = c["slope"] * basis_value + c["intercept"]
            out[market] = float(np.clip(lam, c["clip_min"], c["clip_max"]))
        return out
