"""Mundialytics prediction models.

Active models used by the prediction pipeline:
  GoalLambdaModel      — Poisson GLM, predicts team goal lambda
  MatchProbabilityResult / match_probabilities — Skellam 1X2 probs from lambdas
  MinutesModel         — player expected minutes from recent history

Note: PlayerEventModel (models/player_event_model.py) has a pre-existing
broken import (player_global_id missing from identity.normalization) and
is excluded from this package until fixed. Use PlayerProfileModel from
statistical_core.event_model instead for player stats.
"""

from mundialytics.models.goal_model import GoalLambdaModel, GoalModelConfig
from mundialytics.models.result_model import MatchProbabilityResult, match_probabilities
from mundialytics.models.minutes_model import MinutesModel

__all__ = [
    "GoalLambdaModel",
    "GoalModelConfig",
    "MatchProbabilityResult",
    "match_probabilities",
    "MinutesModel",
]
