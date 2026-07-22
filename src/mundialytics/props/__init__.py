"""Props layer: team-event and player-prop probability models.

Validated 2026-07-22 on walk-forward folds 2021/22-2025/26 before wiring
(scripts/backtest_team_props.py, scripts/backtest_player_props.py). Decoupled
from PredictionEngine: match-context lambdas are injected by the caller.
"""

from mundialytics.props.player_props import PlayerPropsModel
from mundialytics.props.team_props import TeamPropsModel

__all__ = ["TeamPropsModel", "PlayerPropsModel"]
