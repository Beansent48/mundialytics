"""Mundialytics statistical core.

Manual-input-first modules for match prediction, team counts, player props,
tournament simulation, betting value, output contracts and reporting.
"""

from mundialytics.statistical_core.betting_value import BettingValueConfig, BettingValueEngine
from mundialytics.statistical_core.scorer_model import CompetitionForecastEngine, ScorerForecastConfig
from mundialytics.statistical_core.match_model import MatchOutcomeModel
from mundialytics.statistical_core.model_lab import run_model_lab
from mundialytics.statistical_core.event_evaluation import EventEvaluationConfig, evaluate_event_models_temporal
from mundialytics.statistical_core.event_model_lab import run_event_model_lab
from mundialytics.statistical_core.player_prop_champion import ChampionPropConfig, run_player_prop_champion_lab
from mundialytics.statistical_core.rolling_validation import RollingMatchConfig, rolling_match_backtest, run_rolling_model_lab
from mundialytics.statistical_core.dynamic_lines import DynamicLineConfig, build_dynamic_market_lines
from mundialytics.statistical_core.matchday_summary import MATCHDAY_SUMMARY_VERSION, build_matchday_summary
from mundialytics.statistical_core.tournament_report import TOURNAMENT_REPORT_VERSION, build_tournament_report
from mundialytics.statistical_core.team_stats_model import TeamStatsModel
from mundialytics.statistical_core.tournament_simulator import TournamentSimulationConfig, TournamentSimulator
from mundialytics.statistical_core.simulation_evaluation import SIMULATION_EVALUATION_VERSION, evaluate_simulation_predictions
from mundialytics.statistical_core.simulation_contract import (
    CONTRACT_VERSION,
    SIMULATOR_OUTPUT_CONTRACTS,
    build_simulator_contract_report,
    expected_output_files,
)

__all__ = [
    "BettingValueConfig",
    "DynamicLineConfig",
    "build_dynamic_market_lines",
    "MATCHDAY_SUMMARY_VERSION",
    "build_matchday_summary",
    "TOURNAMENT_REPORT_VERSION",
    "build_tournament_report",
    "SIMULATION_EVALUATION_VERSION",
    "evaluate_simulation_predictions",
    "BettingValueEngine",
    "CompetitionForecastEngine",
    "MatchOutcomeModel",
    "run_model_lab",
    "EventEvaluationConfig",
    "evaluate_event_models_temporal",
    "run_event_model_lab",
    "ChampionPropConfig",
    "run_player_prop_champion_lab",
    "RollingMatchConfig",
    "rolling_match_backtest",
    "run_rolling_model_lab",
    "ScorerForecastConfig",
    "TeamStatsModel",
    "TournamentSimulationConfig",
    "TournamentSimulator",
    "CONTRACT_VERSION",
    "SIMULATOR_OUTPUT_CONTRACTS",
    "build_simulator_contract_report",
    "expected_output_files",
]
