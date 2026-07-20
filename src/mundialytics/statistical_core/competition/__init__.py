"""
Competition layer — stateful, "from the current point" simulation of a league
or tournament, built ON TOP of the existing PredictionEngine (never replacing it).

The normal PredictionEngine stays stateless (two teams -> a match prediction) and
is still used by SquadLab and pre-season simulations. This package adds the state
machine that a real football site needs:

  - LeagueState        : the current state of a competition (played results,
                         remaining fixtures, live standings). This is BOTH the
                         data backbone the site browses AND the simulator input.
  - compute_standings  : a league table from played matches, with tiebreakers.
  - load_league_state_from_foundation : build a LeagueState from the foundation
                         match CSV at any cutoff (the "from the current point"
                         mechanism; also doubles as a leakage-free backtest split).

Phase 1 scope: leagues only, team-level probabilities (title / top-N / relegation
/ position distribution) + analytic xPoints. Player props (Golden Boot) and
group/knockout tournament rules are later phases. Live data feeds swap in behind
the loader without changing LeagueState.
"""
from __future__ import annotations

from mundialytics.statistical_core.competition.standings import (
    StandingsRow,
    compute_standings,
)
from mundialytics.statistical_core.competition.state import LeagueState
from mundialytics.statistical_core.competition.cutoff import (
    load_league_state_from_foundation,
)
from mundialytics.statistical_core.competition.engine_provider import (
    train_engine_before_cutoff,
    fixture_lambdas,
)
from mundialytics.statistical_core.competition.xpoints import expected_points_table
from mundialytics.statistical_core.competition.resume_simulator import (
    LeagueForecast,
    simulate_rest_of_season,
)
from mundialytics.statistical_core.competition.evolution import forecast_timeline
from mundialytics.statistical_core.competition.forecast_cache import (
    build_bundle,
    get_or_build,
    load_bundle,
    save_bundle,
    bundle_path,
    snapshot_for,
    available_matchdays,
    build_timeline,
)

__all__ = [
    "StandingsRow",
    "compute_standings",
    "LeagueState",
    "load_league_state_from_foundation",
    "train_engine_before_cutoff",
    "fixture_lambdas",
    "expected_points_table",
    "LeagueForecast",
    "simulate_rest_of_season",
    "forecast_timeline",
    "build_bundle",
    "get_or_build",
    "load_bundle",
    "save_bundle",
    "bundle_path",
    "snapshot_for",
    "available_matchdays",
    "build_timeline",
]
