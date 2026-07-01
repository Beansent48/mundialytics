"""Tests for the SquadLab match-by-match season simulation engine
(src/mundialytics/statistical_core/squadlab/). See
C:\\Users\\Vicente\\.claude\\plans\\mossy-snuggling-manatee.md for the
design and the reasoning behind each check.

Two checks from the original plan (a per-club calibration ground-truth
test, and a "Barcelona should land near their real table position"
realism test) were adapted rather than implemented literally: the
per-club regression they assumed was tried and abandoned mid-build
(player_profiles_with_positions.csv has no season column, so
reconstructing "team X's current best-11" mixes incompatible eras — see
calibration_constants.py's docstring). The replacement calibration is
range-based and only promises correct ORDERING and a realistic SCALE, not
per-club precision, so the tests below check exactly that instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mundialytics.statistical_core.attack_defense_model import AttackDefenseModel
from mundialytics.statistical_core.player_strength import PlayerStrengthModel
from mundialytics.statistical_core.prediction_engine import PredictionEngine
from mundialytics.statistical_core.squadlab.calendar import generate_double_round_robin
from mundialytics.statistical_core.squadlab.lambda_source import RealTeamLambdaSource, SeasonLambdaSource
from mundialytics.statistical_core.squadlab.player_rating import (
    attribute_cards,
    attribute_goals,
    compute_match_ratings,
)
from mundialytics.statistical_core.squadlab.season_simulator import SeasonOrchestrator
from mundialytics.statistical_core.squadlab.squad_lambda_model import SquadLambdaModel

POSITION_SLOTS = {"Goalkeeper": 1, "Defender": 4, "Midfielder": 3, "Forward": 3}


@pytest.fixture(scope="module")
def strength_model() -> PlayerStrengthModel:
    m = PlayerStrengthModel()
    m.fit()
    return m


@pytest.fixture(scope="module")
def engine(strength_model: PlayerStrengthModel) -> PredictionEngine:
    df_clubs = pd.read_csv("data/processed/foundation_big5_multi_season.csv")
    e = PredictionEngine()
    e.fit(df_clubs)
    return e


def _extreme_squad(model: PlayerStrengthModel, best: bool) -> list:
    squad = []
    for pos, n in POSITION_SLOTS.items():
        cands = [p for p in model.profiles_.values() if p.position == pos and p.matches >= 10]
        cands = sorted(cands, key=lambda p: -p.overall if best else p.overall)
        squad.extend(cands[:n])
    return squad


# 1. Calendar correctness ----------------------------------------------------

def test_round_robin_each_pair_plays_home_and_away_once() -> None:
    teams = [f"T{i}" for i in range(6)]
    fixtures = generate_double_round_robin(teams)
    assert len(fixtures) == 6 * 5  # n*(n-1)

    pair_counts: dict[tuple[str, str], int] = {}
    ordered_counts: dict[tuple[str, str], int] = {}
    matchday_teams: dict[int, list[str]] = {}
    for f in fixtures:
        pair = tuple(sorted([f.home, f.away]))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        ordered_counts[(f.home, f.away)] = ordered_counts.get((f.home, f.away), 0) + 1
        matchday_teams.setdefault(f.matchday, []).extend([f.home, f.away])

    assert all(v == 2 for v in pair_counts.values())
    assert all(v == 1 for v in ordered_counts.values())
    assert max(matchday_teams) == 2 * (len(teams) - 1)
    for teams_in_md in matchday_teams.values():
        assert len(teams_in_md) == len(set(teams_in_md))  # no team plays twice on one matchday


def test_round_robin_handles_odd_team_counts() -> None:
    teams = [f"T{i}" for i in range(7)]
    fixtures = generate_double_round_robin(teams)
    pair_counts: dict[tuple[str, str], int] = {}
    for f in fixtures:
        pair = tuple(sorted([f.home, f.away]))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    assert len(pair_counts) == 7 * 6 // 2
    assert all(v == 2 for v in pair_counts.values())


# 2. Squad lambda bridge — ordering and bounds, not per-club precision ------

def test_squad_lambda_model_orders_squads_by_strength_and_stays_in_bounds(
    strength_model: PlayerStrengthModel,
) -> None:
    from mundialytics.statistical_core.squadlab.calibration_constants import (
        ATTACK_PARAM_CLIP,
        DEFENSE_PARAM_CLIP,
    )

    bridge = SquadLambdaModel(strength_model)
    best = bridge.fit_squad(_extreme_squad(strength_model, best=True))
    worst = bridge.fit_squad(_extreme_squad(strength_model, best=False))

    assert best.attack_param > worst.attack_param
    assert best.defense_param > worst.defense_param
    for params in (best, worst):
        assert ATTACK_PARAM_CLIP[0] <= params.attack_param <= ATTACK_PARAM_CLIP[1]
        assert DEFENSE_PARAM_CLIP[0] <= params.defense_param <= DEFENSE_PARAM_CLIP[1]


# 3. LambdaSource end-to-end --------------------------------------------------

def test_season_lambda_source_squad_vs_real_team_produces_valid_distribution(
    strength_model: PlayerStrengthModel, engine: PredictionEngine,
) -> None:
    from mundialytics.statistical_core.distributions import outcome_probabilities

    squad = _extreme_squad(strength_model, best=True)
    real_source = RealTeamLambdaSource(engine)
    bridge = SquadLambdaModel(strength_model)
    source = SeasonLambdaSource("Tu Equipo", squad, bridge, real_source, engine.ad_model_)

    for home, away in [("Tu Equipo", "getafe"), ("barcelona", "Tu Equipo")]:
        lh, la = source.match_lambdas(home, away, competition="LaLiga", neutral=False)
        assert 0.05 <= lh <= 6.0
        assert 0.05 <= la <= 6.0
        probs = outcome_probabilities(lh, la, dixon_coles_rho=-0.07)
        total = probs["p_home_win"] + probs["p_draw"] + probs["p_away_win"]
        assert abs(total - 1.0) < 1e-6


def test_season_lambda_source_delegates_real_vs_real_unchanged(
    strength_model: PlayerStrengthModel, engine: PredictionEngine,
) -> None:
    squad = _extreme_squad(strength_model, best=True)
    real_source = RealTeamLambdaSource(engine)
    bridge = SquadLambdaModel(strength_model)
    source = SeasonLambdaSource("Tu Equipo", squad, bridge, real_source, engine.ad_model_)

    direct = real_source.match_lambdas("barcelona", "getafe", competition="LaLiga", neutral=False)
    via_dispatcher = source.match_lambdas("barcelona", "getafe", competition="LaLiga", neutral=False)
    assert direct == via_dispatcher


# 4. Player match ratings ------------------------------------------------------

def test_match_ratings_scorer_gets_bonus_and_ratings_stay_in_range(
    strength_model: PlayerStrengthModel,
) -> None:
    squad = _extreme_squad(strength_model, best=True)
    rng = np.random.default_rng(7)
    goal_events = attribute_goals(squad, n_goals=3, rng=rng)
    card_players = attribute_cards(squad, n_cards=1, rng=rng)
    ratings = compute_match_ratings(squad, goal_events, card_players, goals_conceded=0, rng=rng)

    assert len(ratings) == len(squad)  # every squad player gets an entry
    assert sum(ev.goals for ev in ratings.values()) == 3
    assert all(2.0 <= ev.rating <= 10.0 for ev in ratings.values())

    scorers = {scorer for scorer, _ in goal_events}
    non_scorer_ratings = [ev.rating for name, ev in ratings.items() if name not in scorers]
    scorer_ratings = [ev.rating for name, ev in ratings.items() if name in scorers]
    assert max(scorer_ratings) > min(non_scorer_ratings, default=0.0)


def test_attribute_goals_zero_goals_returns_empty() -> None:
    rng = np.random.default_rng(1)
    assert attribute_goals([], n_goals=3, rng=rng) == []


# 5. SeasonOrchestrator structural checks -------------------------------------

def _small_orchestrator(strength_model, engine, n_sims_seed=42):
    squad = _extreme_squad(strength_model, best=True)
    real_opponents = ["getafe", "sevilla", "osasuna", "celta vigo", "alaves"]
    fixtures = generate_double_round_robin(["Tu Equipo"] + real_opponents)
    real_source = RealTeamLambdaSource(engine)
    bridge = SquadLambdaModel(strength_model)
    lambda_source = SeasonLambdaSource("Tu Equipo", squad, bridge, real_source, engine.ad_model_)
    return SeasonOrchestrator(
        lambda_source, fixtures, squad_roster={"Tu Equipo": squad},
        competition="LaLiga", random_seed=n_sims_seed,
    ), squad, real_opponents


def test_season_orchestrator_play_once_produces_complete_table(
    strength_model: PlayerStrengthModel, engine: PredictionEngine,
) -> None:
    orch, squad, real_opponents = _small_orchestrator(strength_model, engine)
    result = orch.play_once(narrative=True)

    n_teams = 1 + len(real_opponents)
    assert len(result.table) == n_teams
    assert set(result.table["played"]) == {2 * (n_teams - 1)}
    assert not result.player_season_tallies.empty
    # Only the squad's 11 players are tracked -- real-team players are out of scope.
    assert set(result.player_season_tallies["player"]) == {p.player for p in squad}
    assert sum(1 for m in result.matches if m.home == "Tu Equipo" or m.away == "Tu Equipo") == 2 * (n_teams - 1)


def test_monte_carlo_and_narrative_share_same_lambda_path(
    strength_model: PlayerStrengthModel, engine: PredictionEngine,
) -> None:
    orch, _, _ = _small_orchestrator(strength_model, engine)

    narrative = orch.play_once(narrative=False)
    narrative_scores = [(m.home, m.away, m.home_goals, m.away_goals) for m in narrative.matches]

    rng = np.random.default_rng(orch.random_seed)
    lams_h, lams_a = orch._lambda_arrays()
    hg_all, ag_all = rng.poisson(lams_h), rng.poisson(lams_a)
    mc_first_iter_scores = [
        (f.home, f.away, int(hg), int(ag)) for f, hg, ag in zip(orch.fixtures, hg_all, ag_all)
    ]
    assert narrative_scores == mc_first_iter_scores


def test_run_monte_carlo_probabilities_are_internally_consistent(
    strength_model: PlayerStrengthModel, engine: PredictionEngine,
) -> None:
    orch, squad, real_opponents = _small_orchestrator(strength_model, engine)
    mc = orch.run_monte_carlo(n_sims=500)

    n_teams = 1 + len(real_opponents)
    assert len(mc) == n_teams
    assert abs(mc["p_champion"].sum() - 1.0) < 1e-9
    for _, row in mc.iterrows():
        assert row["p_champion"] <= row["p_top2"] + 1e-9
        assert row["p_top2"] <= row["p_top4"] + 1e-9
        assert 0.0 <= row["p_relegation"] <= 1.0


# 6. Regression guard: squad quality should visibly move outcomes ------------

def test_elite_squad_massively_outperforms_weak_squad_in_a_shared_league(
    strength_model: PlayerStrengthModel, engine: PredictionEngine,
) -> None:
    """Not a claim that a specific squad lands at a specific real-world table
    position (the calibration is range-based, not per-club precise — see
    module docstring) -- just that the engine responds sensibly to squad
    quality: an all-star XI should crush the same fixture list a bottom-of-
    the-pool XI would struggle with.
    """
    real_opponents = ["getafe", "sevilla", "osasuna", "celta vigo", "alaves"]
    fixtures = generate_double_round_robin(["Tu Equipo"] + real_opponents)
    real_source = RealTeamLambdaSource(engine)
    bridge = SquadLambdaModel(strength_model)

    def _run(best: bool) -> pd.DataFrame:
        squad = _extreme_squad(strength_model, best=best)
        lambda_source = SeasonLambdaSource("Tu Equipo", squad, bridge, real_source, engine.ad_model_)
        orch = SeasonOrchestrator(
            lambda_source, fixtures, squad_roster={"Tu Equipo": squad}, competition="LaLiga",
        )
        return orch.run_monte_carlo(n_sims=1500)

    best_mc = _run(best=True)
    worst_mc = _run(best=False)

    best_champion_p = best_mc.loc[best_mc["team"] == "Tu Equipo", "p_champion"].iloc[0]
    worst_champion_p = worst_mc.loc[worst_mc["team"] == "Tu Equipo", "p_champion"].iloc[0]
    assert best_champion_p > worst_champion_p

    best_avg_pts = best_mc.loc[best_mc["team"] == "Tu Equipo", "avg_pts"].iloc[0]
    worst_avg_pts = worst_mc.loc[worst_mc["team"] == "Tu Equipo", "avg_pts"].iloc[0]
    assert best_avg_pts > worst_avg_pts
