from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd

from mundialytics.statistical_core.betting_value import BettingValueEngine
from mundialytics.statistical_core.distributions import scoreline_distribution
from mundialytics.statistical_core.match_model import MatchOutcomeModel
from mundialytics.statistical_core.player_event_model import PlayerEventModel
from mundialytics.statistical_core.reporting import build_daily_html_report
from mundialytics.statistical_core.team_stats_model import TeamStatsModel
from mundialytics.statistical_core.tournament_simulator import TournamentSimulationConfig, TournamentSimulator
import pytest


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"match_id": "m1", "date": "2026-06-21", "home_team": "Spain", "away_team": "Uruguay", "neutral": 1, "competition": "World Cup", "stage": "Group", "group": "A"},
            {"match_id": "m2", "date": "2026-06-22", "home_team": "Saudi Arabia", "away_team": "Spain", "neutral": 1, "competition": "World Cup", "stage": "Group", "group": "A"},
        ]
    )


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"match_id": "h1", "date": "2024-01-01", "team": "Spain", "opponent": "Uruguay", "player": "Lamine Yamal", "position": "RW", "minutes": 80, "shots": 3, "shots_on_target": 1, "fouls_committed": 1, "yellow_cards": 0, "goals": 1},
            {"match_id": "h1", "date": "2024-01-01", "team": "Spain", "opponent": "Uruguay", "player": "Alvaro Morata", "position": "ST", "minutes": 75, "shots": 4, "shots_on_target": 2, "fouls_committed": 2, "yellow_cards": 0, "goals": 1},
            {"match_id": "h1", "date": "2024-01-01", "team": "Uruguay", "opponent": "Spain", "player": "Federico Valverde", "position": "CM", "minutes": 90, "shots": 2, "shots_on_target": 1, "fouls_committed": 2, "yellow_cards": 1, "goals": 1},
            {"match_id": "h2", "date": "2024-06-01", "team": "Spain", "opponent": "Saudi Arabia", "player": "Lamine Yamal", "position": "RW", "minutes": 85, "shots": 2, "shots_on_target": 1, "fouls_committed": 0, "yellow_cards": 0, "goals": 0},
            {"match_id": "h2", "date": "2024-06-01", "team": "Saudi Arabia", "opponent": "Spain", "player": "Salem Al-Dawsari", "position": "LW", "minutes": 90, "shots": 1, "shots_on_target": 0, "fouls_committed": 1, "yellow_cards": 0, "goals": 0},
        ]
    )


def test_scoreline_probabilities_sum_to_one() -> None:
    dist = scoreline_distribution(1.4, 0.9, max_goals=10, normalize=True)
    total = float(dist.matrix.to_numpy().sum())
    assert abs(total - 1.0) < 1e-12
    assert abs(dist.p_home_win + dist.p_draw + dist.p_away_win - 1.0) < 1e-12


def test_match_model_outputs_valid_distribution() -> None:
    pred, scores = MatchOutcomeModel().fit(_events()).predict_fixtures(_fixtures())
    assert {"p_home_win", "p_draw", "p_away_win", "lambda_home", "lambda_away"}.issubset(pred.columns)
    assert ((pred["p_home_win"] + pred["p_draw"] + pred["p_away_win"] - 1.0).abs() < 1e-6).all()
    assert not scores.empty


def test_team_stats_schema_and_no_corner_invention() -> None:
    mp, _ = MatchOutcomeModel().fit(_events()).predict_fixtures(_fixtures())
    team_stats = TeamStatsModel().fit(_events()).predict_fixtures(_fixtures(), mp)
    assert {"match_id", "team", "market", "expected_count", "availability", "confidence"}.issubset(team_stats.columns)
    corners = team_stats[team_stats["market"].eq("corners")]
    assert not corners.empty
    assert set(corners["availability"]) == {"not_available"}


def test_player_props_only_current_candidates_and_no_retired_players() -> None:
    fixtures = _fixtures()
    lineups = pd.DataFrame(
        [
            {"match_id": "m1", "team": "Spain", "player": "Lamine Yamal", "position": "RW", "started": 1, "expected_minutes": 85},
            {"match_id": "m1", "team": "Uruguay", "player": "Retired Legend", "position": "ST", "started": 0, "expected_minutes": 20, "status": "retired"},
        ]
    )
    mp, _ = MatchOutcomeModel().fit(_events()).predict_fixtures(fixtures)
    ts = TeamStatsModel().fit(_events()).predict_fixtures(fixtures, mp)
    props, warnings = PlayerEventModel().fit(_events()).predict(fixtures, lineups, pd.DataFrame(), ts)
    assert not props.empty
    assert set(props["player"].unique()) == {"lamine yamal"}
    assert any("dropped_non_current_players" in w for w in warnings)


def test_betting_ev_calculation_and_schema() -> None:
    mp, _ = MatchOutcomeModel().fit(_events()).predict_fixtures(_fixtures())
    odds = pd.DataFrame(
        [
            {"match_id": "m1", "market": "1x2", "selection": "home", "line": "", "odds_decimal": 2.6, "bookmaker": "book"},
            {"match_id": "m1", "market": "1x2", "selection": "draw", "line": "", "odds_decimal": 3.2, "bookmaker": "book"},
            {"match_id": "m1", "market": "1x2", "selection": "away", "line": "", "odds_decimal": 2.8, "bookmaker": "book"},
        ]
    )
    edges = BettingValueEngine().evaluate(odds, mp)
    assert {"model_probability", "implied_probability", "edge", "ev", "recommended", "paper_mode"}.issubset(edges.columns)
    first = edges.iloc[0]
    assert abs(first["ev"] - (first["model_probability"] * first["odds_decimal"] - 1.0)) < 1e-12
    assert edges["paper_mode"].all()


def test_tournament_simulation_reproducible_with_seed() -> None:
    mp, _ = MatchOutcomeModel().fit(_events()).predict_fixtures(_fixtures())
    cfg = TournamentSimulationConfig(n_simulations=50, seed=7)
    a, _ = TournamentSimulator(cfg).simulate(_fixtures(), mp)
    b, _ = TournamentSimulator(cfg).simulate(_fixtures(), mp)
    pd.testing.assert_frame_equal(a, b)
    assert "champion_probability" in a.columns


def test_report_generation(tmp_path: Path) -> None:
    mp, _ = MatchOutcomeModel().fit(_events()).predict_fixtures(_fixtures())
    ts = TeamStatsModel().fit(_events()).predict_fixtures(_fixtures(), mp)
    path = build_daily_html_report(tmp_path / "report.html", mp, ts, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"warnings": []})
    assert path.exists()
    # The report was renamed to "Advanced Match Report" in statistical_core/
    # reporting.py; this assertion still carried the v0.20 title. The generator
    # itself works, so the contract is updated rather than the code changed.
    assert "Mundialytics Advanced Match Report" in path.read_text(encoding="utf-8")


def test_run_statistical_matchday_outputs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixtures = tmp_path / "fixtures.csv"
    events = tmp_path / "events.csv"
    lineups = tmp_path / "lineups.csv"
    odds = tmp_path / "odds.csv"
    _fixtures().to_csv(fixtures, index=False)
    _events().to_csv(events, index=False)
    pd.DataFrame(
        [
            {"match_id": "m1", "team": "Spain", "player": "Lamine Yamal", "position": "RW", "started": 1, "expected_minutes": 85},
            {"match_id": "m1", "team": "Uruguay", "player": "Federico Valverde", "position": "CM", "started": 1, "expected_minutes": 90},
        ]
    ).to_csv(lineups, index=False)
    pd.DataFrame(
        [{"match_id": "m1", "market": "player_shots", "selection": "lamine yamal", "line": "1+", "odds_decimal": 1.8, "bookmaker": "book"}]
    ).to_csv(odds, index=False)
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(root / "scripts" / "run_statistical_matchday.py"),
        "--fixtures",
        str(fixtures),
        "--lineups",
        str(lineups),
        "--odds",
        str(odds),
        "--historical-events",
        str(events),
        "--out-dir",
        str(out_dir),
        "--n-simulations",
        "20",
    ]
    subprocess.run(cmd, check=True, cwd=root)
    for name in ["match_predictions.csv", "team_stats_predictions.csv", "player_event_predictions.csv", "betting_edges.csv", "recommended_picks.csv", "daily_report.html", "audit_report.json"]:
        assert (out_dir / name).exists()


def test_v021_identity_resolves_accents_hyphens_and_full_names() -> None:
    fixtures = pd.DataFrame([
        {"match_id": "m3", "date": "2026-06-21", "home_team": "Spain", "away_team": "Saudi Arabia", "neutral": 1, "competition": "World Cup", "stage": "Group"},
        {"match_id": "m4", "date": "2026-06-22", "home_team": "Uruguay", "away_team": "Spain", "neutral": 1, "competition": "World Cup", "stage": "Group"},
    ])
    events = pd.DataFrame([
        {"match_id": "h1", "team": "Spain", "player": "Álvaro Borja Morata Martín", "position": "ST", "minutes": 994, "shots": 40, "shots_on_target": 15, "fouls_committed": 20, "yellow_cards": 2, "goals": 10},
        {"match_id": "h2", "team": "Juventus", "player": "Alvaro Borja Morata Martin", "position": "ST", "minutes": 1547, "shots": 55, "shots_on_target": 21, "fouls_committed": 30, "yellow_cards": 3, "goals": 12},
        {"match_id": "h3", "team": "Uruguay", "player": "Federico Santiago Valverde Dipetta", "position": "CM", "minutes": 799, "shots": 25, "shots_on_target": 8, "fouls_committed": 35, "yellow_cards": 5, "goals": 3},
        {"match_id": "h4", "team": "Real Madrid", "player": "Federico Santiago Valverde Dipetta", "position": "CM", "minutes": 328, "shots": 10, "shots_on_target": 4, "fouls_committed": 11, "yellow_cards": 1, "goals": 1},
        {"match_id": "h5", "team": "Saudi Arabia", "player": "Salem Mohammed Al Dawsari", "position": "LW", "minutes": 540, "shots": 20, "shots_on_target": 7, "fouls_committed": 18, "yellow_cards": 2, "goals": 4},
        {"match_id": "h6", "team": "Saudi Arabia", "player": "Nasser Al Dawsari", "position": "CM", "minutes": 5, "shots": 0, "shots_on_target": 0, "fouls_committed": 1, "yellow_cards": 0, "goals": 0},
    ])
    lineups = pd.DataFrame([
        {"match_id": "m3", "team": "Spain", "player": "Álvaro Morata", "position": "ST", "started": 1, "expected_minutes": 75},
        {"match_id": "m3", "team": "Saudi Arabia", "player": "Salem Al-Dawsari", "position": "LW", "started": 1, "expected_minutes": 85},
        {"match_id": "m4", "team": "Uruguay", "player": "Federico Valverde", "position": "CM", "started": 1, "expected_minutes": 90},
    ])
    mp, _ = MatchOutcomeModel().fit(events).predict_fixtures(fixtures)
    ts = TeamStatsModel().fit(events).predict_fixtures(fixtures, mp)
    props, _ = PlayerEventModel().fit(events).predict(fixtures, lineups, pd.DataFrame(), ts)
    shots = props[props["market"].eq("player_shots")].copy()
    morata = shots[shots["player_input_name"].eq("alvaro morata")].iloc[0]
    salem = shots[shots["player_input_name"].eq("salem al dawsari")].iloc[0]
    valverde = shots[shots["player_input_name"].eq("federico valverde")].iloc[0]
    assert morata["canonical_player_name"] == "alvaro borja morata martin"
    assert float(morata["sample_size_minutes"]) == 2541.0
    assert "juventus" in morata["historical_teams_used"] and "spain" in morata["historical_teams_used"]
    assert salem["canonical_player_name"] == "salem mohammed al dawsari"
    assert float(salem["sample_size_minutes"]) == 540.0
    assert valverde["canonical_player_name"] == "federico santiago valverde dipetta"
    assert float(valverde["sample_size_minutes"]) == 1127.0
    assert set(shots["identity_status"]) == {"matched"}


def test_v021_betting_blocks_zero_sample_player_pick() -> None:
    fixtures = _fixtures().iloc[[0]].copy()
    events = _events().copy()
    lineups = pd.DataFrame([
        {"match_id": "m1", "team": "Spain", "player": "Unknown Prospect", "position": "ST", "started": 1, "expected_minutes": 90},
    ])
    mp, _ = MatchOutcomeModel().fit(events).predict_fixtures(fixtures)
    ts = TeamStatsModel().fit(events).predict_fixtures(fixtures, mp)
    props, _ = PlayerEventModel().fit(events).predict(fixtures, lineups, pd.DataFrame(), ts)
    odds = pd.DataFrame([
        {"match_id": "m1", "market": "player_shots", "selection": "Unknown Prospect", "line": "1+", "odds_decimal": 3.5, "bookmaker": "book"},
    ])
    edges = BettingValueEngine().evaluate(odds, mp, ts, props)
    assert len(edges) == 1
    assert not bool(edges.iloc[0]["recommended"])
    assert "sample_size_zero_no_player_pick" in str(edges.iloc[0]["warnings"])
    assert edges.iloc[0]["confidence"] == "low"


def test_v022_temporal_evaluation_and_calibration_outputs() -> None:
    from mundialytics.statistical_core.evaluation import TemporalEvaluationConfig, apply_match_calibration, evaluate_match_model_temporal

    rows = []
    # Build enough historical matches for a temporal holdout. Use duplicated
    # player rows per team to mimic the processed event file shape.
    for i in range(30):
        date = f"2024-01-{(i % 28) + 1:02d}"
        rows.extend(
            [
                {"match_id": f"h{i}", "date": date, "team": "Spain", "opponent": "Uruguay", "player": "A", "minutes": 90, "shots": 10, "shots_on_target": 4, "fouls_committed": 10, "yellow_cards": 1, "goals": 2 if i % 3 else 1},
                {"match_id": f"h{i}", "date": date, "team": "Uruguay", "opponent": "Spain", "player": "B", "minutes": 90, "shots": 8, "shots_on_target": 3, "fouls_committed": 11, "yellow_cards": 2, "goals": 1 if i % 2 else 0},
            ]
        )
    events = pd.DataFrame(rows)
    preds, bins, summary, calibration = evaluate_match_model_temporal(events, TemporalEvaluationConfig(test_fraction=0.25, min_train_matches=10))
    assert summary["status"] == "completed"
    assert not preds.empty
    assert not bins.empty
    assert calibration["status"] == "fitted_temporal_holdout"
    current, _ = MatchOutcomeModel().fit(events).predict_fixtures(_fixtures().iloc[[0]])
    calibrated = apply_match_calibration(current, calibration)
    assert abs(float(calibrated[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1).iloc[0]) - 1.0) < 1e-9


def test_v022_competition_forecast_top_scorer_probabilities() -> None:
    from mundialytics.statistical_core.scorer_model import CompetitionForecastEngine, ScorerForecastConfig

    player_events = pd.DataFrame(
        [
            {"match_id": "m1", "team": "Spain", "player": "Striker A", "position": "ST", "market": "player_shots", "expected_count": 3.0, "sample_size_minutes": 900},
            {"match_id": "m1", "team": "Uruguay", "player": "Winger B", "position": "LW", "market": "player_shots", "expected_count": 2.0, "sample_size_minutes": 500},
            {"match_id": "m1", "team": "Spain", "player": "Striker A", "position": "ST", "market": "player_shots_on_target", "expected_count": 1.1, "sample_size_minutes": 900},
        ]
    )
    tournament = pd.DataFrame(
        [
            {"team": "spain", "qf_probability": 0.8, "sf_probability": 0.5, "final_probability": 0.25, "champion_probability": 0.1, "qualify_group_probability": 0.9, "expected_points": 6, "expected_goals_for": 5},
            {"team": "uruguay", "qf_probability": 0.4, "sf_probability": 0.2, "final_probability": 0.1, "champion_probability": 0.05, "qualify_group_probability": 0.7, "expected_points": 4, "expected_goals_for": 3},
        ]
    )
    top, awards, comp = CompetitionForecastEngine(ScorerForecastConfig(n_simulations=500, seed=1)).build_outputs(player_events, tournament, pd.DataFrame())
    assert not top.empty
    assert abs(float(top["top_scorer_probability"].sum()) - 1.0) < 1e-9
    assert "golden_boot" in set(awards["award"])
    assert "best_goalkeeper_clean_sheets" in set(awards["award"])
    assert not comp.empty


def test_v022_run_script_clean_out_dir_and_no_demo_picks(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixtures = tmp_path / "fixtures.csv"
    events = tmp_path / "events.csv"
    lineups = tmp_path / "lineups.csv"
    odds = tmp_path / "odds.csv"
    _fixtures().to_csv(fixtures, index=False)
    _events().to_csv(events, index=False)
    pd.DataFrame([{"match_id": "m1", "team": "Spain", "player": "Lamine Yamal", "position": "RW", "started": 1, "expected_minutes": 85}]).to_csv(lineups, index=False)
    pd.DataFrame([{"match_id": "m1", "market": "player_shots", "selection": "lamine yamal", "line": "1+", "odds_decimal": 2.5, "bookmaker": "demo_book"}]).to_csv(odds, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "stale.txt"
    stale.write_text("delete me")
    cmd = [
        sys.executable,
        str(root / "scripts" / "run_statistical_matchday.py"),
        "--fixtures", str(fixtures),
        "--lineups", str(lineups),
        "--odds", str(odds),
        "--historical-events", str(events),
        "--out-dir", str(out_dir),
        "--clean-out-dir",
        "--no-demo-picks",
        "--n-simulations", "20",
    ]
    subprocess.run(cmd, check=True, cwd=root)
    assert not stale.exists()
    picks = pd.read_csv(out_dir / "recommended_picks.csv")
    assert picks.empty
    assert (out_dir / "top_scorer_predictions.csv").exists()
    assert (out_dir / "competition_summary.csv").exists()


def test_v023_match_model_config_reduces_extreme_lambdas() -> None:
    events = pd.DataFrame([
        {"match_id": f"h{i}", "date": f"2024-01-{(i % 28) + 1:02d}", "team": "A", "opponent": "B", "player": "a", "goals": 8, "minutes": 90, "shots": 20, "shots_on_target": 10, "fouls_committed": 5, "yellow_cards": 0}
        for i in range(8)
    ] + [
        {"match_id": f"h{i}", "date": f"2024-01-{(i % 28) + 1:02d}", "team": "B", "opponent": "A", "player": "b", "goals": 0, "minutes": 90, "shots": 2, "shots_on_target": 0, "fouls_committed": 10, "yellow_cards": 2}
        for i in range(8)
    ])
    fixtures = pd.DataFrame([{"match_id": "m", "date": "2026-01-01", "home_team": "A", "away_team": "B", "neutral": 1}])
    pred, _ = MatchOutcomeModel(goal_cap=3.5, attack_cap=2.0, defense_cap=2.0, profile_shrinkage_k=10, low_sample_blend_k=10).fit(events).predict_fixtures(fixtures)
    assert float(pred.iloc[0]["lambda_home"]) <= 3.5
    assert float(pred.iloc[0]["lambda_away"]) <= 3.5
    assert "model_config" in MatchOutcomeModel(goal_cap=3.5).fit(events).audit


def test_v023_model_lab_outputs(tmp_path: Path) -> None:
    from mundialytics.statistical_core.model_lab import run_model_lab

    rows = []
    for i in range(24):
        date = f"2024-02-{(i % 28) + 1:02d}"
        rows.extend([
            {"match_id": f"h{i}", "date": date, "team": "Spain", "opponent": "Uruguay", "player": "A", "minutes": 90, "shots": 10, "shots_on_target": 4, "fouls_committed": 10, "yellow_cards": 1, "goals": 2 if i % 3 else 1},
            {"match_id": f"h{i}", "date": date, "team": "Uruguay", "opponent": "Spain", "player": "B", "minutes": 90, "shots": 8, "shots_on_target": 3, "fouls_committed": 11, "yellow_cards": 2, "goals": 1 if i % 2 else 0},
        ])
    leaderboard, best = run_model_lab(pd.DataFrame(rows), tmp_path, n_trials=2, test_fraction=0.25, min_train_matches=10)
    assert not leaderboard.empty
    assert best["status"] == "completed"
    assert (tmp_path / "experiment_leaderboard.csv").exists()
    assert (tmp_path / "best_model_config.json").exists()
    assert (tmp_path / "model_lab_report.html").exists()


def test_event_evaluation_outputs_market_policy() -> None:
    from mundialytics.statistical_core.event_evaluation import EventEvaluationConfig, evaluate_event_models_temporal

    rows = []
    dates = pd.date_range("2024-01-01", periods=8, freq="30D")
    for i, d in enumerate(dates):
        for team, opp, goals, shots, sot, fouls, cards in [
            ("Spain", "Uruguay", 2 + (i % 2), 13 + i % 3, 5, 10, 1),
            ("Uruguay", "Spain", 1, 9 + i % 2, 3, 12, 2),
        ]:
            rows.append({"match_id": f"m{i}", "date": str(d.date()), "team": team, "opponent": opp, "player": f"{team} Player A", "position": "ST", "minutes": 80, "started": 1, "shots": shots // 2, "shots_on_target": max(1, sot // 2), "fouls_committed": fouls // 2, "yellow_cards": 0, "goals": goals})
            rows.append({"match_id": f"m{i}", "date": str(d.date()), "team": team, "opponent": opp, "player": f"{team} Player B", "position": "CM", "minutes": 90, "started": 1, "shots": shots - shots // 2, "shots_on_target": sot - max(1, sot // 2), "fouls_committed": fouls - fouls // 2, "yellow_cards": cards, "goals": 0})
    events = pd.DataFrame(rows)
    team_scored, team_lines, player_scored, player_lines, summary = evaluate_event_models_temporal(events, EventEvaluationConfig(min_train_matches=3, test_fraction=0.25))
    assert summary["status"] == "completed"
    assert not team_scored.empty
    assert not player_scored.empty
    assert "yellow_cards" in summary["team_event_performance"]["count_metrics_by_market"]
    assert "player_yellow_card" in summary["player_prop_performance"]["prop_metrics_by_market"]
    assert "market_policy" in summary


def test_team_and_player_event_models_accept_configs() -> None:
    mp, _ = MatchOutcomeModel().fit(_events()).predict_fixtures(_fixtures())
    team_stats = TeamStatsModel(own_weight=0.65, profile_shrinkage_k=5.0).fit(_events()).predict_fixtures(_fixtures(), mp)
    assert not team_stats.empty
    lineups = pd.DataFrame([{"match_id": "m1", "team": "Spain", "player": "Lamine Yamal", "position": "RW", "started": 1, "expected_minutes": 80}])
    props, _ = PlayerEventModel(share_weight=0.50, yellow_card_cap=0.50).fit(_events()).predict(_fixtures(), lineups, pd.DataFrame(), team_stats)
    assert not props.empty
    assert props["expected_count"].notna().all()


def test_v025_player_prop_champion_lab_selects_market_champions(tmp_path: Path) -> None:
    from mundialytics.statistical_core.player_prop_champion import (
        ChampionPropConfig,
        run_player_prop_champion_lab,
        write_player_prop_champion_outputs,
    )

    rows = []
    dates = pd.date_range("2024-01-01", periods=14, freq="14D")
    for i, d in enumerate(dates):
        for team, opp in [("Spain", "Uruguay"), ("Uruguay", "Spain")]:
            rows.append({
                "match_id": f"m{i}", "date": str(d.date()), "competition": "UEFA Euro", "team_scope": "national",
                "team_type": "national_team", "competition_context": "international_national_tournament", "gender": "men",
                "team": team, "opponent": opp, "player": f"{team} Striker", "player_id_global": f"{team.lower()}_st",
                "position": "ST", "started": 1, "minutes": 80, "shots": 2 + (i % 2), "shots_on_target": 1, "fouls_committed": 1, "yellow_cards": 0,
            })
            rows.append({
                "match_id": f"m{i}", "date": str(d.date()), "competition": "UEFA Euro", "team_scope": "national",
                "team_type": "national_team", "competition_context": "international_national_tournament", "gender": "men",
                "team": team, "opponent": opp, "player": f"{team} Mid", "player_id_global": f"{team.lower()}_mid",
                "position": "CM", "started": 1, "minutes": 85, "shots": i % 2, "shots_on_target": 0, "fouls_committed": 2, "yellow_cards": 1 if i % 4 == 0 else 0,
            })
    cfg = ChampionPropConfig(min_train_matches=5, max_test_matches=3, max_calibration_matches=3, min_calibration_rows=10, min_group_rows=10, min_segment_rows=4)
    leaderboard, champions, segments, payload = run_player_prop_champion_lab(pd.DataFrame(rows), cfg)
    assert payload["status"] == "completed"
    assert not leaderboard.empty
    assert not champions.empty
    assert {"player_shots", "player_shots_on_target", "player_fouls_committed", "player_yellow_card"}.issubset(set(champions["market"]))
    paths = write_player_prop_champion_outputs(tmp_path, leaderboard, champions, segments, payload)
    assert (tmp_path / "prediction_registry.json").exists()
    assert "player_prop_champion_report.html" in paths


def test_v026_dixon_coles_optional_draw_correction_changes_distribution() -> None:
    base = scoreline_distribution(1.2, 1.2, max_goals=6, normalize=True, dixon_coles_rho=0.0)
    dc = scoreline_distribution(1.2, 1.2, max_goals=6, normalize=True, dixon_coles_rho=-0.08)
    assert abs(float(dc.matrix.to_numpy().sum()) - 1.0) < 1e-9
    assert dc.p_draw > base.p_draw


def test_v026_player_prop_count_links_and_starter_minutes() -> None:
    from mundialytics.statistical_core.player_prop_champion import (
        ChampionPropConfig,
        _predict_prop_rows,
        _probability_1plus_from_count,
        run_player_prop_champion_lab,
    )

    assert _probability_1plus_from_count(1.0, "player_shots", {"count_distribution": "negative_binomial", "nb_shape": 2.0}) < _probability_1plus_from_count(1.0, "player_shots", {})

    rows = []
    dates = pd.date_range("2024-01-01", periods=12, freq="14D")
    for i, d in enumerate(dates):
        for team, opp in [("Spain", "Uruguay"), ("Uruguay", "Spain")]:
            rows.append({
                "match_id": f"m{i}", "date": str(d.date()), "competition": "UEFA Euro", "team_scope": "national",
                "team_type": "national_team", "competition_context": "international_national_tournament", "gender": "men",
                "team": team, "opponent": opp, "player": f"{team} Starter", "player_id_global": f"{team.lower()}_starter",
                "position": "RW", "started": 1, "minutes": 82, "shots": 2, "shots_on_target": 1, "fouls_committed": 1, "yellow_cards": 0,
            })
            rows.append({
                "match_id": f"m{i}", "date": str(d.date()), "competition": "UEFA Euro", "team_scope": "national",
                "team_type": "national_team", "competition_context": "international_national_tournament", "gender": "men",
                "team": team, "opponent": opp, "player": f"{team} Sub", "player_id_global": f"{team.lower()}_sub",
                "position": "RW", "started": 0, "minutes": 24, "shots": i % 2, "shots_on_target": 0, "fouls_committed": 1, "yellow_cards": 0,
            })
    df = pd.DataFrame(rows)
    train = df[df["match_id"].isin([f"m{i}" for i in range(8)])]
    target = df[df["match_id"].isin(["m8"])]
    pred = _predict_prop_rows(train, target, {"name": "starter_role_test", "minutes_model": "starter_role"}, "test")
    assert not pred.empty
    by_started = pred.groupby("started")["expected_minutes"].mean().to_dict()
    assert by_started[1] > by_started[0]
    cfg = ChampionPropConfig(min_train_matches=5, max_test_matches=2, max_calibration_matches=2, min_calibration_rows=5, min_group_rows=5, min_segment_rows=4, n_trials=2)
    leaderboard, champions, _, payload = run_player_prop_champion_lab(df, cfg)
    assert payload["status"] == "completed"
    assert not leaderboard.empty
    assert not champions.empty


def test_v027_rolling_model_lab_smoke(tmp_path: Path) -> None:
    from mundialytics.statistical_core.rolling_validation import RollingMatchConfig, rolling_match_backtest, run_rolling_model_lab

    rows = []
    dates = pd.date_range("2023-01-01", periods=18, freq="14D")
    for i, d in enumerate(dates):
        # Two deterministic teams plus two rows per match so the event table can
        # be collapsed to team goals.
        for team, opp, goals in [("Spain", "Uruguay", 2 + (i % 2)), ("Uruguay", "Spain", 1)]:
            rows.append({
                "match_id": f"m{i}", "date": str(d.date()), "team": team, "opponent": opp,
                "player": f"{team} Player A", "position": "ST", "minutes": 80, "started": 1,
                "goals": goals, "shots": 3, "shots_on_target": 1, "fouls_committed": 1, "yellow_cards": 0,
            })
    events = pd.DataFrame(rows)
    cfg = RollingMatchConfig(min_train_matches=6, calibration_matches=3, test_matches=2, step_matches=2, max_folds=2)
    preds, folds, summary = rolling_match_backtest(events, cfg)
    assert summary["status"] == "completed"
    assert not preds.empty
    assert not folds.empty
    leaderboard, best = run_rolling_model_lab(events, tmp_path, n_trials=2, cfg=cfg)
    assert not leaderboard.empty
    assert best["status"] == "completed"
    assert (tmp_path / "rolling_model_leaderboard.csv").exists()
    assert (tmp_path / "best_rolling_model_config.json").exists()


def test_v027_prediction_registry_contains_segment_policies(tmp_path: Path) -> None:
    from mundialytics.statistical_core.player_prop_champion import (
        ChampionPropConfig,
        run_player_prop_champion_lab,
        write_player_prop_champion_outputs,
    )
    import json

    rows = []
    dates = pd.date_range("2024-01-01", periods=14, freq="14D")
    for i, d in enumerate(dates):
        for team, opp in [("Spain", "Uruguay"), ("Uruguay", "Spain")]:
            rows.append({
                "match_id": f"m{i}", "date": str(d.date()), "competition": "UEFA Euro", "team_scope": "national",
                "team_type": "national_team", "competition_context": "international_national_tournament", "gender": "men",
                "team": team, "opponent": opp, "player": f"{team} Starter", "player_id_global": f"{team.lower()}_starter",
                "position": "RW", "started": 1, "minutes": 82, "shots": 2, "shots_on_target": 1, "fouls_committed": 1, "yellow_cards": 0,
            })
            rows.append({
                "match_id": f"m{i}", "date": str(d.date()), "competition": "UEFA Euro", "team_scope": "national",
                "team_type": "national_team", "competition_context": "international_national_tournament", "gender": "men",
                "team": team, "opponent": opp, "player": f"{team} Defender", "player_id_global": f"{team.lower()}_def",
                "position": "CB", "started": 1, "minutes": 90, "shots": 0, "shots_on_target": 0, "fouls_committed": 2, "yellow_cards": 1 if i % 3 == 0 else 0,
            })
    cfg = ChampionPropConfig(min_train_matches=5, max_test_matches=3, max_calibration_matches=3, min_calibration_rows=10, min_group_rows=10, min_segment_rows=4, n_trials=2)
    leaderboard, champions, segments, payload = run_player_prop_champion_lab(pd.DataFrame(rows), cfg)
    assert not segments.empty
    write_player_prop_champion_outputs(tmp_path, leaderboard, champions, segments, payload)
    reg = json.loads((tmp_path / "prediction_registry.json").read_text())
    assert reg["version"] == "v0.28_prediction_registry"
    assert "segment_policies" in reg


def test_v028_position_groups_and_goalkeeper_guardrail() -> None:
    from mundialytics.statistical_core.player_event_model import _position_key, _position_group, PlayerEventModel

    assert _position_key("Left Wing") == "lw"
    assert _position_group("Left Wing") == "winger"
    assert _position_key("Right Center Midfield") == "cm"
    assert _position_group("Right Center Midfield") == "central_midfield"
    assert _position_group("Goalkeeper") == "goalkeeper"

    hist = pd.DataFrame([
        {"match_id":"m1","date":"2024-01-01","team":"Spain","opponent":"France","player":"Keeper A","position":"Goalkeeper","minutes":90,"started":1,"shots":3,"shots_on_target":2,"fouls_committed":0,"yellow_cards":0},
        {"match_id":"m1","date":"2024-01-01","team":"Spain","opponent":"France","player":"Winger A","position":"Left Wing","minutes":80,"started":1,"shots":2,"shots_on_target":1,"fouls_committed":1,"yellow_cards":0},
    ])
    fixtures = pd.DataFrame([{"match_id":"f1","date":"2024-02-01","home_team":"Spain","away_team":"France"}])
    lineups = pd.DataFrame([{"match_id":"f1","team":"Spain","opponent":"France","player":"Keeper A","position":"Goalkeeper","expected_minutes":90,"started":1}])
    team_stats = pd.DataFrame([
        {"match_id":"f1","team":"Spain","market":"shots","expected_count":10,"availability":"available"},
        {"match_id":"f1","team":"Spain","market":"shots_on_target","expected_count":4,"availability":"available"},
        {"match_id":"f1","team":"Spain","market":"fouls","expected_count":12,"availability":"available"},
        {"match_id":"f1","team":"Spain","market":"yellow_cards","expected_count":2,"availability":"available"},
    ])
    preds, warnings = PlayerEventModel().fit(hist).predict(fixtures, lineups, pd.DataFrame(), team_stats)
    shot = preds[preds["market"].eq("player_shots")].iloc[0]
    sot = preds[preds["market"].eq("player_shots_on_target")].iloc[0]
    assert shot["position_group"] == "goalkeeper"
    assert float(shot["safe_probability"]) == 0.0
    assert float(sot["safe_probability"]) == 0.0
    assert "role_guardrail_goalkeeper_attacking_prop_blocked" in str(shot["warnings"])


def test_v029_dynamic_lines_have_structured_evidence_and_not_available_corners() -> None:
    from mundialytics.statistical_core.dynamic_lines import DynamicLineConfig, build_dynamic_market_lines

    fixtures = _fixtures().iloc[[0]].copy()
    events = _events().copy()
    mp, scores = MatchOutcomeModel().fit(events).predict_fixtures(fixtures)
    ts = TeamStatsModel().fit(events).predict_fixtures(fixtures, mp)
    lineups = pd.DataFrame([
        {"match_id": "m1", "team": "Spain", "player": "Lamine Yamal", "position": "RW", "started": 1, "expected_minutes": 85},
    ])
    props, _ = PlayerEventModel().fit(events).predict(fixtures, lineups, pd.DataFrame(), ts)
    lines = build_dynamic_market_lines(fixtures, mp, scores, ts, props, events, config=DynamicLineConfig(recent_n=3, h2h_years=5))
    assert not lines.empty
    assert {"market", "scope", "line", "over_under", "model_probability", "recent_hit_rate", "evidence_tags", "reason_code"}.issubset(lines.columns)
    assert ((lines[lines["availability"].eq("available")]["model_probability"].dropna() >= 0).all())
    assert ((lines[lines["availability"].eq("available")]["model_probability"].dropna() <= 1).all())
    goal_lines = lines[(lines["market"].eq("goals")) & (lines["scope"].eq("match"))]
    assert {0.5, 1.5, 2.5, 3.5}.issubset(set(goal_lines["line"].astype(float)))
    corners = lines[lines["market"].eq("corners")]
    assert not corners.empty
    assert set(corners["availability"].unique()) == {"not_available"}


def test_v029_dynamic_lines_block_goalkeeper_attacking_props() -> None:
    from mundialytics.statistical_core.dynamic_lines import build_dynamic_market_lines

    fixtures = _fixtures().iloc[[0]].copy()
    events = _events().copy()
    mp, scores = MatchOutcomeModel().fit(events).predict_fixtures(fixtures)
    ts = TeamStatsModel().fit(events).predict_fixtures(fixtures, mp)
    props = pd.DataFrame([
        {
            "match_id": "m1", "team": "Spain", "opponent": "Uruguay", "player": "Goalkeeper One",
            "market": "player_shots", "expected_count": 0.0, "position_group": "goalkeeper",
            "expected_minutes": 90, "sample_size_minutes": 900, "warnings": "role_guardrail_goalkeeper_attacking_prop_blocked",
        }
    ])
    lines = build_dynamic_market_lines(fixtures, mp, scores, ts, props, events)
    gk = lines[(lines["scope"].eq("player")) & (lines["market"].eq("player_shots"))]
    assert not gk.empty
    assert set(gk["availability"].unique()) == {"not_available"}
    assert gk["evidence_tags"].str.contains("role_guardrail_blocked").any()


def test_v030_player_prop_evidence_is_line_specific() -> None:
    from mundialytics.statistical_core.dynamic_lines import DynamicLineConfig, build_dynamic_market_lines

    fixtures = _fixtures().iloc[[0]].copy()
    events = _events().copy()
    mp, scores = MatchOutcomeModel().fit(events).predict_fixtures(fixtures)
    ts = TeamStatsModel().fit(events).predict_fixtures(fixtures, mp)
    lineups = pd.DataFrame([
        {"match_id": "m1", "team": "Spain", "player": "Lamine Yamal", "position": "RW", "started": 1, "expected_minutes": 85},
    ])
    props, _ = PlayerEventModel().fit(events).predict(fixtures, lineups, pd.DataFrame(), ts)
    lines = build_dynamic_market_lines(fixtures, mp, scores, ts, props, events, config=DynamicLineConfig(recent_n=10, min_context_sample=1, min_strong_context_sample=2))
    player_lines = lines[(lines["market"].eq("player_shots")) & (lines["player"].eq("lamine yamal")) & (lines["over_under"].eq("over"))]
    assert not player_lines.empty
    over_05 = player_lines[player_lines["line"].astype(float).eq(0.5)].iloc[0]
    over_25 = player_lines[player_lines["line"].astype(float).eq(2.5)].iloc[0]
    assert over_05["recent_hit_rate"] != over_25["recent_hit_rate"]
    assert over_05["recent_hit_rate"] == "2/2"
    assert over_25["recent_hit_rate"] == "1/2"


def test_v030_signal_label_is_separate_from_price_value_label_and_total_goals_odds_attach() -> None:
    from mundialytics.statistical_core.dynamic_lines import build_dynamic_market_lines

    fixtures = _fixtures().iloc[[0]].copy()
    events = _events().copy()
    mp, scores = MatchOutcomeModel().fit(events).predict_fixtures(fixtures)
    ts = TeamStatsModel().fit(events).predict_fixtures(fixtures, mp)
    odds = pd.DataFrame([
        {"match_id": "m1", "market": "total_goals", "selection": "over", "line": 2.5, "odds_decimal": 2.5, "bookmaker": "book"},
    ])
    lines = build_dynamic_market_lines(fixtures, mp, scores, ts, pd.DataFrame(), events, odds=odds)
    row = lines[(lines["market"].eq("goals")) & (lines["scope"].eq("match")) & (lines["line"].astype(float).eq(2.5)) & (lines["over_under"].eq("over"))].iloc[0]
    assert "signal_label" in lines.columns
    assert float(row["book_odds"]) == 2.5
    assert row["value_label"] in {"high_value", "medium_value", "fair_price", "no_value"}
    team_rows = lines[(lines["market"].eq("goals")) & (lines["scope"].eq("team")) & (lines["line"].astype(float).eq(2.5))]
    if not team_rows.empty:
        assert team_rows["book_odds"].isna().all()


def test_v031_demo_odds_are_labelled_not_real_value() -> None:
    from mundialytics.statistical_core.dynamic_lines import build_dynamic_market_lines

    fixtures = _fixtures().iloc[[0]].copy()
    events = _events().copy()
    mp, scores = MatchOutcomeModel().fit(events).predict_fixtures(fixtures)
    ts = TeamStatsModel().fit(events).predict_fixtures(fixtures, mp)
    odds = pd.DataFrame([
        {"match_id": "m1", "market": "total_goals", "selection": "over", "line": 2.5, "odds_decimal": 3.0, "bookmaker": "demo_book"},
    ])
    lines = build_dynamic_market_lines(fixtures, mp, scores, ts, pd.DataFrame(), events, odds=odds)
    row = lines[(lines["market"].eq("goals")) & (lines["scope"].eq("match")) & (lines["line"].astype(float).eq(2.5)) & (lines["over_under"].eq("over"))].iloc[0]
    assert float(row["book_odds"]) == 3.0
    assert row["value_label"] == "demo_odds_only"
    assert row["value_reason_code"] == "demo_odds_detected_not_for_real_value"
    assert "demo_odds_only" in row["evidence_tags"]


def test_v031_player_evidence_source_falls_back_to_canonical_history() -> None:
    from mundialytics.statistical_core.dynamic_lines import DynamicLineConfig, build_dynamic_market_lines

    fixtures = pd.DataFrame([{"match_id": "f1", "date": "2024-02-01", "home_team": "Spain", "away_team": "Saudi Arabia"}])
    hist = pd.DataFrame([
        {"match_id": "h1", "date": "2024-01-01", "team": "Atletico Madrid", "opponent": "Real Madrid", "player": "Alvaro Morata", "shots": 3, "shots_on_target": 1, "fouls_committed": 1, "yellow_cards": 0, "goals": 1},
        {"match_id": "h2", "date": "2024-01-10", "team": "Atletico Madrid", "opponent": "Barcelona", "player": "Alvaro Morata", "shots": 1, "shots_on_target": 0, "fouls_committed": 0, "yellow_cards": 0, "goals": 0},
    ])
    mp, scores = MatchOutcomeModel().fit(hist).predict_fixtures(fixtures)
    ts = TeamStatsModel().fit(hist).predict_fixtures(fixtures, mp)
    props = pd.DataFrame([{
        "match_id": "f1", "team": "Spain", "opponent": "Saudi Arabia", "player": "Alvaro Morata",
        "market": "player_shots", "expected_count": 1.4, "position_group": "forward", "expected_minutes": 70,
        "sample_size_minutes": 500, "warnings": "",
    }])
    lines = build_dynamic_market_lines(fixtures, mp, scores, ts, props, hist, config=DynamicLineConfig(min_context_sample=3, recent_n=10))
    row = lines[(lines["market"].eq("player_shots")) & (lines["over_under"].eq("over")) & (lines["line"].astype(float).eq(0.5))].iloc[0]
    assert row["recent_hit_rate"] == "2/2"
    assert row["recent_evidence_source"] == "canonical_player_recent"


def test_v033_generic_roster_position_uses_historical_frequent_position() -> None:
    from mundialytics.statistical_core.player_event_model import PlayerEventModel

    hist = pd.DataFrame([
        {"match_id":"h1","date":"2024-01-01","team":"Spain","opponent":"France","player":"Wide Player","position":"Left Wing","minutes":90,"started":1,"shots":3,"shots_on_target":1,"fouls_committed":1,"yellow_cards":0},
        {"match_id":"h2","date":"2024-02-01","team":"Spain","opponent":"Italy","player":"Wide Player","position":"Left Wing","minutes":80,"started":1,"shots":2,"shots_on_target":1,"fouls_committed":1,"yellow_cards":0},
    ])
    fixtures = pd.DataFrame([{"match_id":"m1","date":"2026-06-20","home_team":"Spain","away_team":"France","competition":"FIFA World Cup","team_type":"national_team","competition_context":"international_national_tournament","gender":"men"}])
    squads = pd.DataFrame([{"match_id":"m1","team":"Spain","player":"Wide Player","position":"F","started":0,"expected_minutes":35}])
    team_stats = pd.DataFrame([
        {"match_id":"m1","team":"Spain","market":"shots","expected_count":12,"availability":"available"},
        {"match_id":"m1","team":"Spain","market":"shots_on_target","expected_count":5,"availability":"available"},
        {"match_id":"m1","team":"Spain","market":"fouls","expected_count":11,"availability":"available"},
        {"match_id":"m1","team":"Spain","market":"yellow_cards","expected_count":2,"availability":"available"},
    ])
    preds, _ = PlayerEventModel().fit(hist).predict(fixtures, pd.DataFrame(), squads, team_stats)
    row = preds[preds["market"].eq("player_shots")].iloc[0]
    assert row["input_position"] == "F"
    assert row["position_group"] == "winger"
    assert row["position_source"] == "historical_frequent_position_fallback"
    assert row["player_input_source"] == "squads"
    assert row["player_selection_confidence"] in {"low", "medium_low", "very_low"}
