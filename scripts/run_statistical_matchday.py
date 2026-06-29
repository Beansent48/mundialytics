from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mundialytics.statistical_core import (  # noqa: E402
    BettingValueEngine,
    CompetitionForecastEngine,
    MatchOutcomeModel,
    PlayerEventModel,
    TeamStatsModel,
    TournamentSimulationConfig,
    TournamentSimulator,
)
from mundialytics.statistical_core.evaluation import apply_match_calibration, load_calibration  # noqa: E402
from mundialytics.statistical_core.reporting import build_daily_html_report  # noqa: E402
from mundialytics.statistical_core.matchday_summary import build_matchday_summary  # noqa: E402
from mundialytics.statistical_core.tournament_report import build_tournament_report  # noqa: E402
from mundialytics.statistical_core.dynamic_lines import DynamicLineConfig, build_dynamic_market_lines  # noqa: E402
from mundialytics.statistical_core.schemas import read_csv_optional, standardize_fixtures, write_json  # noqa: E402
from mundialytics.statistical_core.simulation_contract import build_simulator_contract_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Mundialytics v0.48.4 statistical simulator core in paper mode.")
    p.add_argument("--fixtures", required=True, help="Manual fixtures.csv")
    p.add_argument("--lineups", default=None, help="current_lineups.csv with expected_minutes")
    p.add_argument("--squads", default=None, help="squads.csv fallback if lineups are unavailable")
    p.add_argument("--odds", default=None, help="odds.csv for paper-mode value comparison")
    p.add_argument("--tournament-config", default=None, help="Optional tournament_config.csv with group/stage metadata")
    p.add_argument("--historical-events", default=None, help="Historical StatsBomb/player event CSV already processed")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--n-simulations", type=int, default=1000, help="Monte Carlo tournament simulations. Use 50000 for serious tournament probability reports; use smaller values for smoke tests.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--detail-sample-simulations", type=int, default=50, help="Number of simulations to retain in tournament_details.csv for audit/sample rows.")
    p.add_argument("--clean-out-dir", action="store_true", help="Delete out-dir before writing fresh outputs")
    p.add_argument("--calibration-model", default=None, help="Optional JSON calibration model from evaluate_statistical_core.py")
    p.add_argument("--model-config", default=None, help="Optional JSON model config or best_model_config.json from run_model_lab.py")
    p.add_argument("--event-model-config", default=None, help="Optional JSON event model config from run_event_model_lab.py")
    p.add_argument("--no-demo-picks", action="store_true", help="Keep demo odds in edges but block recommended picks")
    p.add_argument("--dynamic-lines", action="store_true", default=True, help="Generate dynamic line board with structured evidence")
    p.add_argument("--no-dynamic-lines", dest="dynamic_lines", action="store_false", help="Disable dynamic line board generation")
    p.add_argument("--recent-n", type=int, default=10, help="Recent sample size for structured evidence")
    p.add_argument("--h2h-years", type=int, default=5, help="Recency cutoff for H2H evidence")
    p.add_argument("--similar-elo-years", type=int, default=4, help="Recency cutoff for similar-Elo evidence")
    p.add_argument("--similar-elo-range", type=float, default=100.0, help="Opponent rating distance for similar-Elo evidence")
    p.add_argument("--h2h-max-matches", type=int, default=8, help="Maximum recent H2H matches used for evidence")
    p.add_argument("--similar-elo-max-matches", type=int, default=12, help="Maximum recent similar-Elo matches used for evidence")
    p.add_argument("--min-context-sample", type=int, default=3, help="Minimum sample before context evidence is considered usable")
    p.add_argument("--min-strong-context-sample", type=int, default=5, help="Minimum sample before evidence can be tagged as strong rather than thin")
    p.add_argument("--max-player-rows-per-market", type=int, default=60, help="Limit player rows per market in dynamic line board")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.clean_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit: dict = {
        "version": "v0.48.4_simulation_evaluation_foundation",
        "paper_mode": True,
        "status": "started",
        "warnings": [],
        "focus": "statistical_simulator_first",
        "experimental": [
            "Dixon-Coles/Bivariate Poisson not implemented yet; using auditable independent Poisson profile model.",
            "Tournament knockout bracket is approximate unless explicit knockout fixtures are provided.",
            "Top scorer and awards probabilities are approximate from player shots/progression, not dedicated xG/award models yet.",
            "Odds comparison remains optional; statistical simulation must run without odds.",
            "Advanced match report surfaces existing simulator outputs; it does not retrain models or create betting recommendations.",
            "Matchday summary rankings are statistical ordering views, not betting picks.",
            "Tournament visual report is a summary layer over Monte Carlo outputs; it does not change the simulator model.",
        ],
        "simulation_policy": {
            "recommended_large_run_n_simulations": 50000,
            "current_run_n_simulations": int(args.n_simulations),
            "seed": int(args.seed),
            "detail_sample_simulations": int(args.detail_sample_simulations),
            "large_run_note": "Use 50,000+ simulations for serious tournament reports; smoke tests intentionally use smaller values.",
        },
        "generated_files": [],
    }

    fixtures_raw = pd.read_csv(args.fixtures)
    fixtures = standardize_fixtures(fixtures_raw)
    lineups = read_csv_optional(args.lineups)
    squads = read_csv_optional(args.squads)
    odds = read_csv_optional(args.odds)
    historical_events = read_csv_optional(args.historical_events)
    tournament_config = read_csv_optional(args.tournament_config)

    audit["inputs"] = {
        "fixtures_rows": int(len(fixtures)),
        "lineups_rows": int(len(lineups)),
        "squads_rows": int(len(squads)),
        "odds_rows": int(len(odds)),
        "historical_events_rows": int(len(historical_events)),
        "tournament_config_rows": int(len(tournament_config)),
    }

    model_config = {}
    if args.model_config:
        model_config = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
        if isinstance(model_config, dict) and "model_config" in model_config:
            model_config = model_config.get("model_config") or {}
    event_model_config = {}
    if args.event_model_config:
        event_model_config = json.loads(Path(args.event_model_config).read_text(encoding="utf-8"))
        if isinstance(event_model_config, dict) and "event_model_config" in event_model_config:
            event_model_config = event_model_config.get("event_model_config") or {}
    match_model = MatchOutcomeModel(**model_config).fit(historical_events)
    if model_config:
        audit["model_config_applied"] = model_config
    match_predictions, scoreline_distribution = match_model.predict_fixtures(fixtures)
    calibration_model = load_calibration(args.calibration_model)
    if calibration_model:
        match_predictions = apply_match_calibration(match_predictions, calibration_model)
        audit["calibration_applied"] = {
            "source": str(args.calibration_model),
            "status": calibration_model.get("status", "loaded"),
            "method": calibration_model.get("method", "unknown"),
        }
    audit["match_model"] = match_model.audit

    team_stats_config = event_model_config.get("team_model_config", {}) if isinstance(event_model_config, dict) else {}
    player_event_config = event_model_config.get("player_model_config", {}) if isinstance(event_model_config, dict) else {}
    if event_model_config:
        audit["event_model_config_applied"] = event_model_config
    team_model = TeamStatsModel(**team_stats_config).fit(historical_events)
    team_stats_predictions = team_model.predict_fixtures(fixtures, match_predictions)
    audit["team_stats_model"] = team_model.audit
    if not team_model.availability.get("corners", False):
        audit["warnings"].append("corners_not_available_not_invented")
    save_cols = {"saves", "goalkeeper_saves", "shots_saved"}
    if historical_events.empty or not any(c in historical_events.columns for c in save_cols):
        audit["warnings"].append("goalkeeper_saves_not_available_not_invented")

    player_model = PlayerEventModel(**player_event_config).fit(historical_events)
    player_event_predictions, player_warnings = player_model.predict(fixtures, lineups, squads, team_stats_predictions)
    audit["player_event_model"] = player_model.audit
    audit["warnings"].extend(player_warnings)

    betting_engine = BettingValueEngine()
    betting_edges = betting_engine.evaluate(odds, match_predictions, team_stats_predictions, player_event_predictions)
    demo_odds_detected = not odds.empty and "bookmaker" in odds.columns and odds["bookmaker"].astype(str).str.lower().eq("demo_book").any()
    if demo_odds_detected and args.no_demo_picks and not betting_edges.empty:
        betting_edges = betting_edges.copy()
        was_recommended = betting_edges["recommended"] == True
        betting_edges.loc[was_recommended, "recommended"] = False
        betting_edges.loc[was_recommended, "stake_virtual"] = 0.0
        betting_edges.loc[was_recommended, "warnings"] = betting_edges.loc[was_recommended, "warnings"].astype(str).replace({"nan": ""}).str.strip(";") + ";demo_odds_pick_blocked"
        betting_edges.loc[was_recommended, "reason"] = "not_recommended: demo odds detected; real value picks are blocked by --no-demo-picks. " + betting_edges.loc[was_recommended, "reason"].astype(str)
        audit["demo_picks_blocked"] = int(was_recommended.sum())
    recommended_picks = betting_edges[betting_edges["recommended"] == True].copy() if not betting_edges.empty else betting_edges.copy()
    if demo_odds_detected:
        audit["warnings"].append("demo_odds_detected_do_not_use_for_real_value")

    simulate = not tournament_config.empty or fixtures["stage"].astype(str).str.lower().str.contains("group|quarter|semi|final", na=False).any()
    if simulate:
        # If supplied, tournament_config can enrich fixtures by match_id/group/stage.
        sim_fixtures = fixtures.copy()
        if not tournament_config.empty and "match_id" in tournament_config.columns:
            enrich_cols = [c for c in ["match_id", "group", "stage", "round"] if c in tournament_config.columns]
            sim_fixtures = sim_fixtures.drop(columns=[c for c in ["group", "stage", "round"] if c in sim_fixtures.columns and c != "match_id"], errors="ignore")
            sim_fixtures = sim_fixtures.merge(tournament_config[enrich_cols], on="match_id", how="left")
        simulator = TournamentSimulator(TournamentSimulationConfig(n_simulations=args.n_simulations, seed=args.seed, detail_sample_simulations=args.detail_sample_simulations))
        tournament_simulation, tournament_details = simulator.simulate(sim_fixtures, match_predictions, player_event_predictions)
        audit["tournament_simulator"] = {
            "status": "completed",
            "n_simulations": int(args.n_simulations),
            "seed": int(args.seed),
            "detail_sample_simulations": int(args.detail_sample_simulations),
            "summary_rows": int(len(tournament_simulation)),
            "detail_rows": int(len(tournament_details)),
            "large_run_ready": int(args.n_simulations) >= 50000,
        }
    else:
        tournament_simulation = pd.DataFrame()
        tournament_details = pd.DataFrame()
        audit["tournament_simulator"] = {
            "status": "skipped_no_tournament_context",
            "n_simulations": int(args.n_simulations),
            "seed": int(args.seed),
            "summary_rows": 0,
            "detail_rows": 0,
            "large_run_ready": False,
        }

    competition_engine = CompetitionForecastEngine()
    top_scorer_predictions, award_predictions, competition_summary = competition_engine.build_outputs(
        player_event_predictions,
        tournament_simulation,
        match_predictions,
    )
    audit["competition_forecast_model"] = competition_engine.audit

    dynamic_line_config = DynamicLineConfig(
        recent_n=args.recent_n,
        h2h_years=args.h2h_years,
        h2h_max_matches=args.h2h_max_matches,
        similar_elo_years=args.similar_elo_years,
        similar_elo_range=args.similar_elo_range,
        similar_elo_max_matches=args.similar_elo_max_matches,
        min_context_sample=args.min_context_sample,
        min_strong_context_sample=args.min_strong_context_sample,
        max_player_rows_per_market=args.max_player_rows_per_market,
    )
    if args.dynamic_lines:
        dynamic_market_lines = build_dynamic_market_lines(
            fixtures,
            match_predictions,
            scoreline_distribution,
            team_stats_predictions,
            player_event_predictions,
            historical_events,
            odds,
            dynamic_line_config,
        )
        audit["dynamic_lines"] = {
            "status": "completed",
            "rows": int(len(dynamic_market_lines)),
            "config": dynamic_line_config.__dict__,
            "policy": "dynamic lines by match/team/player scope; line-specific player evidence; signal_label separated from value_label; demo odds are labelled and cannot become real value",
            "available_rows": int(dynamic_market_lines["availability"].astype(str).eq("available").sum()) if not dynamic_market_lines.empty and "availability" in dynamic_market_lines.columns else 0,
            "priced_rows": int(dynamic_market_lines["book_odds"].notna().sum()) if not dynamic_market_lines.empty and "book_odds" in dynamic_market_lines.columns else 0,
        }
    else:
        dynamic_market_lines = pd.DataFrame()
        audit["dynamic_lines"] = {"status": "disabled", "rows": 0, "config": dynamic_line_config.__dict__}

    matchday_summary, matchday_summary_payload = build_matchday_summary(
        match_predictions=match_predictions,
        scoreline_distribution=scoreline_distribution,
        dynamic_market_lines=dynamic_market_lines,
        team_stats_predictions=team_stats_predictions,
        player_event_predictions=player_event_predictions,
        audit=audit,
    )
    audit["matchday_summary"] = {
        "status": "completed",
        "rows": int(len(matchday_summary)),
        "categories": sorted(matchday_summary["ranking_category"].dropna().astype(str).unique().tolist()) if not matchday_summary.empty and "ranking_category" in matchday_summary.columns else [],
        "policy": "statistical matchday rankings only; no betting picks, stake sizing, ROI or live automation",
    }

    tournament_report, tournament_report_payload = build_tournament_report(
        tournament_simulation=tournament_simulation,
        tournament_details=tournament_details,
        match_predictions=match_predictions,
        fixtures=sim_fixtures if simulate else fixtures,
        competition_summary=competition_summary,
        top_scorer_predictions=top_scorer_predictions,
        audit=audit,
    )
    audit["tournament_report"] = {
        "status": "completed",
        "rows": int(len(tournament_report)),
        "categories": sorted(tournament_report["report_section"].dropna().astype(str).unique().tolist()) if not tournament_report.empty and "report_section" in tournament_report.columns else [],
        "policy": "visual tournament summary over existing Monte Carlo outputs; no betting recommendations or model changes",
    }

    # Core audits.
    if not match_predictions.empty:
        prob_sum = match_predictions["p_home_win"] + match_predictions["p_draw"] + match_predictions["p_away_win"]
        audit["max_outcome_probability_sum_error"] = float((prob_sum - 1.0).abs().max())
        if audit["max_outcome_probability_sum_error"] > 1e-6:
            audit["warnings"].append("outcome_probabilities_do_not_sum_to_one")
    if not player_event_predictions.empty:
        candidate_sources = sorted(player_event_predictions["candidate_source"].dropna().astype(str).unique().tolist())
        audit["player_candidate_sources"] = candidate_sources
        audit["retired_players_in_inference"] = False
        audit["player_identity_audit"] = {
            "rows": int(len(player_event_predictions)),
            "zero_sample_rows": int((pd.to_numeric(player_event_predictions.get("sample_size_minutes", 0), errors="coerce").fillna(0) <= 0).sum()),
            "unresolved_rows": int(player_event_predictions.get("identity_status", pd.Series(dtype=str)).astype(str).ne("matched").sum()) if "identity_status" in player_event_predictions.columns else 0,
            "match_levels": {str(k): int(v) for k, v in player_event_predictions.get("identity_match_level", pd.Series(dtype=str)).astype(str).value_counts().to_dict().items()} if "identity_match_level" in player_event_predictions.columns else {},
        }
        if audit["player_identity_audit"]["zero_sample_rows"]:
            audit["warnings"].append("player_identity_zero_sample_rows_present")
        if audit["player_identity_audit"]["unresolved_rows"]:
            audit["warnings"].append("player_identity_unresolved_rows_present")
    if not betting_edges.empty:
        bad_rec_mask = (betting_edges["recommended"] == True) & betting_edges["warnings"].astype(str).str.contains("sample_size_zero_no_player_pick|identity_unresolved|identity_ambiguous", regex=True, na=False)
        audit["blocked_bad_player_pick_recommendations"] = int(bad_rec_mask.sum())
        if int(bad_rec_mask.sum()) > 0:
            audit["warnings"].append("bad_player_pick_recommendation_not_blocked")
    if not team_stats_predictions.empty:
        absurd = []
        caps = {"shots": 35, "shots_on_target": 18, "fouls": 40, "yellow_cards": 9, "total_shots": 70, "total_shots_on_target": 30, "total_fouls": 80, "total_yellow_cards": 16}
        for market, cap in caps.items():
            frame = team_stats_predictions[team_stats_predictions["market"].astype(str).eq(market)]
            if not frame.empty:
                n_bad = int((pd.to_numeric(frame["expected_count"], errors="coerce") > cap).sum())
                if n_bad:
                    absurd.append(f"{market}>{cap}:rows={n_bad}")
        audit["team_stats_absurd_count_checks"] = absurd
        if absurd:
            audit["warnings"].append("team_stats_absurd_count_check_failed")
    audit["leakage_policy"] = {
        "future_real_minutes_used": False,
        "candidate_gate": "current_lineups_first_squads_second",
        "manual_expected_minutes_required_or_defaulted": True,
    }

    outputs = {
        "match_predictions.csv": match_predictions,
        "scoreline_distribution.csv": scoreline_distribution,
        "team_stats_predictions.csv": team_stats_predictions,
        "player_event_predictions.csv": player_event_predictions,
        "betting_edges.csv": betting_edges,
        "recommended_picks.csv": recommended_picks,
        "tournament_simulation.csv": tournament_simulation,
        "tournament_details.csv": tournament_details,
        "top_scorer_predictions.csv": top_scorer_predictions,
        "award_predictions.csv": award_predictions,
        "competition_summary.csv": competition_summary,
        "dynamic_market_lines.csv": dynamic_market_lines,
        "matchday_summary.csv": matchday_summary,
        "tournament_report.csv": tournament_report,
    }
    for name, frame in outputs.items():
        path = out_dir / name
        frame.to_csv(path, index=False)
        audit["generated_files"].append(str(path))

    matchday_summary_json_path = out_dir / "matchday_summary.json"
    write_json(matchday_summary_json_path, matchday_summary_payload)
    audit["generated_files"].append(str(matchday_summary_json_path))

    tournament_report_json_path = out_dir / "tournament_report.json"
    write_json(tournament_report_json_path, tournament_report_payload)
    audit["generated_files"].append(str(tournament_report_json_path))

    audit["status"] = "completed"
    audit_path = out_dir / "audit_report.json"
    write_json(audit_path, audit)
    audit["generated_files"].append(str(audit_path))
    report_path = build_daily_html_report(
        out_dir / "daily_report.html",
        match_predictions,
        team_stats_predictions,
        player_event_predictions,
        betting_edges,
        tournament_simulation,
        top_scorer_predictions=top_scorer_predictions,
        award_predictions=award_predictions,
        competition_summary=competition_summary,
        dynamic_market_lines=dynamic_market_lines,
        audit=audit,
        scoreline_distribution=scoreline_distribution,
        matchday_summary=matchday_summary,
        tournament_report=tournament_report,
    )
    audit["generated_files"].append(str(report_path))

    contract_report = build_simulator_contract_report(
        out_dir=out_dir,
        frames=outputs,
        audit=audit,
    )
    contract_path = out_dir / "simulation_contract_report.json"
    write_json(contract_path, contract_report)
    audit["simulation_contract"] = {
        "version": contract_report["contract_version"],
        "status": contract_report["status"],
        "missing_files": contract_report["missing_files"],
        "schema_failures": contract_report["schema_failures"],
    }
    audit["generated_files"].append(str(contract_path))
    write_json(audit_path, audit)

    print(f"Statistical matchday complete: {out_dir}")
    print(f"Recommended picks: {len(recommended_picks)}")
    print(f"Audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
