from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


CONTRACT_VERSION = "v0.48.4_simulation_evaluation_foundation_contract"


REQUIRED_HTML_REPORT_SECTIONS: tuple[str, ...] = (
    "Mundialytics Statistical Simulator v0.48.4",
    "Executive Summary",
    "Matchday Summary Rankings",
    "Match Probabilities",
    "Advanced Match Cards",
    "Top Scorelines",
    "Dynamic Goal Lines",
    "Not Available Markets",
    "Team Statistics",
    "Player Statistics",
    "Data Quality",
    "Tournament Visual Report",
    "Simulation Metadata",
)


@dataclass(frozen=True)
class OutputContract:
    file_name: str
    required_columns: tuple[str, ...]
    description: str


SIMULATOR_OUTPUT_CONTRACTS: tuple[OutputContract, ...] = (
    OutputContract(
        "match_predictions.csv",
        (
            "match_id",
            "home_team",
            "away_team",
            "lambda_home",
            "lambda_away",
            "p_home_win",
            "p_draw",
            "p_away_win",
            "p_over_25",
            "p_btts",
            "most_likely_score",
        ),
        "Per-fixture probabilistic match prediction contract.",
    ),
    OutputContract(
        "scoreline_distribution.csv",
        ("match_id", "home_goals", "away_goals", "probability"),
        "Scoreline probability mass by fixture.",
    ),
    OutputContract(
        "team_stats_predictions.csv",
        ("match_id", "team", "market", "expected_count", "availability", "confidence", "warnings"),
        "Team-level expected event counts, with explicit availability and quality flags.",
    ),
    OutputContract(
        "player_event_predictions.csv",
        ("match_id", "team", "player", "market", "expected_count", "safe_probability", "confidence_flag", "candidate_source"),
        "Player-level event projections gated by current lineups/squads.",
    ),
    OutputContract(
        "dynamic_market_lines.csv",
        ("match_id", "market", "scope", "line", "over_under", "model_probability", "fair_odds", "availability", "data_quality_flag", "value_label", "evidence_tags"),
        "Dynamic line board for statistical market-style outputs.",
    ),
    OutputContract(
        "tournament_simulation.csv",
        ("team", "qualify_group_probability", "champion_probability", "expected_points", "expected_goals_for", "simulations", "seed"),
        "Monte Carlo tournament probability summary.",
    ),
    OutputContract(
        "tournament_details.csv",
        ("simulation", "match_id", "home_goals", "away_goals"),
        "Sampled simulation details plus optional experimental player award rows.",
    ),
    OutputContract(
        "competition_summary.csv",
        ("record_type", "headline"),
        "Compact competition-level summary.",
    ),
    OutputContract(
        "matchday_summary.csv",
        (
            "ranking_category",
            "rank",
            "match_id",
            "match",
            "metric_name",
            "metric_value",
            "statistical_label",
            "data_quality_flag",
            "short_structured_reason",
        ),
        "Simulator-first matchday rankings and daily statistical summary.",
    ),
    OutputContract(
        "matchday_summary.json",
        ("version", "categories", "category_counts"),
        "Machine-readable grouped matchday summary rankings.",
    ),
    OutputContract(
        "tournament_report.csv",
        (
            "report_section",
            "rank",
            "team",
            "metric_name",
            "metric_value",
            "statistical_label",
            "data_quality_flag",
            "short_structured_reason",
        ),
        "Visual tournament summary rows built from existing Monte Carlo outputs.",
    ),
    OutputContract(
        "tournament_report.json",
        ("version", "categories", "category_counts"),
        "Machine-readable tournament visual report summary.",
    ),
    OutputContract(
        "audit_report.json",
        ("status", "version", "paper_mode", "warnings"),
        "Machine-readable run audit.",
    ),
    OutputContract(
        "daily_report.html",
        (),
        "Human-readable statistical matchday report.",
    ),
)


def expected_output_files() -> list[str]:
    return [contract.file_name for contract in SIMULATOR_OUTPUT_CONTRACTS]


def validate_frame_columns(frame: pd.DataFrame | None, required_columns: tuple[str, ...]) -> dict[str, Any]:
    if frame is None:
        return {"status": "missing_frame", "missing_columns": list(required_columns), "row_count": 0}
    missing = [column for column in required_columns if column not in frame.columns]
    return {
        "status": "ok" if not missing else "missing_columns",
        "missing_columns": missing,
        "row_count": int(len(frame)),
    }


def validate_html_report_sections(report_path: str | Path, required_sections: tuple[str, ...] = REQUIRED_HTML_REPORT_SECTIONS) -> dict[str, Any]:
    path = Path(report_path)
    if not path.exists():
        return {"status": "missing_file", "missing_sections": list(required_sections), "checked_sections": list(required_sections)}
    text = path.read_text(encoding="utf-8", errors="replace")
    missing = [section for section in required_sections if section not in text]
    return {
        "status": "ok" if not missing else "missing_sections",
        "missing_sections": missing,
        "checked_sections": list(required_sections),
    }


def build_simulator_contract_report(
    *,
    out_dir: str | Path,
    frames: dict[str, pd.DataFrame],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact contract report for the v0.48 statistical simulator.

    This report is intentionally descriptive rather than punitive: it records
    expected files, schemas and known limitations so the simulator can evolve
    without silently changing its public outputs.
    """
    out = Path(out_dir)
    files_report: dict[str, Any] = {}
    missing_files: list[str] = []
    schema_failures: list[str] = []

    for contract in SIMULATOR_OUTPUT_CONTRACTS:
        file_exists = (out / contract.file_name).exists()
        if not file_exists:
            missing_files.append(contract.file_name)
        frame_report = {"status": "not_applicable", "missing_columns": [], "row_count": None}
        if contract.file_name.endswith(".csv"):
            frame_report = validate_frame_columns(frames.get(contract.file_name), contract.required_columns)
            if frame_report["status"] != "ok":
                schema_failures.append(contract.file_name)
        files_report[contract.file_name] = {
            "exists": file_exists,
            "description": contract.description,
            "required_columns": list(contract.required_columns),
            "schema": frame_report,
        }

    html_report_sections = validate_html_report_sections(out / "daily_report.html")
    report_section_failures = list(html_report_sections.get("missing_sections", []))

    n_simulations = None
    if "tournament_simulation.csv" in frames and not frames["tournament_simulation.csv"].empty and "simulations" in frames["tournament_simulation.csv"].columns:
        n_simulations = int(pd.to_numeric(frames["tournament_simulation.csv"]["simulations"], errors="coerce").dropna().max())

    report = {
        "contract_version": CONTRACT_VERSION,
        "status": "ok" if not missing_files and not schema_failures and not report_section_failures else "needs_attention",
        "run_version": audit.get("version"),
        "paper_mode": bool(audit.get("paper_mode", True)),
        "n_simulations": n_simulations,
        "recommended_large_run_n_simulations": 50000,
        "large_run_policy": "Use 50,000+ simulations for serious tournament probability reports; use small runs only for smoke tests.",
        "core_principles": {
            "odds_required": False,
            "live_betting": False,
            "current_player_gate_required": True,
            "missing_unreliable_markets_must_be_not_available": True,
            "historical_player_only_inference_forbidden": True,
        },
        "files": files_report,
        "missing_files": missing_files,
        "schema_failures": schema_failures,
        "html_report_sections": html_report_sections,
        "report_section_failures": report_section_failures,
        "warnings": list(audit.get("warnings", [])),
    }
    return report


def output_contracts_as_dicts() -> list[dict[str, Any]]:
    return [asdict(contract) for contract in SIMULATOR_OUTPUT_CONTRACTS]
