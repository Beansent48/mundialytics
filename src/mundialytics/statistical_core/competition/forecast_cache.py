"""
Forecast caching — precompute once, serve instantly, scrub the whole season.

A daily-traffic site must NEVER retrain the engine on a page load (~10-12s per
cutoff). This module computes a self-contained JSON "bundle" per (competition,
season) holding a full-season SNAPSHOT SET: the forecast at each matchday on a
grid. One bundle lets the UI's matchday slider scrub the entire season instantly
(table, probabilities, position matrix all update from cache), and the probability
evolution timeline is derived from the same snapshots.

Each snapshot holds everything the web renders at that point:
  - standings   : the live table
  - fixtures    : played (with results) + remaining (with predicted 1X2 + lambdas)
  - forecast    : team probabilities + the full position matrix

Incremental by design: a snapshot for a completed matchday is IMMUTABLE — it only
ever depended on data up to then, which never changes again. A refresh recomputes
only newly-available matchdays.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mundialytics.statistical_core.competition.cutoff import load_league_state_from_foundation
from mundialytics.statistical_core.competition.engine_provider import (
    fixture_lambdas,
    train_engine_before_cutoff,
)
from mundialytics.statistical_core.competition.resume_simulator import simulate_rest_of_season
from mundialytics.statistical_core.distributions import outcome_probabilities

DEFAULT_CACHE_DIR = Path("data/processed/competition_cache")
DIXON_COLES_RHO = -0.07
BUNDLE_SCHEMA = 2


# ── Paths / fingerprint ─────────────────────────────────────────────────────────

def _slug(competition: str, season: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{competition}__{season}")


def bundle_path(competition: str, season: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{_slug(competition, season)}.json"


def _fingerprint(state) -> dict:
    last_date = pd.to_datetime(state.played["date"]).max() if state.n_played else None
    return {
        "n_played": int(state.n_played),
        "last_played_date": str(last_date.date()) if last_date is not None else None,
    }


def _total_rounds(state) -> int:
    n_teams = len(state.teams)
    return 2 * (n_teams - 1)


def _matchday_grid(total_rounds: int, step: int, max_matchday: int | None) -> list[int]:
    top = max_matchday if max_matchday is not None else total_rounds - 1
    top = min(top, total_rounds - 1)
    grid = list(range(step, top + 1, step))
    if top not in grid:
        grid.append(top)
    return grid


def _full_grid(probe_state, total_rounds: int, step: int, max_matchday: int | None) -> list[int]:
    """Forecast grid plus the true final-table point, when it actually exists.

    The final point (matchday == total_rounds, 0 fixtures remaining) is only
    added when the season is ALREADY complete in the data and the caller isn't
    deliberately capping below it — this keeps an in-progress live season (once
    a live loader replaces the CSV+cutoff one) from getting a fabricated "final"
    snapshot for a season that hasn't finished yet.
    """
    grid = _matchday_grid(total_rounds, step, max_matchday)
    within_cap = max_matchday is None or max_matchday >= total_rounds
    if probe_state.is_complete and within_cap and total_rounds not in grid:
        grid.append(total_rounds)
    return grid


# ── Snapshot serialisation ──────────────────────────────────────────────────────

def _remaining_predictions(engine, state) -> list[dict]:
    lam = fixture_lambdas(engine, state)
    out = []
    for r in lam.itertuples(index=False):
        probs = outcome_probabilities(r.lambda_home, r.lambda_away, dixon_coles_rho=DIXON_COLES_RHO)
        out.append({
            "date": str(pd.to_datetime(r.date).date()) if pd.notna(r.date) else None,
            "home_team": r.home_team, "away_team": r.away_team,
            "lambda_home": round(float(r.lambda_home), 3), "lambda_away": round(float(r.lambda_away), 3),
            "p_home": round(float(probs["p_home_win"]), 4),
            "p_draw": round(float(probs["p_draw"]), 4),
            "p_away": round(float(probs["p_away_win"]), 4),
        })
    return out


def _played_records(state) -> list[dict]:
    cols = [c for c in ("date", "home_team", "away_team", "home_goals", "away_goals") if c in state.played.columns]
    p = state.played[cols].copy()
    if "date" in p.columns:
        p["date"] = pd.to_datetime(p["date"]).dt.date.astype(str)
    return p.to_dict("records")


def _forecast_dict(forecast) -> dict:
    pm = forecast.position_matrix
    return {
        "team_probs": forecast.team_probs.round(4).to_dict("records"),
        "position_matrix": {
            "teams": list(pm.index),
            "positions": [int(c) for c in pm.columns],
            "values": pm.round(4).values.tolist(),
        },
    }


def _build_snapshot(competition, season, foundation, matchday, n_sims) -> dict:
    state = load_league_state_from_foundation(
        competition, season, cutoff_matchday=matchday, foundation=foundation
    )
    # A "matchday == total_rounds" snapshot represents the actual completed
    # season (0 remaining fixtures) — nothing to forecast, so skip training the
    # engine entirely rather than pay ~10s to fit a model that predicts nothing.
    engine = None if state.is_complete else train_engine_before_cutoff(state, foundation)
    lam = fixture_lambdas(engine, state)
    fc = simulate_rest_of_season(lam, state, n_sims=n_sims)
    return {
        "matchday": matchday,
        "fingerprint": _fingerprint(state),
        "n_remaining": state.n_remaining,
        "standings": state.standings.to_dict("records"),
        "fixtures": {
            "played": _played_records(state),
            "remaining": _remaining_predictions(engine, state) if engine is not None else [],
        },
        "forecast": _forecast_dict(fc),
    }


# ── Build / refresh ─────────────────────────────────────────────────────────────

def build_bundle(
    competition: str,
    season: str,
    foundation: pd.DataFrame,
    max_matchday: int | None = None,
    timeline_step: int = 5,
    n_sims: int = 10_000,
    existing: dict | None = None,
) -> dict:
    """Compute (or incrementally extend) the full-season snapshot bundle.

    Reuses immutable snapshots present in ``existing``; only missing matchdays are
    computed. ``max_matchday`` caps the grid (default: total_rounds - 1, so the
    last snapshot still has a game left to forecast).
    """
    probe = load_league_state_from_foundation(competition, season, foundation=foundation)
    total_rounds = _total_rounds(probe)
    forecast_grid = _matchday_grid(total_rounds, timeline_step, max_matchday)
    grid = _full_grid(probe, total_rounds, timeline_step, max_matchday)

    cached = {int(k): v for k, v in (existing or {}).get("snapshots", {}).items()} if existing else {}
    reuse_ok = (existing or {}).get("meta", {}).get("schema") == BUNDLE_SCHEMA

    snapshots: dict[str, dict] = {}
    for md in grid:
        if reuse_ok and md in cached:
            snapshots[str(md)] = cached[md]
        else:
            snapshots[str(md)] = _build_snapshot(competition, season, foundation, md, n_sims)

    # "current" stays the last in-progress forecast point (not the final-table
    # point) so the default view still shows a live title race, not a settled one.
    current_md = forecast_grid[-1]
    return {
        "meta": {
            "schema": BUNDLE_SCHEMA,
            "competition": competition,
            "season": season,
            "matchdays": grid,
            "current_matchday": current_md,
            "total_rounds": total_rounds,
            "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fingerprint": snapshots[str(current_md)]["fingerprint"],
            "n_sims": n_sims,
            "timeline_step": timeline_step,
            "model_note": "leakage-free (trained on data < cutoff); no xG/ELO yet; uncalibrated",
        },
        "snapshots": snapshots,
    }


def save_bundle(bundle: dict, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = bundle_path(bundle["meta"]["competition"], bundle["meta"]["season"], cache_dir)
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_bundle(competition: str, season: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict | None:
    path = bundle_path(competition, season, cache_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_or_build(
    competition: str,
    season: str,
    foundation: pd.DataFrame,
    max_matchday: int | None = None,
    timeline_step: int = 5,
    n_sims: int = 10_000,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> dict:
    """Return the cached bundle if it already covers the requested grid, else build."""
    existing = load_bundle(competition, season, cache_dir)
    if existing and not force and existing.get("meta", {}).get("schema") == BUNDLE_SCHEMA:
        probe = load_league_state_from_foundation(competition, season, foundation=foundation)
        want = _full_grid(probe, _total_rounds(probe), timeline_step, max_matchday)
        have = set(existing.get("meta", {}).get("matchdays", []))
        if set(want).issubset(have):
            return existing
    bundle = build_bundle(competition, season, foundation, max_matchday=max_matchday,
                          timeline_step=timeline_step, n_sims=n_sims, existing=existing)
    save_bundle(bundle, cache_dir)
    return bundle


# ── Read helpers (used by the web page) ─────────────────────────────────────────

def available_matchdays(bundle: dict) -> list[int]:
    return sorted(int(k) for k in bundle.get("snapshots", {}))


def snapshot_for(bundle: dict, matchday: int) -> tuple[int, dict]:
    """Nearest available snapshot to ``matchday`` — the slider snaps to this."""
    mds = available_matchdays(bundle)
    nearest = min(mds, key=lambda m: abs(m - matchday))
    return nearest, bundle["snapshots"][str(nearest)]


def build_timeline(bundle: dict) -> list[dict]:
    """Derive the matchday-by-matchday probability evolution from the snapshots."""
    rows = []
    for md in available_matchdays(bundle):
        snap = bundle["snapshots"][str(md)]
        for rec in snap["forecast"]["team_probs"]:
            rows.append({
                "matchday": md, "team": rec["team"],
                "p_champion": rec["p_champion"], "p_top4": rec.get("p_top4"),
                "p_relegation": rec["p_relegation"], "exp_points": rec["exp_points"],
            })
    return rows
