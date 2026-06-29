from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mundialytics.statistical_core.distributions import scoreline_distribution, softmax
from mundialytics.statistical_core.schemas import canonical_name, standardize_fixtures


@dataclass
class TournamentSimulationConfig:
    n_simulations: int = 1000
    seed: int = 42
    group_qualifiers: int = 2
    detail_sample_simulations: int = 50


class TournamentSimulator:
    """Monte Carlo tournament simulator using match score distributions."""

    def __init__(self, config: TournamentSimulationConfig | None = None):
        self.config = config or TournamentSimulationConfig()

    def simulate(
        self,
        fixtures: pd.DataFrame,
        match_predictions: pd.DataFrame,
        player_event_predictions: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        fixtures_std = standardize_fixtures(fixtures)
        if match_predictions is None or match_predictions.empty:
            return pd.DataFrame(), pd.DataFrame()
        rng = np.random.default_rng(self.config.seed)
        teams = sorted(set(fixtures_std["home_team"]).union(set(fixtures_std["away_team"])))
        counters = {team: {"group_winner": 0, "qualify_group": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0, "points": 0.0, "goals_for": 0.0} for team in teams}
        match_dist_rows: list[dict[str, Any]] = []
        pred_lookup = {str(r["match_id"]): r for _, r in match_predictions.iterrows()}
        score_samplers = {
            match_id: _build_score_sampler(float(pred["lambda_home"]), float(pred["lambda_away"]))
            for match_id, pred in pred_lookup.items()
        }
        group_fixtures = fixtures_std[fixtures_std["stage"].astype(str).str.lower().str.contains("group", na=False)].copy()
        if group_fixtures.empty:
            group_fixtures = fixtures_std.copy()
        sampled_group_scores = _pre_sample_group_scores(
            group_fixtures,
            score_samplers,
            rng,
            self.config.n_simulations,
        )
        group_fixture_records = [
            {
                "match_id": str(row["match_id"]),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
            }
            for _, row in group_fixtures.iterrows()
        ]
        group_definitions = _build_group_definitions(group_fixtures, teams)
        rating_lookup = _build_rating_lookup(match_predictions)
        for sim in range(self.config.n_simulations):
            table = {team: {"points": 0, "gf": 0, "ga": 0, "gd": 0} for team in teams}
            for fixture_record in group_fixture_records:
                sampled = sampled_group_scores.get(fixture_record["match_id"])
                if sampled is None:
                    continue
                h = int(sampled["home_goals"][sim])
                a = int(sampled["away_goals"][sim])
                _apply_result(table, fixture_record["home_team"], fixture_record["away_team"], h, a)
                if sim < self.config.detail_sample_simulations:
                    match_dist_rows.append({"simulation": sim, "match_id": fixture_record["match_id"], "home_goals": h, "away_goals": a})
            qualified: list[str] = []
            if group_definitions:
                for group_teams in group_definitions:
                    ordered = _rank_teams(group_teams, table)
                    if ordered:
                        counters[ordered[0]]["group_winner"] += 1
                    for t in ordered[: self.config.group_qualifiers]:
                        counters[t]["qualify_group"] += 1
                        qualified.append(t)
            else:
                ordered = _rank_teams(teams, table)
                for t in ordered[: max(2, len(ordered) // 2)]:
                    counters[t]["qualify_group"] += 1
                    qualified.append(t)
                if ordered:
                    counters[ordered[0]]["group_winner"] += 1
            for team in teams:
                counters[team]["points"] += table[team]["points"]
                counters[team]["goals_for"] += table[team]["gf"]
            champion, stage_counts = _simulate_knockout_or_champion(qualified, rating_lookup, rng)
            for stage, stage_teams in stage_counts.items():
                for t in stage_teams:
                    if t in counters:
                        counters[t][stage] += 1
            if champion in counters:
                counters[champion]["champion"] += 1
        rows = []
        n = float(self.config.n_simulations)
        for team in teams:
            c = counters[team]
            rows.append(
                {
                    "team": team,
                    "group_winner_probability": c["group_winner"] / n,
                    "qualify_group_probability": c["qualify_group"] / n,
                    "r16_probability": c["r16"] / n,
                    "qf_probability": c["qf"] / n,
                    "sf_probability": c["sf"] / n,
                    "final_probability": c["final"] / n,
                    "champion_probability": c["champion"] / n,
                    "expected_points": c["points"] / n,
                    "expected_goals_for": c["goals_for"] / n,
                    "simulations": int(n),
                    "seed": self.config.seed,
                }
            )
        result = pd.DataFrame(rows).sort_values(["champion_probability", "qualify_group_probability"], ascending=False).reset_index(drop=True)
        top_scorers = _top_scorer_probabilities(player_event_predictions)
        return result, pd.concat([pd.DataFrame(match_dist_rows), top_scorers], ignore_index=True, sort=False) if match_dist_rows or not top_scorers.empty else pd.DataFrame()


def _build_score_sampler(lambda_home: float, lambda_away: float) -> dict[str, np.ndarray]:
    """Precompute a scoreline sampling table for one fixture.

    v0.48 keeps the same independent scoreline distribution as previous
    versions, but avoids rebuilding the score matrix on every Monte Carlo
    iteration. This makes larger runs such as 50,000 simulations practical
    without changing the probability model.
    """
    dist = scoreline_distribution(lambda_home, lambda_away, max_goals=10, normalize=True)
    values = dist.matrix.to_numpy(dtype=float).ravel()
    probabilities = values / values.sum()
    home_idx, away_idx = np.unravel_index(np.arange(len(probabilities)), dist.matrix.shape)
    home_goals = dist.matrix.index.to_numpy(dtype=int)[home_idx]
    away_goals = dist.matrix.columns.to_numpy(dtype=int)[away_idx]
    return {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "probabilities": probabilities,
    }


def _sample_score_from_sampler(sampler: dict[str, np.ndarray], rng: np.random.Generator) -> tuple[int, int]:
    probabilities = sampler["probabilities"]
    idx = int(rng.choice(np.arange(len(probabilities)), p=probabilities))
    return int(sampler["home_goals"][idx]), int(sampler["away_goals"][idx])


def _sample_scores_from_sampler(
    sampler: dict[str, np.ndarray],
    rng: np.random.Generator,
    size: int,
) -> dict[str, np.ndarray]:
    probabilities = sampler["probabilities"]
    indices = rng.choice(np.arange(len(probabilities)), size=size, p=probabilities)
    return {
        "home_goals": sampler["home_goals"][indices],
        "away_goals": sampler["away_goals"][indices],
    }


def _pre_sample_group_scores(
    group_fixtures: pd.DataFrame,
    score_samplers: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
    n_simulations: int,
) -> dict[str, dict[str, np.ndarray]]:
    sampled: dict[str, dict[str, np.ndarray]] = {}
    for _, fixture in group_fixtures.iterrows():
        match_id = str(fixture["match_id"])
        sampler = score_samplers.get(match_id)
        if sampler is not None:
            sampled[match_id] = _sample_scores_from_sampler(sampler, rng, n_simulations)
    return sampled


def _sample_score(lambda_home: float, lambda_away: float, rng: np.random.Generator) -> tuple[int, int]:
    """Backward-compatible one-off score sampler."""
    return _sample_score_from_sampler(_build_score_sampler(lambda_home, lambda_away), rng)


def _apply_result(table: dict, home: str, away: str, h: int, a: int) -> None:
    table[home]["gf"] += h
    table[home]["ga"] += a
    table[away]["gf"] += a
    table[away]["ga"] += h
    table[home]["gd"] = table[home]["gf"] - table[home]["ga"]
    table[away]["gd"] = table[away]["gf"] - table[away]["ga"]
    if h > a:
        table[home]["points"] += 3
    elif a > h:
        table[away]["points"] += 3
    else:
        table[home]["points"] += 1
        table[away]["points"] += 1


def _build_group_definitions(group_fixtures: pd.DataFrame, fallback_teams: list[str]) -> list[list[str]]:
    if "group" not in group_fixtures.columns or not group_fixtures["group"].astype(str).ne("unknown").any():
        return []
    groups: list[list[str]] = []
    for _, gf in group_fixtures.groupby("group"):
        group_teams = sorted(set(gf["home_team"]).union(set(gf["away_team"])))
        if group_teams:
            groups.append(group_teams)
    return groups or [fallback_teams]


def _build_rating_lookup(match_predictions: pd.DataFrame) -> dict[str, float]:
    rating: dict[str, float] = {}
    for _, r in match_predictions.iterrows():
        rating[canonical_name(r.get("home_team"))] = float(r.get("home_rating", 1500))
        rating[canonical_name(r.get("away_team"))] = float(r.get("away_rating", 1500))
    return rating


def _rank_teams(teams: list[str], table: dict) -> list[str]:
    return sorted(teams, key=lambda t: (table[t]["points"], table[t]["gd"], table[t]["gf"], t), reverse=True)


def _simulate_knockout_or_champion(qualified: list[str], rating: dict[str, float], rng: np.random.Generator) -> tuple[str | None, dict[str, list[str]]]:
    if not qualified:
        return None, {"r16": [], "qf": [], "sf": [], "final": []}
    teams = list(dict.fromkeys([canonical_name(t) for t in qualified]))
    stage_counts = {"r16": teams.copy(), "qf": [], "sf": [], "final": []}
    # Approximate bracket from profile ratings if explicit knockout fixtures are not supplied.
    current = sorted(teams, key=lambda t: rating.get(t, 1500), reverse=True)
    stage_sequence = ["qf", "sf", "final"]
    for stage in stage_sequence:
        if len(current) <= 1:
            break
        winners = []
        pairings = list(zip(current[::2], current[1::2]))
        if len(current) % 2 == 1:
            winners.append(current[-1])
        for t1, t2 in pairings:
            probs = softmax([rating.get(t1, 1500), rating.get(t2, 1500)], temperature=250.0)
            winners.append(t1 if rng.random() < probs[0] else t2)
        stage_counts[stage] = winners.copy()
        current = winners
    return (current[0] if current else None), stage_counts


def _top_scorer_probabilities(player_events: pd.DataFrame | None) -> pd.DataFrame:
    if player_events is None or player_events.empty:
        return pd.DataFrame()
    work = player_events[player_events["market"].astype(str).eq("player_shots")].copy()
    if work.empty:
        return pd.DataFrame()
    # Goals are experimental in v0.20; approximate expected goals from shots with a conservative 0.10 conversion.
    work["expected_goals"] = pd.to_numeric(work["expected_count"], errors="coerce").fillna(0) * 0.10
    by_player = work.groupby(["player", "team"], dropna=False)["expected_goals"].sum().reset_index()
    total = by_player["expected_goals"].sum()
    if total <= 0:
        by_player["top_scorer_probability_approx"] = 0.0
    else:
        by_player["top_scorer_probability_approx"] = by_player["expected_goals"] / total
    by_player["record_type"] = "top_scorer_experimental"
    return by_player.sort_values("top_scorer_probability_approx", ascending=False).reset_index(drop=True)
