from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Dict

import pandas as pd

from mundialytics.identity.normalization import canonical_team_name


@dataclass
class EloConfig:
    initial_rating: float = 1500.0
    k_base: float = 32.0
    home_advantage: float = 55.0
    goal_diff_multiplier: bool = True
    tournament_weights: dict[str, float] = field(default_factory=lambda: {
        "friendly": 0.70,
        "qualifier": 1.00,
        "continental": 1.15,
        "world_cup": 1.25,
    })


class EloRater:
    """Simple, reproducible football ELO rater.

    It can be used for national teams or clubs. For national-tournament neutral
    games set neutral=1, which removes home advantage.
    """

    def __init__(self, config: EloConfig | None = None):
        self.config = config or EloConfig()
        self.ratings: Dict[str, float] = {}
        self.history: list[dict] = []

    def get(self, team: str) -> float:
        return self.ratings.get(canonical_team_name(team), self.config.initial_rating)

    def expected_score(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    @staticmethod
    def actual_score(goals_for: float, goals_against: float) -> float:
        if goals_for > goals_against:
            return 1.0
        if goals_for == goals_against:
            return 0.5
        return 0.0

    @staticmethod
    def goal_multiplier(goal_diff: float) -> float:
        gd = abs(goal_diff)
        if gd <= 1:
            return 1.0
        return log(gd + 1.0) * 1.10

    def competition_weight(self, competition: str | None) -> float:
        comp = (competition or "").lower().replace(" ", "_")
        if "friendly" in comp:
            return self.config.tournament_weights.get("friendly", 0.7)
        if "qual" in comp:
            return self.config.tournament_weights.get("qualifier", 1.0)
        if "world" in comp:
            return self.config.tournament_weights.get("world_cup", 1.25)
        if "euro" in comp or "copa" in comp or "afcon" in comp:
            return self.config.tournament_weights.get("continental", 1.15)
        return 1.0

    def update_match(self, row: pd.Series | dict) -> dict:
        home = canonical_team_name(row["home_team"])
        away = canonical_team_name(row["away_team"])
        hg = float(row["home_goals"])
        ag = float(row["away_goals"])
        neutral = int(row.get("neutral", 0) or 0)
        competition = row.get("competition", "")

        ra = self.get(home)
        rb = self.get(away)
        adj_ra = ra if neutral else ra + self.config.home_advantage
        expected_home = self.expected_score(adj_ra, rb)
        actual_home = self.actual_score(hg, ag)

        k = self.config.k_base * self.competition_weight(competition)
        if self.config.goal_diff_multiplier:
            k *= self.goal_multiplier(hg - ag)
        delta = k * (actual_home - expected_home)

        self.ratings[home] = ra + delta
        self.ratings[away] = rb - delta

        record = {
            "match_id": row.get("match_id"),
            "date": row.get("date"),
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "home_elo_pre": ra,
            "away_elo_pre": rb,
            "home_elo_post": self.ratings[home],
            "away_elo_post": self.ratings[away],
            "elo_diff_pre": ra - rb,
            "expected_home": expected_home,
            "delta_home": delta,
        }
        self.history.append(record)
        return record

    def fit(self, matches: pd.DataFrame) -> pd.DataFrame:
        completed = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        completed = completed.sort_values(["date", "match_id"])
        for _, row in completed.iterrows():
            self.update_match(row)
        return pd.DataFrame(self.history)

    def transform_fixture(self, home_team: str, away_team: str, neutral: int = 1) -> dict:
        home_team = canonical_team_name(home_team)
        away_team = canonical_team_name(away_team)
        home_elo = self.get(home_team)
        away_elo = self.get(away_team)
        adj_home = home_elo if neutral else home_elo + self.config.home_advantage
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "expected_home_score_elo": self.expected_score(adj_home, away_elo),
        }
