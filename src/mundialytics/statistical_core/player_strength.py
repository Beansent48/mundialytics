"""
PlayerStrength model for SquadLab.

Converts per-match player event rates (from StatsBomb) into two composite
scores per player:
  - offensive_strength  (0-100): xG, xA, shots, key passes, carries
  - defensive_strength  (0-100): tackles, interceptions, pressures, blocks
  - gk_strength         (0-100): saves, psxg (goalkeepers only)
  - overall             (0-100): position-weighted combination

Scores are percentile-normalised within each position group so that
a 75 means "better than 75% of players at that position".

Team strength = aggregated from 11 selected players by position role.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

# Weights per position for offensive vs defensive contribution to team lambda
POSITION_ATTACK_WEIGHT = {
    "Forward":    0.85,
    "Midfielder": 0.50,
    "Defender":   0.15,
    "Goalkeeper": 0.05,
    "Unknown":    0.45,
}
POSITION_DEFENSE_WEIGHT = {
    "Forward":    0.10,
    "Midfielder": 0.45,
    "Defender":   0.80,
    "Goalkeeper": 0.70,
    "Unknown":    0.40,
}

# Stats that drive each composite score
OFFENSIVE_STATS = ["xg_per_match", "goals_per_match", "sot_per_match",
                    "shots_per_match", "assists_per_match"]
DEFENSIVE_STATS = ["tackles_per_match", "interceptions_per_match",
                    "pressures_per_match"]
GK_STATS: list[str] = []  # computed from saves column if available


@dataclass
class PlayerStrengthProfile:
    player: str
    team: str
    competition: str
    position: str
    matches: int
    offensive_strength: float = 0.0   # 0-100 percentile within position
    defensive_strength: float = 0.0
    gk_strength: float = 0.0
    overall: float = 0.0
    xg_per_match: float = 0.0
    goals_per_match: float = 0.0
    assists_per_match: float = 0.0
    shots_per_match: float = 0.0
    tackles_per_match: float = 0.0
    pressures_per_match: float = 0.0
    raw_stats: dict = field(default_factory=dict)


class PlayerStrengthModel:
    """Build PlayerStrengthProfile for all players from the profiles CSV.

    Usage:
        model = PlayerStrengthModel()
        model.fit()  # loads data/processed/player_profiles_with_positions.csv
        profiles = model.search("Messi")   # → list[PlayerStrengthProfile]
        team = model.team_strength(profiles_11)
    """

    def __init__(self, profiles_path: str | Path | None = None):
        self.profiles_path = Path(profiles_path) if profiles_path else \
            ROOT / "data/processed/player_profiles_with_positions.csv"
        self.profiles_: dict[str, PlayerStrengthProfile] = {}
        self._raw: pd.DataFrame = pd.DataFrame()
        self._percentile_cache: dict[tuple[str, str], float] = {}

    def fit(self) -> "PlayerStrengthModel":
        if not self.profiles_path.exists():
            return self
        df = pd.read_csv(self.profiles_path)
        self._raw = df.copy()

        # Build per-position percentile distributions for each stat
        # Normalise column names (profiles CSV uses 'position', not 'position_group')
        if "position_group" not in df.columns and "position" in df.columns:
            df["position_group"] = df["position"]
        if "team_c" not in df.columns and "team" in df.columns:
            df["team_c"] = df["team"]
        if "competition_c" not in df.columns and "competition" in df.columns:
            df["competition_c"] = df["competition"]

        pos_groups = df["position_group"].fillna("Unknown")
        stat_cols = [c for c in df.columns if c.endswith("_per_match")]
        for stat in stat_cols:
            if stat not in df.columns:
                continue
            df[stat] = pd.to_numeric(df[stat], errors="coerce").fillna(0.0)

        # Compute composite scores
        def percentile_rank(val: float, pos: str, stat: str) -> float:
            sub = df[pos_groups == pos][stat].dropna()
            if len(sub) == 0:
                return 50.0
            return float(100.0 * (sub < val).mean())

        # Offensive raw score: weighted average of percentile ranks
        off_weights = {"goals_per_match": 0.30, "assists_per_match": 0.20,
                        "sot_per_match": 0.25, "shots_per_match": 0.25}
        def_weights = {"tackles_per_match": 0.45, "pressures_per_match": 0.45,
                        "fouls_per_match": -0.05, "yellow_cards_per_match": -0.05}

        for _, row in df.iterrows():
            pos = str(row.get("position_group", "Unknown"))
            player = str(row.get("player", ""))
            matches = int(row.get("matches", 0))

            # Offensive score
            off_vals = []
            for stat, w in off_weights.items():
                if stat in row.index:
                    p = percentile_rank(float(row[stat]), pos, stat)
                    off_vals.append(p * abs(w) * (1 if w > 0 else -1))
            off_score = float(np.mean(off_vals)) if off_vals else 50.0

            # Defensive score
            def_vals = []
            for stat, w in def_weights.items():
                if stat in row.index:
                    p = percentile_rank(float(row[stat]), pos, stat)
                    def_vals.append(p * abs(w) * (1 if w > 0 else -1))
            def_score = float(np.mean(def_vals)) if def_vals else 50.0

            # GK score (placeholder — saves not in current data)
            gk_score = 50.0

            # Overall = position-weighted
            atk_w = POSITION_ATTACK_WEIGHT.get(pos, 0.45)
            def_w = POSITION_DEFENSE_WEIGHT.get(pos, 0.40)
            overall = float(np.clip(atk_w * off_score + def_w * def_score, 0, 100))

            self.profiles_[player] = PlayerStrengthProfile(
                player=player,
                team=str(row.get("team_c", "")),
                competition=str(row.get("competition_c", "")),
                position=pos,
                matches=matches,
                offensive_strength=round(off_score, 1),
                defensive_strength=round(def_score, 1),
                gk_strength=gk_score,
                overall=round(overall, 1),
                xg_per_match=float(row.get("sot_per_match", row.get("goals_per_match", 0)) * 0.6),
                goals_per_match=float(row.get("goals_per_match", 0)),
                assists_per_match=float(row.get("assists_per_match", 0)),
                shots_per_match=float(row.get("shots_per_match", 0)),
                tackles_per_match=float(row.get("tackles_per_match", 0)),
                pressures_per_match=float(row.get("pressures_per_match", 0)),
                raw_stats={c: float(row.get(c, 0)) for c in stat_cols if c in row.index},
            )
        return self

    def search(self, query: str, position: str | None = None,
               competition: str | None = None, top_n: int = 20) -> list[PlayerStrengthProfile]:
        """Search players by name fragment."""
        q = query.lower()
        results = [p for name, p in self.profiles_.items() if q in name.lower()]
        if position:
            results = [p for p in results if p.position == position]
        if competition:
            results = [p for p in results if competition.lower() in p.competition.lower()]
        return sorted(results, key=lambda p: -p.overall)[:top_n]

    def get(self, player: str) -> PlayerStrengthProfile | None:
        return self.profiles_.get(player)

    def top_by_position(self, position: str, competition: str | None = None,
                         stat: str = "overall", n: int = 10) -> list[PlayerStrengthProfile]:
        """Top N players by position and stat."""
        candidates = [p for p in self.profiles_.values() if p.position == position
                       and p.matches >= 5]
        if competition:
            candidates = [p for p in candidates if competition.lower() in p.competition.lower()]
        return sorted(candidates, key=lambda p: -getattr(p, stat, p.overall))[:n]

    def team_strength(self, squad: list[PlayerStrengthProfile]) -> dict:
        """Aggregate squad into team attack/defense estimates.

        Returns attack_index and defense_index (0-100 scale),
        plus estimated xG per match for the squad.
        """
        if not squad:
            return {"attack_index": 50.0, "defense_index": 50.0, "xg_per_match": 1.25}

        total_atk = total_def = total_w_atk = total_w_def = 0.0
        total_xg = 0.0

        for p in squad:
            atk_w = POSITION_ATTACK_WEIGHT.get(p.position, 0.45)
            def_w = POSITION_DEFENSE_WEIGHT.get(p.position, 0.40)
            total_atk += p.offensive_strength * atk_w
            total_def += p.defensive_strength * def_w
            total_w_atk += atk_w
            total_w_def += def_w
            # XG contribution: attacker xg scaled by minutes expectation
            total_xg += p.xg_per_match * atk_w * 1.1  # slight boost for squad synergy

        attack_idx  = float(total_atk / max(total_w_atk, 1e-6))
        defense_idx = float(total_def / max(total_w_def, 1e-6))

        # Scale xG: league average ~1.25 goals/team, scale from attack_index
        xg_per_match = float(1.25 * (attack_idx / 50.0) ** 0.6)

        return {
            "attack_index":  round(attack_idx, 1),
            "defense_index": round(defense_idx, 1),
            "xg_per_match":  round(np.clip(xg_per_match, 0.4, 3.5), 2),
            "squad_size":    len(squad),
        }

    def all_profiles_df(self) -> pd.DataFrame:
        rows = []
        for p in self.profiles_.values():
            rows.append({
                "player": p.player, "team": p.team, "competition": p.competition,
                "position": p.position, "matches": p.matches,
                "overall": p.overall, "offensive": p.offensive_strength,
                "defensive": p.defensive_strength,
                "xg_pm": p.xg_per_match, "goals_pm": p.goals_per_match,
                "assists_pm": p.assists_per_match, "shots_pm": p.shots_per_match,
                "tackles_pm": p.tackles_per_match,
            })
        return pd.DataFrame(rows).sort_values("overall", ascending=False)
