"""
PlayerStrength model for SquadLab.

Converts per-match player event rates (from StatsBomb) into two composite
scores per player:
  - offensive_strength  (0-100): goals, assists, xG, big-chance conversion
  - defensive_strength  (0-100): tackles, pressures, discipline
  - gk_strength         (0-100): saves, psxg (goalkeepers only — placeholder)
  - overall             (0-100): position-weighted combination

This is an ABSOLUTE rating, not a percentile rank. Each stat is mapped to a
0-100 score through a fixed, hand-calibrated curve (see the ANCHOR_* tables
below) anchored on real reference points — e.g. peak Messi's 0.83
goals/match reads ~95, a median established forward's 0.16 reads ~58. The
curve is calibrated once from the data (median/p75/p90/p95 of players with
20+ matches) but does NOT re-rank players against each other at score time.

This was a deliberate move away from an earlier percentile-based design: a
percentile rank compares each player only to whoever else happens to be in
the data pool, and that pool is dominated by short-sample cameo players
(the median "Forward" row has just 4 matches) — so a merely-good player with
a real sample could out-rank historically great players just by being a
credible regular in a noisy pool (e.g. Troy Deeney over Messi). An absolute
anchor curve doesn't have that failure mode: the same raw stat always maps
to the same score regardless of who else is in the dataset.

Small samples are still discounted via SHRINKAGE_MATCHES (pulls the score
toward a neutral 50 baseline until a player has accumulated enough matches
to trust the rate), so a 3-game hot streak can't fake an anchor score.

Men's competitions only — see WOMENS_COMPETITION_MARKERS.

Team strength = aggregated from 11 selected players by position role.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

# Weights per position for offensive vs defensive contribution to team lambda.
# Only the ratio between the two matters (team_strength() and fit() both
# renormalise so they sum to 1), so they're written pre-normalised here.
POSITION_ATTACK_WEIGHT = {
    "Forward":    0.8947,   # was 0.85 / 0.95
    "Midfielder": 0.5263,   # was 0.50 / 0.95
    "Defender":   0.1579,   # was 0.15 / 0.95
    "Goalkeeper": 0.0667,   # was 0.05 / 0.75
    "Unknown":    0.5294,   # was 0.45 / 0.85
}
POSITION_DEFENSE_WEIGHT = {
    "Forward":    0.1053,
    "Midfielder": 0.4737,
    "Defender":   0.8421,
    "Goalkeeper": 0.9333,
    "Unknown":    0.4706,
}

# Stats that drive each composite score
OFFENSIVE_STATS = ["xg_per_match", "goals_per_match",
                    "assists_per_match", "big_chance_miss_rate"]
DEFENSIVE_STATS = ["tackles_per_match", "pressures_per_match",
                    "fouls_per_match", "yellow_cards_per_match"]
# GK rating (saves, clean sheets) is intentionally not implemented yet: real
# goalkeeper data (data/processed/goalkeeper_match_stats.csv) exists but isn't
# merged into player_profiles_with_positions.csv. gk_strength stays a 50.0
# placeholder and goalkeepers fall back to the generic offense/defense blend.
GK_STATS: list[str] = []

# Players with few matches get their score shrunk toward the 50 (neutral)
# baseline so a 2-3 game hot streak can't fake an elite anchor score.
# credibility = matches / (matches + SHRINKAGE_MATCHES)
SHRINKAGE_MATCHES = 12.0

# --- Absolute anchor curves -------------------------------------------------
# Each table is a sorted list of (raw_stat_value, score) control points;
# values are linearly interpolated (and clamped at the ends) via np.interp.
# Calibrated from two reference sets pulled directly from
# player_profiles_with_positions.csv:
#   1. Players with 20+ matches at each position (median/p75/p90/p95), so the
#      mid-curve reflects "what a real established player's rate looks like".
#   2. Recognisable world-class reference players (peak Messi 0.83
#      goals/match, Ronaldo 0.78 xG/match, Suarez/Mbappe ~0.69, etc.) anchor
#      the top end so genuine stars land around 90-95, not just "best of a
#      noisy pool".
# All curves are pre-oriented so higher score == better (the miss-rate and
# discipline curves are already inverted, unlike the old percentile weights
# which needed a +/-1 sign flip).
ANCHOR_GOALS_PER_MATCH = [
    (0.00, 35), (0.08, 48), (0.156, 58), (0.231, 67),
    (0.343, 75), (0.43, 80), (0.60, 87), (0.834, 95),
]
ANCHOR_ASSISTS_PER_MATCH = [
    (0.00, 35), (0.04, 48), (0.083, 58), (0.135, 66),
    (0.20, 74), (0.238, 79), (0.35, 88), (0.50, 95),
]
ANCHOR_XG_PER_MATCH = [
    (0.00, 35), (0.07, 48), (0.142, 58), (0.233, 66),
    (0.348, 74), (0.442, 80), (0.60, 88), (0.777, 95),
]
# Lower miss rate = better. Deliberately narrow band (48-68): this stat is
# noisy for players with very few big chances, so it shouldn't swing the
# offensive score on its own the way goals/assists/xG can.
ANCHOR_BIG_CHANCE_MISS_RATE = [
    (0.0, 68), (0.33, 62), (0.43, 60), (0.67, 55), (1.0, 48),
]
ANCHOR_TACKLES_PER_MATCH = [
    (0.0, 35), (0.5, 46), (1.0, 52), (1.754, 60),
    (2.081, 67), (2.519, 75), (2.726, 80), (3.95, 92),
]
ANCHOR_PRESSURES_PER_MATCH = [
    (0.0, 35), (5.0, 48), (10.5, 58), (12.18, 65),
    (14.23, 72), (15.45, 77), (21.5, 90),
]
# Fewer fouls = better.
ANCHOR_FOULS_PER_MATCH = [
    (0.0, 70), (0.5, 65), (0.937, 60), (1.17, 56), (1.4, 50), (2.18, 35),
]
# Fewer cards = better.
ANCHOR_YELLOW_CARDS_PER_MATCH = [
    (0.0, 68), (0.06, 63), (0.121, 60), (0.185, 55), (0.24, 50), (0.47, 35),
]

ANCHOR_CURVES: dict[str, list[tuple[float, float]]] = {
    "goals_per_match": ANCHOR_GOALS_PER_MATCH,
    "assists_per_match": ANCHOR_ASSISTS_PER_MATCH,
    "xg_per_match": ANCHOR_XG_PER_MATCH,
    "big_chance_miss_rate": ANCHOR_BIG_CHANCE_MISS_RATE,
    "tackles_per_match": ANCHOR_TACKLES_PER_MATCH,
    "pressures_per_match": ANCHOR_PRESSURES_PER_MATCH,
    "fouls_per_match": ANCHOR_FOULS_PER_MATCH,
    "yellow_cards_per_match": ANCHOR_YELLOW_CARDS_PER_MATCH,
}

# Competitions to exclude entirely from the rating pool. Men's and women's
# football have different physical/statistical baselines (event rates aren't
# directly comparable), and StatsBomb open data mixes both into the same
# player pool — without this filter, percentile ranks and the global overall
# ranking below would compare them as if they were one population.
WOMENS_COMPETITION_MARKERS = ("women", "frauen", "liga f", "nwsl", "féminine", "feminine")


@dataclass
class PlayerStrengthProfile:
    player: str
    team: str
    competition: str
    position: str
    matches: int
    offensive_strength: float = 0.0   # 0-100 absolute (anchor-curve) score
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

        if "competition" in df.columns:
            comp_lower = df["competition"].fillna("").str.lower()
            is_womens = comp_lower.apply(
                lambda c: any(marker in c for marker in WOMENS_COMPETITION_MARKERS)
            )
            df = df.loc[~is_womens].reset_index(drop=True)

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
        if "big_chance_miss_rate" in df.columns:
            df["big_chance_miss_rate"] = pd.to_numeric(df["big_chance_miss_rate"], errors="coerce").fillna(0.0)

        # Goals and assists are weighted equally (both are a finished chance
        # for the team); xg_per_match (real shot-level StatsBomb xG, see
        # scripts/enrich_player_profiles_with_xg.py) replaces raw shot/SOT
        # volume since it already captures shot quality and quantity
        # together. All weights are positive: each stat's ANCHOR_CURVES entry
        # is already oriented so a higher score is always better (miss rate
        # and discipline stats are pre-inverted).
        off_weights = {"goals_per_match": 0.30, "assists_per_match": 0.30,
                        "xg_per_match": 0.25, "big_chance_miss_rate": 0.15}
        def_weights = {"tackles_per_match": 0.45, "pressures_per_match": 0.45,
                        "fouls_per_match": 0.05, "yellow_cards_per_match": 0.05}

        # Map every stat through its fixed anchor curve (absolute, not
        # ranked against the rest of the pool) via np.interp — vectorised,
        # values outside the anchor range just clamp to the nearest end.
        score_cols: dict[str, str] = {}
        for stat in list(off_weights) + list(def_weights):
            if stat not in df.columns:
                continue
            anchors = ANCHOR_CURVES[stat]
            xs, ys = zip(*anchors)
            score_col = f"_score_{stat}"
            df[score_col] = np.interp(df[stat], xs, ys)
            score_cols[stat] = score_col

        # Shrink each stat's anchor score toward the neutral 50 baseline for
        # players with few matches, so a 2-3 game hot streak can't fake an
        # elite anchor score (e.g. 2 goals in 3 matches = 0.67 goals/match,
        # which would otherwise read as near-Messi on the raw curve alone).
        credibility = df["matches"] / (df["matches"] + SHRINKAGE_MATCHES)
        for stat, score_col in score_cols.items():
            df[score_col] = 50.0 + (df[score_col] - 50.0) * credibility

        off_num = sum(df[score_cols[s]] * w for s, w in off_weights.items() if s in score_cols)
        off_den = sum(w for s, w in off_weights.items() if s in score_cols)
        df["off_score"] = (off_num / off_den).clip(0, 100) if off_den > 0 else 50.0

        def_num = sum(df[score_cols[s]] * w for s, w in def_weights.items() if s in score_cols)
        def_den = sum(w for s, w in def_weights.items() if s in score_cols)
        df["def_score"] = (def_num / def_den).clip(0, 100) if def_den > 0 else 50.0

        # overall = position-weighted blend of the two absolute scores. No
        # percentile re-rank, no curve — the anchor tables already calibrate
        # the 0-100 range (median established player ~58-60, genuine legends
        # ~90-95), so the blend doesn't need further reshaping.
        atk_w_s = df["position_group"].map(POSITION_ATTACK_WEIGHT).fillna(0.45)
        def_w_s = df["position_group"].map(POSITION_DEFENSE_WEIGHT).fillna(0.40)
        w_sum_s = atk_w_s + def_w_s
        df["overall"] = (
            (atk_w_s / w_sum_s) * df["off_score"] + (def_w_s / w_sum_s) * df["def_score"]
        ).clip(0, 100)

        # Goalkeepers are a special case: the tackles/pressures anchor curve
        # is calibrated on outfield defenders, and keepers naturally record
        # almost none of either, so the blend above crushes every keeper to
        # ~43-44 regardless of how good they actually are — that's a curve
        # mismatch, not a real signal. Until a dedicated save%/clean-sheet GK
        # rating exists (see GK_STATS), keep their overall at the same
        # neutral 50 placeholder as gk_strength instead of showing a
        # misleadingly bad number.
        df.loc[df["position_group"] == "Goalkeeper", "overall"] = 50.0

        for _, row in df.iterrows():
            pos = str(row.get("position_group", "Unknown"))
            player = str(row.get("player", ""))
            matches = int(row.get("matches", 0))
            off_score = float(row["off_score"])
            def_score = float(row["def_score"])
            overall = float(row["overall"])

            # GK score (placeholder — saves not in current data)
            gk_score = 50.0

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
                xg_per_match=float(row.get("xg_per_match", 0)),
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
