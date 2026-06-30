"""
PlayerStrength model for SquadLab.

Converts per-match player event rates (from StatsBomb) into two composite
scores per player:
  - offensive_strength  (0-100): xG, xA, shots, key passes, carries
  - defensive_strength  (0-100): tackles, interceptions, pressures, blocks
  - gk_strength         (0-100): saves, psxg (goalkeepers only)
  - overall             (0-100): position-weighted combination

offensive_strength/defensive_strength are percentile-normalised within each
position group (a 75 means "better than 75% of players at that position").
overall blends the two with position-specific weights, then ranks that blend
GLOBALLY across every position so star quality from any position can reach
the top — capped at OVERALL_CEILING (~92) rather than a flat 100, so a real
top-tier player reads like one instead of "the platonic ideal".

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
                    "assists_per_match", "big_chances_missed_per_match"]
DEFENSIVE_STATS = ["tackles_per_match", "interceptions_per_match",
                    "pressures_per_match"]
# GK rating (saves, clean sheets) is intentionally not implemented yet: real
# goalkeeper data (data/processed/goalkeeper_match_stats.csv) exists but isn't
# merged into player_profiles_with_positions.csv. gk_strength stays a 50.0
# placeholder and goalkeepers fall back to the generic offense/defense blend.
GK_STATS: list[str] = []

# Players with few matches get their percentile shrunk toward the 50 (average)
# baseline so a hot streak over 1-3 games can't outrank a proven starter.
# credibility = matches / (matches + SHRINKAGE_MATCHES)
SHRINKAGE_MATCHES = 8.0

# Percentile-based scores are centred on 50 by construction (half the pool is
# always below average). That reads as "everyone is mediocre" on a 0-100 card.
# Bending the curve (overall ** CURVE_EXPONENT, then scaled to OVERALL_CEILING)
# keeps 0->0 and pushes a league-average player up toward ~65, while capping
# the single best player in the whole pool at ~92 instead of a flat 100 — a
# literal 100 reads as "the platonic ideal player", whereas a real-world star
# (any position) should read 90-92, leaving room above for outliers.
CURVE_EXPONENT = 0.5
OVERALL_CEILING = 92.0

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

        # Pre-compute percentile rank columns vectorised (O(n·stats) instead of O(n²))
        # scipy.stats.rankdata normalised to 0-100 within each position group
        # Goals and assists are weighted equally (both are a finished chance for
        # the team); xg_per_match (real shot-level StatsBomb xG, see
        # scripts/enrich_player_profiles_with_xg.py) replaces raw shot/SOT volume
        # since it already captures shot quality and quantity together.
        # big_chance_miss_rate (missed / created, both xg>=0.3) penalises
        # wasteful finishing as a RATE rather than a raw count, so a
        # high-volume elite scorer isn't dragged down just for taking more
        # big chances than everyone else.
        off_weights = {"goals_per_match": 0.30, "assists_per_match": 0.30,
                        "xg_per_match": 0.25, "big_chance_miss_rate": -0.15}
        def_weights = {"tackles_per_match": 0.45, "pressures_per_match": 0.45,
                        "fouls_per_match": -0.05, "yellow_cards_per_match": -0.05}

        pct_cols: dict[str, str] = {}
        for stat in list(off_weights) + list(def_weights):
            if stat not in df.columns:
                continue
            pct_col = f"_pct_{stat}"
            df[pct_col] = 50.0
            for pos in df["position_group"].unique():
                mask = df["position_group"] == pos
                vals = df.loc[mask, stat].fillna(0.0)
                if len(vals) > 1:
                    from scipy.stats import rankdata
                    df.loc[mask, pct_col] = 100.0 * (rankdata(vals) - 1) / (len(vals) - 1)
            pct_cols[stat] = pct_col

        # Pass 1: per-row offensive/defensive composites — weighted averages
        # of credibility-shrunk percentiles. off_score and def_score live on
        # different stat sets with different distribution shapes (e.g.
        # defensive percentiles cluster higher at the top because
        # tackles/pressures are correlated workrate stats and there are only
        # two of them, while the offensive blend averages three positively
        # weighted stats plus a penalty, which is structurally harder to max
        # out) — so they are NOT comparable to each other yet.
        row_data: list[dict] = []
        for idx, row in df.iterrows():
            pos = str(row.get("position_group", "Unknown"))
            matches = int(row.get("matches", 0))
            credibility = matches / (matches + SHRINKAGE_MATCHES)

            def shrunk_pct(raw_pct: float, credibility: float = credibility) -> float:
                """Regress toward the 50 baseline for players with few matches."""
                return 50.0 + (raw_pct - 50.0) * credibility

            off_num = off_den = 0.0
            for stat, w in off_weights.items():
                pcol = pct_cols.get(stat)
                if pcol and pcol in row.index:
                    p = shrunk_pct(float(row[pcol]))
                    off_num += p * abs(w) * (1 if w > 0 else -1)
                    off_den += abs(w)
            off_score = float(np.clip(off_num / off_den, 0, 100)) if off_den > 0 else 50.0

            def_num = def_den = 0.0
            for stat, w in def_weights.items():
                pcol = pct_cols.get(stat)
                if pcol and pcol in row.index:
                    p = shrunk_pct(float(row[pcol]))
                    def_num += p * abs(w) * (1 if w > 0 else -1)
                    def_den += abs(w)
            def_score = float(np.clip(def_num / def_den, 0, 100)) if def_den > 0 else 50.0

            row_data.append({"idx": idx, "off_score": off_score, "def_score": def_score})

        scores_df = pd.DataFrame(row_data).set_index("idx")
        df = df.join(scores_df)

        # Pass 2: re-rank off_score and def_score to a percentile WITHIN each
        # position group, independently of each other. This is what makes
        # attack and defense comparable: the best forward by off_score and
        # the best defender by def_score both land at off_pct/def_pct = 100,
        # removing the structural ceiling gap from Pass 1 (different stat
        # counts/correlations) at the component level — not by forcing the
        # final blended overall to tie, just the inputs to that blend.
        from scipy.stats import rankdata

        def _rerank_within_position(col: str, out_col: str) -> None:
            df[out_col] = 50.0
            for pos in df["position_group"].unique():
                mask = df["position_group"] == pos
                vals = df.loc[mask, col]
                if len(vals) > 1:
                    df.loc[mask, out_col] = 100.0 * (rankdata(vals) - 1) / (len(vals) - 1)

        _rerank_within_position("off_score", "off_pct")
        _rerank_within_position("def_score", "def_pct")

        atk_w_s = df["position_group"].map(POSITION_ATTACK_WEIGHT).fillna(0.45)
        def_w_s = df["position_group"].map(POSITION_DEFENSE_WEIGHT).fillna(0.40)
        w_sum_s = atk_w_s + def_w_s
        df["overall_raw"] = (atk_w_s / w_sum_s) * df["off_pct"] + (def_w_s / w_sum_s) * df["def_pct"]

        # Pass 3: rank overall_raw GLOBALLY (across every position at once,
        # not within each position group), so star quality can cluster near
        # the top from whichever positions actually have it instead of
        # forcing the #1 player at every position to tie at the same value.
        if len(df) > 1:
            df["_overall_pctile"] = 100.0 * (rankdata(df["overall_raw"]) - 1) / (len(df) - 1)
        else:
            df["_overall_pctile"] = 50.0

        # FIFA-style curve, then rescaled so the single best player in the
        # whole pool reads ~OVERALL_CEILING (92) instead of a flat 100, and
        # an average player still reads ~65.
        df["overall"] = OVERALL_CEILING * (df["_overall_pctile"].clip(0, 100) / 100.0) ** CURVE_EXPONENT

        for _, row in df.iterrows():
            pos = str(row.get("position_group", "Unknown"))
            player = str(row.get("player", ""))
            matches = int(row.get("matches", 0))
            # Displayed off/def scores are the within-position re-rank
            # (off_pct/def_pct), the same comparable scale the overall blend
            # uses — not the raw stat-weighted composite from Pass 1.
            off_score = float(row["off_pct"])
            def_score = float(row["def_pct"])
            overall = float(np.clip(row["overall"], 0, 100))

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
