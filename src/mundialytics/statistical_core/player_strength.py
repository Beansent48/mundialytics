"""
PlayerStrength model for SquadLab.

Converts per-match player event rates (from StatsBomb) into two composite
scores per player:
  - offensive_strength  (0-100): goals, assists, xG, big-chance conversion
  - defensive_strength  (0-100): tackles, pressures, discipline
  - gk_strength         (0-100): real save%, goals-conceded/match, clean-sheet
                                 rate from goalkeeper_match_stats.csv
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

# Weights per position for offensive / defensive / creation contribution to
# team lambda. Only the ratio between the three matters (team_strength() and
# fit() both renormalise so they sum to 1), so they're written pre-normalised
# here. A third "creation" axis was added 2026-07-01 (see CREATION_STATS
# below) after confirming the pure off/def split gave midfielders no way to
# score for chance creation through passing -- real elite creators (Xavi,
# Modric, Kroos, De Bruyne) ranked below destroyers who simply out-tackled
# them, since neither goals/assists/xG nor tackles/pressures capture "creates
# play through passing".
POSITION_ATTACK_WEIGHT = {
    "Forward":    0.70,
    "Midfielder": 0.35,
    "Defender":   0.10,
    "Goalkeeper": 0.0667,   # unused -- goalkeepers get gk_score, see below
    "Unknown":    0.40,
}
POSITION_DEFENSE_WEIGHT = {
    "Forward":    0.10,
    "Midfielder": 0.30,
    "Defender":   0.75,
    "Goalkeeper": 0.9333,   # unused -- goalkeepers get gk_score, see below
    "Unknown":    0.35,
}
POSITION_CREATION_WEIGHT = {
    "Forward":    0.20,
    "Midfielder": 0.35,
    "Defender":   0.15,
    "Goalkeeper": 0.0,      # unused -- goalkeepers get gk_score, see below
    "Unknown":    0.25,
}

# Stats that drive each composite score
OFFENSIVE_STATS = ["xg_per_match", "goals_per_match",
                    "assists_per_match", "big_chance_miss_rate"]
# Rebuilt 2026-07-01: tackles_per_match/pressures_per_match were dropped
# entirely. Investigated why defense/midfielder ratings were badly broken
# (legendary center-backs like Sergio Ramos/Van Dijk/Piqué ranking in the
# hundreds out of ~2200 defenders) and confirmed tackles/pressures are
# workrate (VOLUME) stats that are NEGATIVELY correlated with real team
# quality at both team and individual level -- a player under constant
# defensive siege racks up more tackles/presses out of necessity, while a
# player on a dominant team barely needs to. Replaced with QUALITY signals
# newly extracted from raw StatsBomb events (duel win/loss outcome,
# "Dribbled Past", Clearance, Block -- see statsbomb.py adapter and
# scripts/enrich_player_profiles_with_defense_creation.py): a good defender
# wins a high share of duels and rarely gets dribbled past regardless of how
# much of the ball their team's dominance affords them. interceptions_per_match
# is kept (a "reading the game" stat, less volume-driven than tackles/presses).
DEFENSIVE_STATS = ["duel_win_rate", "dribbled_past_per_match", "clearances_per_match",
                    "blocks_per_match", "interceptions_per_match",
                    "fouls_per_match", "yellow_cards_per_match"]
# New "creation" axis: chance creation through passing, not goals/assists
# (already captured by OFFENSIVE_STATS) or defending. key_passes_per_match is
# the primary signal; pass_completion is a secondary quality check so a
# highly creative-but-careless passer isn't over-rewarded relative to someone
# equally creative who also protects the ball.
CREATION_STATS = ["key_passes_per_match", "pass_completion"]
# These 3 new-stat groups (defense-quality + creation) only exist for players
# with StatsBomb Big5-league event coverage (~4987/9885 players) -- a
# narrower, more precise sample than the "matches" column, which can include
# Cup/Champions League/international appearances with no per-match event
# breakdown. Shrunk toward neutral 50 by this dedicated credibility instead
# of the general one so thin/zero Big5 coverage doesn't fake confidence.
DEFENSE_CREATION_STATS = {
    "duel_win_rate", "dribbled_past_per_match", "clearances_per_match", "blocks_per_match",
    "interceptions_per_match", "key_passes_per_match", "pass_completion",
}
# duel_win_rate gets its OWN (larger) shrinkage constant: it's a win rate
# over a few dozen discrete duels per match (not a smooth per-match count
# like goals/passes), so it's far noisier at low sample sizes -- confirmed
# 2026-07-01 by comparing several 23-28 Big5-match defenders whose duel_win_rate
# (~0.75-0.76, boosted by small-sample variance) outscored Ramos/Piqué/Puyol
# (~0.61-0.65 over 86-334 matches) even after the standard 12-match shrinkage.
DUEL_SHRINKAGE_MATCHES = 60.0
# GK rating: save_pct, goals-conceded/match, clean-sheet rate from
# data/processed/goalkeeper_match_stats.csv (real match data, merged in at
# fit() time — see _build_gk_scores). Keepers with no rows in that file
# (rare) fall back to the neutral 50.0 placeholder.
GK_STATS = ["save_pct", "ga_per_match", "cs_rate"]
DEFAULT_GK_STATS_PATH = ROOT / "data/processed/goalkeeper_match_stats.csv"

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
# Defensive QUALITY anchors (added 2026-07-01, see DEFENSIVE_STATS docstring
# above). Calibrated from established (20+ Big5 matches) players across all
# outfield positions plus real reference defenders (Van Dijk, Piqué, Puyol,
# Ramos) -- see scripts/enrich_player_profiles_with_defense_creation.py for
# the source columns.
ANCHOR_DUEL_WIN_RATE = [
    (0.35, 30), (0.45, 40), (0.55, 50), (0.60, 55),
    (0.65, 62), (0.70, 72), (0.75, 82), (0.85, 92),
]
# Fewer times dribbled past = better.
ANCHOR_DRIBBLED_PAST_PER_MATCH = [
    (0.0, 88), (0.15, 82), (0.35, 75), (0.7, 62),
    (1.0, 55), (1.3, 48), (1.8, 38), (2.5, 30),
]
ANCHOR_CLEARANCES_PER_MATCH = [
    (0.0, 35), (0.5, 45), (1.5, 55), (3.0, 65),
    (4.5, 75), (6.0, 83), (7.5, 90), (9.0, 95),
]
ANCHOR_BLOCKS_PER_MATCH = [
    (0.0, 35), (0.5, 45), (1.0, 53), (1.5, 60),
    (2.0, 68), (2.5, 76), (3.0, 83), (4.0, 90),
]
ANCHOR_INTERCEPTIONS_PER_MATCH = [
    (0.0, 35), (0.3, 46), (0.6, 54), (0.9, 61),
    (1.3, 69), (1.7, 77), (2.2, 85), (3.0, 92),
]
# Creation anchors -- key_passes_per_match is the primary chance-creation
# signal; calibrated so real elite creators (Xavi 1.93, Modric 1.70, Kroos
# 1.57, De Bruyne 2.76 key passes/match) land 80-96 while destroyer-type
# midfielders who don't create (~0.5-1.0) stay mid-range.
ANCHOR_KEY_PASSES_PER_MATCH = [
    (0.0, 35), (0.2, 45), (0.45, 52), (0.65, 58),
    (0.9, 65), (1.2, 73), (1.6, 82), (2.2, 92), (2.8, 96),
]
ANCHOR_PASS_COMPLETION = [
    (0.55, 35), (0.65, 45), (0.72, 52), (0.78, 58),
    (0.83, 68), (0.87, 78), (0.90, 85), (0.93, 92),
]
# Fewer fouls = better.
ANCHOR_FOULS_PER_MATCH = [
    (0.0, 70), (0.5, 65), (0.937, 60), (1.17, 56), (1.4, 50), (2.18, 35),
]
# Fewer cards = better.
ANCHOR_YELLOW_CARDS_PER_MATCH = [
    (0.0, 68), (0.06, 63), (0.121, 60), (0.185, 55), (0.24, 50), (0.47, 35),
]

# Goalkeeper anchors — calibrated from data/processed/goalkeeper_match_stats.csv
# (real saves/shots-on-target-against/goals-against per match), same method as
# the outfield curves above: percentiles of established keepers (20+ matches)
# plus a real reference point (peak Jan Oblak: save_pct 0.81, ~0.74 goals
# conceded/match, 50% clean sheets — a top-handful keeper, not a max outlier,
# so the curve's own top end is left a little above him).
ANCHOR_SAVE_PCT = [
    (0.55, 35), (0.618, 45), (0.71, 58), (0.792, 78), (0.817, 85), (0.96, 92),
]
# Lower goals-conceded/match = better.
ANCHOR_GA_PER_MATCH = [
    (0.33, 90), (0.66, 82), (1.27, 58), (1.91, 38), (2.13, 32), (2.82, 20),
]
ANCHOR_CS_RATE = [
    (0.0, 35), (0.086, 42), (0.30, 58), (0.487, 75), (0.536, 80), (0.727, 88),
]
GK_SCORE_WEIGHTS = {"save_pct": 0.5, "ga_per_match": 0.3, "cs_rate": 0.2}

ANCHOR_CURVES: dict[str, list[tuple[float, float]]] = {
    "goals_per_match": ANCHOR_GOALS_PER_MATCH,
    "assists_per_match": ANCHOR_ASSISTS_PER_MATCH,
    "xg_per_match": ANCHOR_XG_PER_MATCH,
    "big_chance_miss_rate": ANCHOR_BIG_CHANCE_MISS_RATE,
    "tackles_per_match": ANCHOR_TACKLES_PER_MATCH,
    "pressures_per_match": ANCHOR_PRESSURES_PER_MATCH,
    "fouls_per_match": ANCHOR_FOULS_PER_MATCH,
    "yellow_cards_per_match": ANCHOR_YELLOW_CARDS_PER_MATCH,
    "duel_win_rate": ANCHOR_DUEL_WIN_RATE,
    "dribbled_past_per_match": ANCHOR_DRIBBLED_PAST_PER_MATCH,
    "clearances_per_match": ANCHOR_CLEARANCES_PER_MATCH,
    "blocks_per_match": ANCHOR_BLOCKS_PER_MATCH,
    "interceptions_per_match": ANCHOR_INTERCEPTIONS_PER_MATCH,
    "key_passes_per_match": ANCHOR_KEY_PASSES_PER_MATCH,
    "pass_completion": ANCHOR_PASS_COMPLETION,
}

# Competitions to exclude entirely from the rating pool. Men's and women's
# football have different physical/statistical baselines (event rates aren't
# directly comparable), and StatsBomb open data mixes both into the same
# player pool — without this filter, percentile ranks and the global overall
# ranking below would compare them as if they were one population.
WOMENS_COMPETITION_MARKERS = ("women", "frauen", "liga f", "nwsl", "féminine", "feminine")


def _build_gk_scores(gk_stats_path: Path | None = None) -> dict[str, float]:
    """Real goalkeeper rating from data/processed/goalkeeper_match_stats.csv:
    save_pct + goals-conceded/match (inverted) + clean-sheet rate, each
    mapped through its ANCHOR_* curve, shrunk toward 50 by match-count
    credibility (same SHRINKAGE_MATCHES pattern as the outfield stats),
    then blended via GK_SCORE_WEIGHTS. Returns {lowercased_player: gk_score}
    — goalkeeper_match_stats.csv's player names are all-lowercase while
    player_profiles_with_positions.csv's are Title Case, so callers must
    join on a lowercased key too (df["player"].str.lower()), not the raw
    name. Callers should default to 50.0 for players not in the returned
    dict (keepers with zero rows in the source file)."""
    path = gk_stats_path or DEFAULT_GK_STATS_PATH
    if not path.exists():
        return {}

    gk = pd.read_csv(path)
    gk["player"] = gk["player"].str.lower()
    for c in ("saves", "shots_on_target_against", "goals_against"):
        gk[c] = pd.to_numeric(gk[c], errors="coerce").fillna(0.0)
    gk["clean_sheet"] = (gk["goals_against"] == 0).astype(int)

    agg = gk.groupby("player").agg(
        matches=("match_id", "nunique"),
        saves=("saves", "sum"), sot_against=("shots_on_target_against", "sum"),
        goals_against=("goals_against", "sum"), clean_sheets=("clean_sheet", "sum"),
    ).reset_index()
    matches_safe = agg["matches"].clip(lower=1)
    # Saves can exceed SOT-against for a match where SOT-against wasn't
    # reliably derivable from raw events (a known data-quality quirk in this
    # source) — clamp before scoring rather than let one bad match push
    # save_pct over 1.0.
    agg["save_pct"] = (agg["saves"] / agg["sot_against"].clip(lower=1)).clip(upper=1.0)
    agg["ga_per_match"] = agg["goals_against"] / matches_safe
    agg["cs_rate"] = agg["clean_sheets"] / matches_safe

    credibility = agg["matches"] / (agg["matches"] + SHRINKAGE_MATCHES)
    scores = pd.DataFrame({"player": agg["player"]})
    for stat, anchors in (
        ("save_pct", ANCHOR_SAVE_PCT), ("ga_per_match", ANCHOR_GA_PER_MATCH), ("cs_rate", ANCHOR_CS_RATE),
    ):
        xs, ys = zip(*anchors)
        raw_score = np.interp(agg[stat], xs, ys)
        scores[stat] = 50.0 + (raw_score - 50.0) * credibility

    blend = sum(scores[stat] * w for stat, w in GK_SCORE_WEIGHTS.items())
    gk_score = blend.clip(0, 100)
    return dict(zip(scores["player"], gk_score))


@dataclass
class PlayerStrengthProfile:
    player: str
    team: str
    competition: str
    position: str
    matches: int
    offensive_strength: float = 0.0   # 0-100 absolute (anchor-curve) score
    defensive_strength: float = 0.0
    creation_strength: float = 0.0
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
        # duel_win_rate/pass_completion aren't "_per_match" suffixed, and their
        # neutral fallback isn't 0.0 (see enrich_player_profiles_with_defense_creation.py).
        if "duel_win_rate" in df.columns:
            df["duel_win_rate"] = pd.to_numeric(df["duel_win_rate"], errors="coerce").fillna(0.5)
        if "pass_completion" in df.columns:
            df["pass_completion"] = pd.to_numeric(df["pass_completion"], errors="coerce").fillna(0.75)
        if "defense_creation_matches" in df.columns:
            df["defense_creation_matches"] = pd.to_numeric(
                df["defense_creation_matches"], errors="coerce").fillna(0.0)
        else:
            df["defense_creation_matches"] = 0.0

        # Goals and assists are weighted equally (both are a finished chance
        # for the team); xg_per_match (real shot-level StatsBomb xG, see
        # scripts/enrich_player_profiles_with_xg.py) replaces raw shot/SOT
        # volume since it already captures shot quality and quantity
        # together. All weights are positive: each stat's ANCHOR_CURVES entry
        # is already oriented so a higher score is always better (miss rate
        # and discipline stats are pre-inverted).
        off_weights = {"goals_per_match": 0.30, "assists_per_match": 0.30,
                        "xg_per_match": 0.25, "big_chance_miss_rate": 0.15}
        # duel_win_rate carries most of the weight: confirmed (2026-07-01,
        # 99-team-season regression) it's the ONLY defensive stat here that's
        # POSITIVELY correlated with real team quality (+0.46 with real
        # defense_param) -- clearances/blocks are still volume/possession-
        # confounded (-0.46/-0.56, same flaw as the dropped tackles/pressures,
        # just measured deeper in the defensive third), so they're kept at a
        # small weight rather than dropped outright (still a real signal for
        # last-ditch defending, just a noisier one).
        def_weights = {"duel_win_rate": 0.50, "dribbled_past_per_match": 0.20,
                        "interceptions_per_match": 0.12,
                        "clearances_per_match": 0.08, "blocks_per_match": 0.05,
                        "fouls_per_match": 0.03, "yellow_cards_per_match": 0.02}
        creation_weights = {"key_passes_per_match": 0.65, "pass_completion": 0.35}

        # Map every stat through its fixed anchor curve (absolute, not
        # ranked against the rest of the pool) via np.interp — vectorised,
        # values outside the anchor range just clamp to the nearest end.
        score_cols: dict[str, str] = {}
        for stat in list(off_weights) + list(def_weights) + list(creation_weights):
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
        # Defense-quality/creation stats use their OWN credibility based on
        # defense_creation_matches (the true Big5-scoped sample behind those
        # columns), not the general "matches" count, which can include
        # Cup/Champions League/international appearances with no per-match
        # breakdown -- reusing the general count would fake confidence for
        # players whose real coverage of these specific stats is much thinner.
        credibility = df["matches"] / (df["matches"] + SHRINKAGE_MATCHES)
        credibility_dq = df["defense_creation_matches"] / (df["defense_creation_matches"] + SHRINKAGE_MATCHES)
        credibility_duel = df["defense_creation_matches"] / (df["defense_creation_matches"] + DUEL_SHRINKAGE_MATCHES)
        for stat, score_col in score_cols.items():
            if stat == "duel_win_rate":
                cred = credibility_duel
            elif stat in DEFENSE_CREATION_STATS:
                cred = credibility_dq
            else:
                cred = credibility
            df[score_col] = 50.0 + (df[score_col] - 50.0) * cred

        off_num = sum(df[score_cols[s]] * w for s, w in off_weights.items() if s in score_cols)
        off_den = sum(w for s, w in off_weights.items() if s in score_cols)
        df["off_score"] = (off_num / off_den).clip(0, 100) if off_den > 0 else 50.0

        def_num = sum(df[score_cols[s]] * w for s, w in def_weights.items() if s in score_cols)
        def_den = sum(w for s, w in def_weights.items() if s in score_cols)
        df["def_score"] = (def_num / def_den).clip(0, 100) if def_den > 0 else 50.0

        creation_num = sum(df[score_cols[s]] * w for s, w in creation_weights.items() if s in score_cols)
        creation_den = sum(w for s, w in creation_weights.items() if s in score_cols)
        df["creation_score"] = (creation_num / creation_den).clip(0, 100) if creation_den > 0 else 50.0

        # overall = position-weighted blend of the three absolute scores. No
        # percentile re-rank, no curve — the anchor tables already calibrate
        # the 0-100 range (median established player ~58-60, genuine legends
        # ~90-95), so the blend doesn't need further reshaping.
        atk_w_s = df["position_group"].map(POSITION_ATTACK_WEIGHT).fillna(0.40)
        def_w_s = df["position_group"].map(POSITION_DEFENSE_WEIGHT).fillna(0.35)
        cre_w_s = df["position_group"].map(POSITION_CREATION_WEIGHT).fillna(0.25)
        w_sum_s = atk_w_s + def_w_s + cre_w_s
        df["overall"] = (
            (atk_w_s / w_sum_s) * df["off_score"]
            + (def_w_s / w_sum_s) * df["def_score"]
            + (cre_w_s / w_sum_s) * df["creation_score"]
        ).clip(0, 100)

        # Goalkeepers are a special case: the tackles/pressures anchor curve
        # is calibrated on outfield defenders, and keepers naturally record
        # almost none of either, so the off/def blend above crushes every
        # keeper to ~43-44 regardless of how good they actually are — a
        # curve mismatch, not a real signal. Real save%/goals-conceded/
        # clean-sheet data is merged in below (_build_gk_scores) and used
        # for both gk_strength and overall for goalkeepers; only keepers
        # with zero rows in goalkeeper_match_stats.csv fall back to 50.0.
        gk_scores = _build_gk_scores()
        df["gk_score"] = df["player"].str.lower().map(gk_scores).fillna(50.0)
        is_gk = df["position_group"] == "Goalkeeper"
        df.loc[is_gk, "overall"] = df.loc[is_gk, "gk_score"]
        # Also drives the "Defensa" display slot for keepers — showing the
        # outfield-crushed def_score (~43) next to a real overall (~80+)
        # would look inconsistent in the UI.
        df.loc[is_gk, "def_score"] = df.loc[is_gk, "gk_score"]

        for _, row in df.iterrows():
            pos = str(row.get("position_group", "Unknown"))
            player = str(row.get("player", ""))
            matches = int(row.get("matches", 0))
            off_score = float(row["off_score"])
            def_score = float(row["def_score"])
            creation_score = float(row["creation_score"])
            overall = float(row["overall"])
            gk_score = float(row["gk_score"])

            self.profiles_[player] = PlayerStrengthProfile(
                player=player,
                team=str(row.get("team_c", "")),
                competition=str(row.get("competition_c", "")),
                position=pos,
                matches=matches,
                offensive_strength=round(off_score, 1),
                defensive_strength=round(def_score, 1),
                creation_strength=round(creation_score, 1),
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
            atk_w = POSITION_ATTACK_WEIGHT.get(p.position, 0.40)
            def_w = POSITION_DEFENSE_WEIGHT.get(p.position, 0.35)
            # Creation feeds attack_idx too (AttackDefenseModel has no
            # separate "creation" parameter to bridge onto) -- a squad of
            # good creators should still push a higher attack_index even if
            # their own goals/xG are modest.
            cre_w = POSITION_CREATION_WEIGHT.get(p.position, 0.25)
            total_atk += p.offensive_strength * atk_w + p.creation_strength * cre_w
            total_def += p.defensive_strength * def_w
            total_w_atk += atk_w + cre_w
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
                "defensive": p.defensive_strength, "creation": p.creation_strength,
                "xg_pm": p.xg_per_match, "goals_pm": p.goals_per_match,
                "assists_pm": p.assists_per_match, "shots_pm": p.shots_per_match,
                "tackles_pm": p.tackles_per_match,
            })
        return pd.DataFrame(rows).sort_values("overall", ascending=False)
