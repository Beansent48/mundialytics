"""Real goalkeeper save quality, derived from match-level event data.

data/processed/goalkeeper_match_stats.csv (saves, shots_on_target_against,
goals_against per match) already exists but was never merged into
PlayerStrengthModel — gk_strength/overall there is still a 50.0 placeholder
(see [[project_player_rating_data]] memory). This module does NOT change
that rating; it computes a narrowly-scoped save_pct used only during match
resolution to convert a squad's shots-on-target-faced into goals-conceded
stochastically, instead of ignoring goalkeeper quality entirely.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]

# Same credibility = matches / (matches + SHRINKAGE_MATCHES) pattern as
# player_strength.py — reuse the constant rather than re-deriving one.
from mundialytics.statistical_core.player_strength import SHRINKAGE_MATCHES

DEFAULT_GK_STATS_PATH = ROOT / "data/processed/goalkeeper_match_stats.csv"


@dataclass
class GoalkeeperQuality:
    player: str
    save_pct: float   # shrunk toward the league-average prior
    matches: int


def build_goalkeeper_quality(
    gk_stats_path: Path | str = DEFAULT_GK_STATS_PATH,
    shrinkage_matches: float = SHRINKAGE_MATCHES,
) -> dict[str, GoalkeeperQuality]:
    """Group goalkeeper_match_stats.csv by player -> shrunk save_pct.

    save_pct = saves / shots_on_target_against, aggregated across all of a
    keeper's matches (not averaged per-match, so high-SOT matches count
    proportionally more), then regressed toward the league-wide average
    save_pct based on sample size.
    """
    path = Path(gk_stats_path)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    for col in ("saves", "shots_on_target_against"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    league_saves = df["saves"].sum()
    league_sot = df["shots_on_target_against"].sum()
    league_avg_save_pct = float(league_saves / league_sot) if league_sot > 0 else 0.65

    agg = df.groupby("player").agg(
        total_saves=("saves", "sum"),
        total_sot=("shots_on_target_against", "sum"),
        matches=("match_id", "nunique"),
    ).reset_index()

    out: dict[str, GoalkeeperQuality] = {}
    for _, row in agg.iterrows():
        matches = int(row["matches"])
        raw_pct = float(row["total_saves"] / row["total_sot"]) if row["total_sot"] > 0 else league_avg_save_pct
        # Some matches have shots_on_target_against=0 while saves>0 (a known
        # data-quality artifact flagged in the source CSV's data_quality_flag
        # column — SOT-against isn't always reliably derivable from raw
        # events). Saves can't physically exceed SOT faced, so clamp before
        # shrinking rather than letting one bad match push save_pct over 1.0.
        raw_pct = min(raw_pct, 1.0)
        credibility = matches / (matches + shrinkage_matches)
        shrunk_pct = league_avg_save_pct + (raw_pct - league_avg_save_pct) * credibility
        out[str(row["player"])] = GoalkeeperQuality(
            player=str(row["player"]),
            save_pct=float(max(0.0, min(1.0, shrunk_pct))),
            matches=matches,
        )
    return out
