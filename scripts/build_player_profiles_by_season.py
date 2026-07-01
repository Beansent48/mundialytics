#!/usr/bin/env python3
"""Build a SEASON-SPLIT player profiles CSV, grouped by
(player, team, competition, season) instead of career-aggregated.

Why from raw StatsBomb event JSON rather than the existing
data/external/advanced/statsbomb/statsbomb_player_match_stats.csv (which
already has match-level granularity + real season labels): audited that
file first and found assists, key_passes, sca, and gca are ALL zero across
every single one of its 111,982 rows -- the same class of bug already found
and fixed once this session (see scripts/fix_player_assists.py) in a
DIFFERENT derived file. Rather than trust a second pre-aggregated file with
its own undocumented bugs, this rebuilds directly from the raw event JSON
via the same statsbomb_events_to_player_events() adapter already proven
correct earlier this session (used for the assists fix and the squad
calibration's old-era match rebuild).

Scope: the 5 Big5 leagues only (matches DRAFT_COMPETITIONS in
app/squadlab_page.py) -- same 5 competition IDs used in
scripts/fit_squad_lambda_calibration.py's old-era calibration rebuild.

Position is carried over from the existing (career-aggregated)
player_profiles_with_positions.csv rather than re-derived from raw
StatsBomb position text -- that file already uses the simple 4-bucket
scheme (Forward/Midfielder/Defender/Goalkeeper) the rest of the codebase
expects, and position rarely changes season to season, so reusing it here
avoids reimplementing a raw-text position classifier from scratch.

Run:
    python scripts/build_player_profiles_by_season.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.adapters.statsbomb import statsbomb_events_to_player_events
from mundialytics.statistical_core.schemas import canonical_name

BIG5_COMP_IDS = {9: "1. Bundesliga", 11: "La Liga", 7: "Ligue 1", 2: "Premier League", 12: "Serie A"}
BIG_CHANCE_XG_THRESHOLD = 0.3

OUT_PATH = ROOT / "data/processed/player_profiles_by_season.csv"
PER_MATCH_OUT_PATH = ROOT / "data/processed/player_match_events.csv"
POSITIONS_PATH = ROOT / "data/processed/player_profiles_with_positions.csv"
XG_SHOTS_PATH = ROOT / "data/external/xg/statsbomb/statsbomb_xg_shots.csv"

STAT_COLS = [
    "shots", "shots_on_target", "fouls_committed", "fouls_drawn", "yellow_cards", "red_cards",
    "goals", "assists", "passes", "complete_passes", "key_passes", "pressures",
    "tackles", "interceptions",
    "duels", "duels_won", "duels_lost", "dribbled_past", "clearances", "blocks",
]


def _discover_matches() -> list[dict]:
    """One row per (match_id, competition, season, date) for the 5 Big5 leagues."""
    rows = []
    for comp_id, comp_name in BIG5_COMP_IDS.items():
        for f in (ROOT / f"data/raw/statsbomb/open-data/data/matches/{comp_id}").glob("*.json"):
            for m in json.loads(f.read_text(encoding="utf-8")):
                rows.append({
                    "match_id": m["match_id"],
                    "date": m["match_date"],
                    "competition": comp_name,
                    "season": m["season"]["season_name"],
                })
    return rows


def main() -> None:
    matches = _discover_matches()
    print(f"Found {len(matches)} Big5 matches to process")

    per_match_rows: list[pd.DataFrame] = []
    events_dir = ROOT / "data/raw/statsbomb/open-data/data/events"
    for i, m in enumerate(matches):
        events_path = events_dir / f"{m['match_id']}.json"
        if not events_path.exists():
            continue
        try:
            pe = statsbomb_events_to_player_events(
                events_path, match_id=m["match_id"], date=m["date"], competition=m["competition"],
            )
        except Exception as exc:
            print(f"  skip match {m['match_id']}: {exc}")
            continue
        if pe.empty:
            continue
        pe["season"] = m["season"]
        per_match_rows.append(pe)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(matches)} matches processed...")

    print(f"Processed {len(per_match_rows)} matches with events")
    all_rows = pd.concat(per_match_rows, ignore_index=True)
    all_rows["team"] = all_rows["team"].map(canonical_name)

    # xG: aggregate statsbomb_xg_shots.csv per (player, provider_match_id),
    # then attach to the matching per-match-player row by (player, match_id).
    shots = pd.read_csv(XG_SHOTS_PATH, low_memory=False)
    shots["xg"] = pd.to_numeric(shots["xg"], errors="coerce").fillna(0.0)
    shots["npxg"] = pd.to_numeric(shots["npxg"], errors="coerce").fillna(0.0)
    shots["is_goal"] = shots["outcome"] == "Goal"
    shots["is_big_chance"] = shots["xg"] >= BIG_CHANCE_XG_THRESHOLD
    shots["is_big_chance_missed"] = shots["is_big_chance"] & ~shots["is_goal"]
    xg_per_match = shots.groupby(["player", "provider_match_id"]).agg(
        xg=("xg", "sum"), npxg=("npxg", "sum"),
        big_chances=("is_big_chance", "sum"), big_chances_missed=("is_big_chance_missed", "sum"),
    ).reset_index().rename(columns={"provider_match_id": "match_id"})
    all_rows = all_rows.merge(xg_per_match, on=["player", "match_id"], how="left")
    for c in ("xg", "npxg", "big_chances", "big_chances_missed"):
        all_rows[c] = all_rows[c].fillna(0.0)

    # Save the per-match-per-player granularity too (not just the season
    # aggregate below) — this is what a future "memorable performance"
    # detector needs (e.g. a 4-goal game), and reprocessing 2169 raw event
    # JSON files just to look that up again would be wasteful.
    all_rows.to_csv(PER_MATCH_OUT_PATH, index=False)
    print(f"Wrote {len(all_rows)} per-match-per-player rows -> {PER_MATCH_OUT_PATH}")

    # Position lookup from the existing career-aggregated file (simple 4-bucket scheme).
    positions = pd.read_csv(POSITIONS_PATH)[["player", "position"]].drop_duplicates("player")
    position_map = dict(zip(positions["player"], positions["position"]))

    print("Aggregating by (player, team, competition, season)...")
    agg_dict = {c: "sum" for c in STAT_COLS}
    agg_dict.update({"xg": "sum", "npxg": "sum", "big_chances": "sum", "big_chances_missed": "sum"})
    grouped = all_rows.groupby(["player", "team", "competition", "season"], as_index=False).agg({
        **agg_dict, "match_id": "nunique",
    }).rename(columns={"match_id": "matches"})

    matches_safe = grouped["matches"].clip(lower=1)
    for c in STAT_COLS:
        grouped[f"{c}_per_match"] = grouped[c] / matches_safe
    grouped["xg_per_match"] = grouped["xg"] / matches_safe
    grouped["npxg_per_match"] = grouped["npxg"] / matches_safe
    grouped["big_chances_missed_per_match"] = grouped["big_chances_missed"] / matches_safe
    grouped["big_chance_miss_rate"] = (
        grouped["big_chances_missed"] / grouped["big_chances"].clip(lower=1)
    ).where(grouped["big_chances"] > 0, 0.0)

    # Duel quality, not volume: outcome is untagged for a large share of
    # duels (off-ball challenges), so the rate is computed over
    # known-outcome duels only, falling back to a neutral 0.5 when a player
    # has zero known-outcome duels rather than a misleading 0.
    known_outcome_duels = grouped["duels_won"] + grouped["duels_lost"]
    grouped["duel_win_rate"] = (
        grouped["duels_won"] / known_outcome_duels.clip(lower=1)
    ).where(known_outcome_duels > 0, 0.5)

    grouped["position"] = grouped["player"].map(position_map).fillna("Unknown")

    # Rename to match the existing career-file's column naming convention
    # (shots_on_target -> sot, fouls_committed -> fouls) so downstream code
    # that already expects those names keeps working.
    grouped = grouped.rename(columns={
        "shots_on_target_per_match": "sot_per_match",
        "fouls_committed_per_match": "fouls_per_match",
    })

    keep_cols = [
        "player", "team", "competition", "season", "position", "matches",
        "goals_per_match", "assists_per_match", "shots_per_match", "sot_per_match",
        "fouls_per_match", "yellow_cards_per_match", "tackles_per_match", "pressures_per_match",
        "key_passes_per_match", "passes_per_match", "complete_passes_per_match",
        "interceptions_per_match", "xg_per_match", "npxg_per_match",
        "big_chances_missed_per_match", "big_chance_miss_rate",
        "duels_per_match", "duel_win_rate", "dribbled_past_per_match",
        "clearances_per_match", "blocks_per_match",
    ]
    grouped = grouped[keep_cols].sort_values(["player", "season"]).reset_index(drop=True)

    grouped.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(grouped)} (player, team, competition, season) rows -> {OUT_PATH}")
    print(f"Unique players: {grouped['player'].nunique()}, unique seasons: {grouped['season'].nunique()}")
    print(grouped["season"].value_counts().sort_index())


if __name__ == "__main__":
    main()
