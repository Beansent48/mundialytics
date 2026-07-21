from __future__ import annotations

"""Aggregate Understat shot-level data into match-level team xG.

Understat exposes xG per *shot* (data/external/advanced/understat/understat_shots.csv,
~473k shots, Big5 2014/15-2025/26). The match/competition predictor wants xG per
match per team. This script performs that aggregation and writes:

1. A canonical match-level xG file in the exact schema the existing xG enrichment
   path consumes (mundialytics.enrichment.xg.CANONICAL_XG_COLUMNS), so
   scripts/enrich_matches_with_xg.py can join it onto the modeling foundation.
   Team names are kept as Understat names on purpose -- the team_registry maps
   understat_name -> football_data_name during the join.

2. A richer team-match table (per-team xg/npxg/shots/sot/goals) for later modeling
   (rolling xG-for / xG-against features).

Home/away is derived from the Understat `game` field ("2014-08-16 Arsenal-Crystal
Palace"): strip the date prefix, then for the two teams sharing a game_id, the one
the remaining string starts with is home. This startswith approach is robust to
hyphenated team names (e.g. "Saint-Etienne") that a naive split on "-" would break.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.enrichment.xg import CANONICAL_XG_COLUMNS  # noqa: E402
from mundialytics.enrichment.understat_team_aliases import to_foundation_name  # noqa: E402

DEFAULT_SHOTS = "data/external/advanced/understat/understat_shots.csv"
DEFAULT_OUT_CANONICAL = "data/external/xg/understat/understat_xg_matches.csv"
DEFAULT_OUT_TEAM_MATCH = "data/processed/understat_team_match_xg.csv"

# Understat `situation` value that is a penalty (excluded from non-penalty xG).
PENALTY_SITUATIONS = {"penalty"}
GOAL_RESULTS = {"goal"}  # `result` values counted as a goal (excludes "own goal" scorer credit)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _split_home_away(game: str, date_prefix: str, teams: list[str]) -> tuple[str | None, str | None]:
    """Return (home, away) for the two teams sharing a game string.

    `game` looks like "2014-08-16 Arsenal-Crystal Palace". We strip the leading
    "<date> " and then identify which of the two team names the remaining
    "Home-Away" string starts with.
    """
    if len(teams) != 2:
        return None, None
    teams_str = str(game)
    prefix = f"{date_prefix} "
    if teams_str.startswith(prefix):
        teams_str = teams_str[len(prefix):]
    a, b = teams[0], teams[1]
    a_home = teams_str.startswith(a) and teams_str.endswith(b)
    b_home = teams_str.startswith(b) and teams_str.endswith(a)
    if a_home and not b_home:
        return a, b
    if b_home and not a_home:
        return b, a
    # Ambiguous (prefix collision / unexpected formatting) -> fall back to a
    # plain split on the last hyphen boundary that separates the two known teams.
    for home, away in ((a, b), (b, a)):
        if teams_str == f"{home}-{away}":
            return home, away
    return None, None


def build_understat_xg_matches(shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = shots.copy()
    df["xg"] = pd.to_numeric(df["xg"], errors="coerce")
    df["date10"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")
    situation = df["situation"].astype(str).str.strip().str.lower()
    result = df["result"].astype(str).str.strip().str.lower()
    df["is_penalty"] = situation.isin(PENALTY_SITUATIONS)
    df["is_goal"] = result.isin(GOAL_RESULTS)
    df["npxg"] = df["xg"].where(~df["is_penalty"], 0.0)

    # Per (game_id, team) aggregation.
    grouped = df.groupby(["game_id", "team"], sort=False)
    team_match = grouped.agg(
        league=("league", "first"),
        season=("season", "first"),
        game=("game", "first"),
        date=("date10", "first"),
        team_xg=("xg", "sum"),
        team_npxg=("npxg", "sum"),
        team_shots=("shot_id", "count"),
        team_goals=("is_goal", "sum"),
    ).reset_index()

    # Derive home/away per game from the two participating teams.
    rows: list[dict[str, Any]] = []
    for game_id, g in team_match.groupby("game_id", sort=False):
        teams = g["team"].tolist()
        game_str = g["game"].iloc[0]
        date_prefix = str(g["date"].iloc[0])
        home, away = _split_home_away(game_str, date_prefix, teams)
        if home is None:
            continue
        by_team = g.set_index("team")
        h, a = by_team.loc[home], by_team.loc[away]
        rows.append({
            "provider": "understat",
            "provider_match_id": game_id,
            "date": g["date"].iloc[0],
            "competition": g["league"].iloc[0],
            # Season dtype is normalized to str: a targeted re-download returned
            # season_id as string ("1516") while the original bulk file had int.
            "season": str(g["season"].iloc[0]),
            "home_team": home,
            "away_team": away,
            # Foundation (football-data) convention names, for a registry-free join.
            "home_team_fd": to_foundation_name(home),
            "away_team_fd": to_foundation_name(away),
            "home_xg": round(float(h["team_xg"]), 4),
            "away_xg": round(float(a["team_xg"]), 4),
            "home_npxg": round(float(h["team_npxg"]), 4),
            "away_npxg": round(float(a["team_npxg"]), 4),
            "home_shots": int(h["team_shots"]),
            "away_shots": int(a["team_shots"]),
            "home_goals": int(h["team_goals"]),
            "away_goals": int(a["team_goals"]),
            "xg_match_confidence": "understat_shot_aggregation",
        })

    matches = pd.DataFrame(rows)
    # Canonical file carries foundation-convention names so the xG enrichment join
    # matches on plain normalized names without depending on the team_registry.
    canonical = matches.copy()
    canonical["home_team"] = canonical["home_team_fd"]
    canonical["away_team"] = canonical["away_team_fd"]
    canonical = canonical.reindex(columns=CANONICAL_XG_COLUMNS)
    return canonical, matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", default=DEFAULT_SHOTS)
    parser.add_argument("--out-canonical", default=DEFAULT_OUT_CANONICAL)
    parser.add_argument("--out-team-match", default=DEFAULT_OUT_TEAM_MATCH)
    args = parser.parse_args()

    shots_path = _resolve(args.shots)
    shots = pd.read_csv(shots_path)
    canonical, team_match = build_understat_xg_matches(shots)

    out_canonical = _resolve(args.out_canonical)
    out_team_match = _resolve(args.out_team_match)
    out_canonical.parent.mkdir(parents=True, exist_ok=True)
    out_team_match.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(out_canonical, index=False)
    team_match.to_csv(out_team_match, index=False)

    n_games_in = shots["game_id"].nunique()
    summary = {
        "shots_in": int(len(shots)),
        "distinct_games_in": int(n_games_in),
        "matches_out": int(len(team_match)),
        "matches_dropped_home_away_unresolved": int(n_games_in - len(team_match)),
        "seasons": sorted(shots["season"].astype(str).unique().tolist()),
        "leagues": sorted(shots["league"].astype(str).unique().tolist()),
        "total_home_xg_mean": round(float(team_match["home_xg"].mean()), 3) if len(team_match) else None,
        "total_away_xg_mean": round(float(team_match["away_xg"].mean()), 3) if len(team_match) else None,
        "outputs": {"canonical": str(out_canonical), "team_match": str(out_team_match)},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
