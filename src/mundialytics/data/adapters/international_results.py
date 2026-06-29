from __future__ import annotations

from pathlib import Path

import pandas as pd

from mundialytics.data.schema import normalize_matches


def international_results_to_matches(path: str | Path) -> pd.DataFrame:
    """Convert martj42/international_results CSV into canonical matches.

    Expected columns: date, home_team, away_team, home_score, away_score,
    tournament, country, neutral.
    """
    raw = pd.read_csv(path)
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"International-results CSV missing columns: {sorted(missing)}")
    out = pd.DataFrame({
        "match_id": [f"intl_{i:07d}" for i in range(len(raw))],
        "date": raw["date"],
        "home_team": raw["home_team"],
        "away_team": raw["away_team"],
        "home_goals": raw["home_score"],
        "away_goals": raw["away_score"],
        "neutral": raw.get("neutral", 0).astype(int) if "neutral" in raw.columns else 0,
        "competition": raw.get("tournament", "International"),
        "season": pd.to_datetime(raw["date"], errors="coerce").dt.year.astype("Int64").astype(str),
        "stage": raw.get("tournament", "International"),
        "team_scope": "national",
        "source": "martj42/international_results",
    })
    return normalize_matches(out.dropna(subset=["home_goals", "away_goals"]))
