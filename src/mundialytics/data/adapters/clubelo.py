from __future__ import annotations

from pathlib import Path

import pandas as pd

from mundialytics.data.identity import canonical_team_name, team_id


def clubelo_to_ratings(path: str | Path) -> pd.DataFrame:
    """Normalize ClubElo CSV/API output.

    Common ClubElo columns include Club, Elo, Rank, Country, Level, From, To.
    """
    raw = pd.read_csv(path)
    club_col = "Club" if "Club" in raw.columns else "club"
    elo_col = "Elo" if "Elo" in raw.columns else "elo"
    if club_col not in raw.columns or elo_col not in raw.columns:
        raise ValueError("ClubElo CSV must include Club and Elo columns.")
    out = pd.DataFrame({
        "team": raw[club_col].map(canonical_team_name),
        "team_id": [team_id(x, "club") for x in raw[club_col]],
        "elo": pd.to_numeric(raw[elo_col], errors="coerce"),
        "country": raw.get("Country", raw.get("country", None)),
        "rank": raw.get("Rank", raw.get("rank", None)),
        "source": "clubelo",
    })
    return out.dropna(subset=["elo"]).reset_index(drop=True)
