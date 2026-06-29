from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mundialytics.data.schema import normalize_matches


def openfootball_json_to_matches(path: str | Path, *, competition: str = "openfootball", season: str = "unknown", team_scope: str = "club") -> pd.DataFrame:
    """Convert common OpenFootball JSON structures to canonical matches.

    Supports files with a top-level `rounds` array containing `matches`.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    rounds = data.get("rounds", []) if isinstance(data, dict) else []
    for ridx, rnd in enumerate(rounds):
        stage = rnd.get("name", f"Round {ridx + 1}")
        for midx, m in enumerate(rnd.get("matches", [])):
            score = m.get("score", {}) or {}
            ft = score.get("ft", score) if isinstance(score, dict) else {}
            team1 = m.get("team1") or m.get("home_team")
            team2 = m.get("team2") or m.get("away_team")
            if not team1 or not team2:
                continue
            rows.append({
                "match_id": f"openfootball_{Path(path).stem}_{ridx:03d}_{midx:03d}",
                "date": m.get("date"),
                "home_team": team1,
                "away_team": team2,
                "home_goals": ft.get("team1") if isinstance(ft, dict) else None,
                "away_goals": ft.get("team2") if isinstance(ft, dict) else None,
                "neutral": int(m.get("neutral", 0) or 0),
                "competition": competition,
                "season": season,
                "stage": stage,
                "team_scope": team_scope,
                "source": "openfootball",
            })
    if not rows:
        raise ValueError(f"No matches found in OpenFootball JSON: {path}")
    return normalize_matches(pd.DataFrame(rows).dropna(subset=["home_goals", "away_goals"]))
