from __future__ import annotations

import pandas as pd

from mundialytics.identity.normalization import add_player_identity_columns, add_team_identity_columns, canonical_player_name


class MinutesModel:
    """Simple minutes/titularity estimator.

    A production version would ingest lineups, injuries and manager patterns.
    This version uses recent minutes and the optional projected lineups file.
    """

    def __init__(self):
        self.player_summary: pd.DataFrame | None = None
        self.projected_lineups: pd.DataFrame | None = None

    def fit(self, player_events: pd.DataFrame, projected_lineups: pd.DataFrame | None = None) -> "MinutesModel":
        player_events = add_player_identity_columns(add_team_identity_columns(player_events))
        if projected_lineups is not None:
            pl = add_player_identity_columns(add_team_identity_columns(projected_lineups))
            if "replaced_by" in pl.columns:
                pl["replaced_by"] = pl["replaced_by"].map(
                    lambda x: None if pd.isna(x) or str(x).strip() == "" else canonical_player_name(x)
                )
            projected_lineups = pl
        rows = []
        for player, g in player_events.groupby("player"):
            rows.append({
                "player": player,
                "team": g["team"].iloc[-1],
                "position": g["position"].iloc[-1],
                "expected_minutes": float(g.sort_values("date")["minutes"].tail(5).mean()),
                "start_probability": float((g["minutes"] >= 45).tail(5).mean()),
            })
        self.player_summary = pd.DataFrame(rows)
        self.projected_lineups = projected_lineups
        return self

    def estimate(self, player: str, match_id: int | None = None) -> dict:
        player = canonical_player_name(player)
        if self.projected_lineups is not None and match_id is not None:
            pl = self.projected_lineups[(self.projected_lineups["match_id"] == match_id) & (self.projected_lineups["player"] == player)]
            if not pl.empty:
                r = pl.iloc[0]
                return {
                    "player": player,
                    "start_probability": 0.95 if int(r.get("started", 0)) == 1 else 0.20,
                    "expected_minutes": float(r.get("minutes", 25)),
                    "replaced_by": r.get("replaced_by") if pd.notna(r.get("replaced_by")) else None,
                    "replacement_minute": r.get("replacement_minute") if pd.notna(r.get("replacement_minute")) else None,
                }
        if self.player_summary is not None:
            row = self.player_summary[self.player_summary["player"] == player]
            if not row.empty:
                r = row.iloc[0]
                return {
                    "player": player,
                    "start_probability": float(r["start_probability"]),
                    "expected_minutes": float(r["expected_minutes"]),
                    "replaced_by": None,
                    "replacement_minute": None,
                }
        return {"player": player, "start_probability": 0.50, "expected_minutes": 60.0, "replaced_by": None, "replacement_minute": None}
