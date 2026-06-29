from __future__ import annotations

import pandas as pd

from mundialytics.identity.normalization import add_player_identity_columns, add_team_identity_columns, canonical_player_name
from mundialytics.models.player_event_model import PlayerEventModel, PlayerEventPrediction


class SubstitutePlusModel:
    """Apply Betfair/betting 'Substitute+' style adjustment.

    Assumption for threshold markets: selection wins if original player OR the
    relevant substitute reaches the threshold. Always check the bookmaker's exact
    rules for the market before using it outside paper mode.
    """

    def __init__(self, player_model: PlayerEventModel, lineups: pd.DataFrame | None = None):
        self.player_model = player_model
        if lineups is not None:
            normalized = add_team_identity_columns(lineups)
            normalized = add_player_identity_columns(normalized)
            if "replaced_by" in normalized.columns:
                normalized["replaced_by"] = normalized["replaced_by"].map(
                    lambda x: None if pd.isna(x) or str(x).strip() == "" else canonical_player_name(x)
                )
            self.lineups = normalized
        else:
            self.lineups = None

    def likely_replacement(self, match_id: int, player: str) -> tuple[str | None, float, float | None]:
        if self.lineups is None:
            return None, 0.0, None
        player = canonical_player_name(player)
        row = self.lineups[(self.lineups["match_id"] == match_id) & (self.lineups["player"] == player)]
        if row.empty:
            return None, 0.0, None
        r = row.iloc[0]
        replacement = r.get("replaced_by") if pd.notna(r.get("replaced_by")) else None
        if not replacement:
            return None, 0.0, None
        minute = float(r.get("replacement_minute")) if pd.notna(r.get("replacement_minute")) else 70.0
        p_enter = 0.80 if minute < 85 else 0.55
        return str(replacement), p_enter, minute

    def apply(
        self,
        original_pred: PlayerEventPrediction,
        match_id: int,
        team_context: dict | None = None,
        min_replacement_probability: float = 0.10,
    ) -> dict:
        replacement, p_enter, minute = self.likely_replacement(match_id, original_pred.player)
        if replacement is None or p_enter < min_replacement_probability:
            return {
                "probability_substitute_plus": original_pred.probability,
                "replacement": None,
                "replacement_probability": 0.0,
                "replacement_note": "No projected replacement found; using original probability.",
            }
        replacement_minutes = max(95 - (minute or 70), 1)
        repl_pred = self.player_model.predict_market(
            player=replacement,
            market_type=original_pred.market_type,
            line=original_pred.line,
            expected_minutes=replacement_minutes,
            team_context=team_context,
        )
        # OR combination, conditional on original not already winning.
        p_final = original_pred.probability + (1 - original_pred.probability) * p_enter * repl_pred.probability
        return {
            "probability_substitute_plus": float(min(max(p_final, original_pred.probability), 1.0)),
            "replacement": replacement,
            "replacement_probability": repl_pred.probability,
            "replacement_note": f"Adjusted with {replacement}: p_enter={p_enter:.0%}, replacement P={repl_pred.probability:.1%}.",
        }
