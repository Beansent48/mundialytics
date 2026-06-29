from __future__ import annotations

SUPPORTED_MARKETS = {
    "match_winner": "1X2 match winner outcome",
    "over_under_goals": "Total goals threshold",
    "both_teams_to_score": "BTTS",
    "player_shots": "Player shots threshold",
    "player_shots_on_target": "Player shots on target threshold",
    "player_fouls_committed": "Player fouls committed threshold",
    "player_fouls_drawn": "Player fouls drawn threshold",
    "player_yellow_card": "Player card threshold",
    "player_goals": "Player goals threshold",
    "player_assists": "Player assists threshold",
}

PLAYER_MARKETS = {m for m in SUPPORTED_MARKETS if m.startswith("player_")}


def is_player_market(market_type: str) -> bool:
    return market_type in PLAYER_MARKETS


def validate_market(market_type: str) -> None:
    if market_type not in SUPPORTED_MARKETS:
        raise ValueError(f"Unsupported market '{market_type}'. Supported: {sorted(SUPPORTED_MARKETS)}")
