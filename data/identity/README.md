# Identity maps

This folder stores provider/canonical identity files. For the free MVP, API-Football is the operational source of truth for current fixtures, lineups and provider player IDs. StatsBomb Open Data remains the historical event source used to train/calibrate props.

Typical generated file:

- `player_identity_map.csv`: maps `api_football:<provider_player_id>` to the historical `player_id_global` used by the player-props model.

Do not hand-edit model IDs blindly. If a provider row does not map cleanly, fix the provider export or add a reviewed alias row and keep `match_status`/`match_confidence` honest.
