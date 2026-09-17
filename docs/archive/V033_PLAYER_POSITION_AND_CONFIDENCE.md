# v0.33 Player position fallback and input confidence

This release improves player-prop deployment when only broad roster/squad data is available.

## Why
Free roster feeds such as ESPN often expose broad soccer positions only: `G`, `D`, `M`, `F`. Treating all defenders as centre-backs, all midfielders as central midfielders and all forwards as strikers is too crude for player props.

## Behaviour
- Confirmed/provider-specific positions are still used when they are specific.
- If a matched player has only a broad provider position (`D`, `M`, `F`) or unknown position, the model uses the player's most frequent historical StatsBomb position as the deployment position.
- New audit columns are added to player predictions and dynamic lines: `input_position`, `resolved_position`, `position_source`, `player_input_source`, `player_selection_confidence`.
- Squad/roster candidates remain lower confidence than confirmed lineups.
- Goalkeeper attacking-prop guardrails and unresolved-player blocks remain active.

## Expected lineups
Reliable predicted/expected lineups usually require a paid data source or dedicated scraper. This version keeps the free workflow robust and auditable, and is ready to accept better lineup inputs when available.
