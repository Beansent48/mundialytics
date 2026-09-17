# v0.21 Identity Resolution & Audit Hardening

This iteration fixes the main v0.20 player-prop issue: manual current-lineup names such as `Álvaro Morata`, `Federico Valverde` or `Salem Al-Dawsari` did not always match StatsBomb-style historical full names such as `alvaro borja morata martin`, `federico santiago valverde dipetta` or `salem mohammed al dawsari`.

## What changed

- Added `src/mundialytics/identity/normalization.py`.
- Added `src/mundialytics/identity/player_resolver.py`.
- Added `data/identity/player_aliases.csv`.
- Player props now resolve current candidates to canonical historical identities.
- Player profiles are aggregated across valid historical teams for the same resolved player.
- Output now includes:
  - `player_input_name`
  - `canonical_player_name`
  - `current_team`
  - `historical_teams_used`
  - `identity_match_level`
  - `identity_status`
  - `identity_confidence`
- Betting recommendations are blocked for unresolved/ambiguous identities and sample-size-zero player props.
- The audit report now includes player identity diagnostics and demo-odds warnings.

## Important product rule

Historical teams are allowed for player profiling, but they do not create inference candidates. A player is predicted only if present in `current_lineups.csv` or `squads.csv` for the current match.

## Remaining experimental points

- Team-history weighting is still simple career aggregation. A future v0.22 should add explicit recency/team-context weights.
- Manual aliases should grow as real fixture/lineup providers are connected.
- Provider IDs are still better than name matching and should become the primary key when available.
