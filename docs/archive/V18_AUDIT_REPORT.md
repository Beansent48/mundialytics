# v0.18 Audit Report

## Scope

This audit focused on identity robustness before moving to team/match statistics.

## Risks addressed

### 1. Name-only matching

Previous inference could generate local IDs such as `player_federico_valverde`. This was fragile when historical data used longer names. v0.18 introduces provider IDs and an identity map.

### 2. Multi-provider confusion

The MVP now uses API-Football as operational source of truth for current fixtures/lineups. StatsBomb remains historical training data. Cross-provider mapping is explicit in `player_identity_map.csv`.

### 3. Silent fallback to generic prior

Safe props now emits provider and identity-map status columns. If no historical mapping exists, warnings expose the fallback.

## Tests run

- Provider canonical ID stability
- Provider lineup standardization
- Provider ID to historical identity map construction
- Safe props uses identity map instead of generic prior
- Full test suite/compile check in this build environment

## Known limitations

- API-Football free-plan limits/coverage depend on the user's API subscription and selected leagues.
- StatsBomb Open Data does not cover every current player/league. Some provider players will remain unmatched until more historical data or manual reviewed aliases are added.
- The provider identity map should be refreshed when current lineups include new players.
