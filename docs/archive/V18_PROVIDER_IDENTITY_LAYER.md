# v0.18 Provider Identity Layer

## Decision

For the free MVP, Mundialytics now treats **API-Football / API-Sports** as the operational provider for current fixtures and lineups, while StatsBomb Open Data remains the historical event source used to train and calibrate models.

This avoids relying on fragile name-only matching at matchday time.

## Source-of-truth rules

1. If `provider_player_id` exists, use it as the operational identity.
2. Build a provider identity map that links `api_football:<provider_player_id>` to the historical `player_id_global` used by StatsBomb-derived player props.
3. Use name/fuzzy matching only once when building the map or as a clearly warned fallback.
4. Preserve all identity fields in outputs so bad joins are auditable.

## New files

- `src/mundialytics/data/provider_identity.py`
- `src/mundialytics/data/adapters/api_football.py`
- `scripts/fetch_api_football_fixtures.py`
- `scripts/fetch_api_football_lineups.py`
- `scripts/build_provider_identity_map.py`
- `scripts/diagnose_provider_identity.py`
- `tests/test_v18_provider_identity_layer.py`
- `data/identity/README.md`
- `data/sample/provider/api_football_lineups_sample.csv`

## Typical workflow

### 1. Fetch current lineups from API-Football

Requires an API key from the free/API-Sports plan:

```powershell
$env:API_FOOTBALL_KEY="YOUR_KEY"

python scripts/fetch_api_football_lineups.py `
  --fixture-id 123456 `
  --date 2026-06-26 `
  --competition "FIFA World Cup" `
  --out outputs/api_football_current_lineups.csv `
  --raw-out outputs/api_football_current_lineups_raw.json
```

The output contains `provider`, `provider_player_id`, `canonical_player_id`, `player`, `team`, `position`, `started`, and `expected_minutes`.

### 2. Build provider identity map

```powershell
python scripts/build_provider_identity_map.py `
  --provider-players outputs/api_football_current_lineups.csv `
  --historical-player-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --provider api_football `
  --out data/identity/player_identity_map.csv
```

This produces:

- `canonical_player_id`: e.g. `api_football:12345`
- `historical_player_id_global`: model lookup ID from the historical props dataset
- `match_status`, `match_method`, `match_confidence`
- historical sample columns

### 3. Diagnose identity map

```powershell
python scripts/diagnose_provider_identity.py `
  --identity-map data/identity/player_identity_map.csv `
  --lineups outputs/api_football_current_lineups.csv
```

Rows with `unmatched` or `ambiguous` should be reviewed before trusting props.

### 4. Run safe props with identity map

```powershell
python scripts/run_safe_props_for_lineups.py `
  --lineups outputs/api_football_current_lineups.csv `
  --identity-map data/identity/player_identity_map.csv `
  --player-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --calibration-predictions outputs/player_props_national_men_v16/validation/player_props_backtest_predictions.csv `
  --calibration-policy outputs/player_props_national_men_v16/player_props_policy.json `
  --out outputs/safe_lineup_props_provider.csv `
  --strict-lineup-contract
```

## Output identity columns

Safe props now preserve:

- `provider`
- `provider_player_id`
- `provider_player_name`
- `canonical_player_id`
- `historical_player_id_global`
- `historical_player_name`
- `identity_map_status`
- `identity_map_method`
- `identity_map_confidence`
- `resolved_player_id_global`
- `matched_player_name`
- `player_match_status`

## Audit stance

A player with `identity_map_status=unmatched` may still receive a fallback prediction, but the row is warned. For betting decisions, unmatched high-profile players should be treated as **not ready** until mapped.
