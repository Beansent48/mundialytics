# v0.46 — OddsPapi Historical Odds Backfill

Purpose: download historical odds in a way that can be linked to Mundialytics historical match data without polluting the model with post-match/live leakage.

## Design rules

1. Discover OddsPapi `fixtureId`s first.
2. Match provider fixtures to internal `match_id`s before downloading odds.
3. Download raw historical odds JSON by `fixtureId` + one `bookmaker`.
4. Store raw JSON permanently for reproducibility.
5. Normalize locally and keep only Mundialytics target markets.
6. Build pre-match snapshots (`t24h`, `t6h`, `t1h`, `t10m`, `closing`) before training/backtesting.
7. Never train directly on all historical ticks.
8. Never use post-kickoff prices for pre-match betting models.

## Documentation constraints reflected in the code

- OddsPapi fixtures endpoint supports `sportId`, `from`, `to`, `statusId`, `hasOdds`, and `bookmakers`.
- Historical odds endpoint requires `fixtureId` plus `bookmaker` if `oddsIds` are not supplied.
- Historical odds endpoint has a 5000ms cooldown; the backfill script enforces `--min-interval-sec 5.1` by default.
- Historical odds availability starts at January 2026 according to OddsPapi docs.

## Target markets

The backfill keeps these families by default:

- `1x2`
- `btts`
- `goals`, `team_goals`
- `corners`, `team_corners`
- `yellow_cards`, `team_yellow_cards`
- `shots`, `team_shots`
- `shots_on_target`, `team_shots_on_target`
- `fouls`, `team_fouls`
- `goalkeeper_saves`, `team_goalkeeper_saves`
- `player_shots`
- `player_shots_on_target`
- `player_fouls_committed`
- `player_yellow_card`

## Commands

### 0. Config

```powershell
cd "C:\Users\Vicente\Desktop\BetBot\mundialytics_betting_engine"
$env:MUNDIALYTICS_API_CONFIG="C:\Users\Vicente\Desktop\BetBot\mundialytics_betting_engine\config\mundialytics_api_config.local.yaml"
$env:RAPIDAPI_KEY="TU_RAPIDAPI_KEY"
```

### 1. Probe markets if needed

```powershell
python scripts/oddspapi_probe.py `
  --provider-config config/mundialytics_api_config.local.yaml `
  --mode rapidapi `
  --out-dir outputs/oddspapi_probe_current `
  --max-api-calls 3 `
  --monthly-budget 250
```

### 2. Build market mapping candidates

```powershell
python scripts/oddspapi_build_market_mapping_candidates.py `
  --markets outputs/oddspapi_probe_current/soccer_markets.csv `
  --out-dir outputs/oddspapi_market_mapping_current
```

Review:

```text
outputs/oddspapi_market_mapping_current/oddspapi_target_market_mapping_candidates.csv
```

### 3. Build historical fixture request plan

Use any internal historical matches CSV that has at least:

```text
match_id,date or kickoff_utc,home_team,away_team
```

Example:

```powershell
python scripts/oddspapi_build_historical_fixture_plan.py `
  --matches outputs/event_line_backtest_current_v0391/settled_event_line_signals.csv `
  --out-dir outputs/oddspapi_historical_fixture_plan_current `
  --min-date 2026-01-01 `
  --chunk-hours 24 `
  --pad-hours 4 `
  --max-windows 7
```

For a full backfill later, remove `--max-windows`.

### 4. Fetch historical provider fixtures

Dry-run first:

```powershell
python scripts/oddspapi_fetch_historical_fixtures.py `
  --windows outputs/oddspapi_historical_fixture_plan_current/fixture_request_windows.csv `
  --out-dir outputs/oddspapi_historical_fixtures_current `
  --provider-config config/mundialytics_api_config.local.yaml `
  --mode rapidapi `
  --bookmakers pinnacle `
  --max-api-calls 7 `
  --monthly-budget 250 `
  --dry-run
```

Real:

```powershell
python scripts/oddspapi_fetch_historical_fixtures.py `
  --windows outputs/oddspapi_historical_fixture_plan_current/fixture_request_windows.csv `
  --out-dir outputs/oddspapi_historical_fixtures_current `
  --provider-config config/mundialytics_api_config.local.yaml `
  --mode rapidapi `
  --bookmakers pinnacle `
  --max-api-calls 7 `
  --monthly-budget 250
```

If RapidAPI rejects `from/to`, rerun with:

```powershell
  --param-style v5_epoch
```

### 5. Match provider fixtures to internal matches

```powershell
python scripts/oddspapi_match_historical_fixtures.py `
  --internal-matches outputs/oddspapi_historical_fixture_plan_current/internal_matches_prepared.csv `
  --provider-fixtures outputs/oddspapi_historical_fixtures_current/oddspapi_historical_fixtures.csv `
  --out-dir outputs/oddspapi_historical_fixture_mapping_current `
  --auto-threshold 0.86 `
  --max-hours-diff 30
```

Review:

```text
outputs/oddspapi_historical_fixture_mapping_current/fixture_mapping_selected.csv
outputs/oddspapi_historical_fixture_mapping_current/fixture_mapping_manual_review.csv
```

### 6. Download historical odds raw JSON and normalized ticks

Pilot first with 10 fixtures:

```powershell
python scripts/oddspapi_fetch_historical_odds_backfill.py `
  --fixture-mapping outputs/oddspapi_historical_fixture_mapping_current/fixture_mapping_selected.csv `
  --markets outputs/oddspapi_probe_current/soccer_markets.csv `
  --out-dir outputs/oddspapi_historical_odds_backfill_current `
  --provider-config config/mundialytics_api_config.local.yaml `
  --mode rapidapi `
  --bookmaker pinnacle `
  --max-fixtures 10 `
  --max-api-calls 10 `
  --monthly-budget 250
```

Output:

```text
outputs/oddspapi_historical_odds_backfill_current/raw/pinnacle/*.json
outputs/oddspapi_historical_odds_backfill_current/historical_odds_ticks.csv
```

### 7. Build leakage-safe snapshots

```powershell
python scripts/oddspapi_build_snapshot_odds.py `
  --historical-odds-ticks outputs/oddspapi_historical_odds_backfill_current/historical_odds_ticks.csv `
  --fixture-mapping outputs/oddspapi_historical_fixture_mapping_current/fixture_mapping_selected.csv `
  --out-dir outputs/oddspapi_snapshot_odds_current `
  --snapshots t24h,t6h,t1h,t10m,closing
```

Output:

```text
outputs/oddspapi_snapshot_odds_current/historical_odds_snapshots.csv
outputs/oddspapi_snapshot_odds_current/historical_odds_input.csv
```

`historical_odds_input.csv` is backward-compatible with the existing EV scripts and uses `t1h` if available, otherwise closing.

### 8. Build training odds features

```powershell
python scripts/oddspapi_build_training_odds_features_from_snapshots.py `
  --snapshot-odds outputs/oddspapi_snapshot_odds_current/historical_odds_snapshots.csv `
  --out-dir outputs/odds_training_features_from_snapshots_current
```

Output:

```text
outputs/odds_training_features_from_snapshots_current/odds_features_market_lines_by_snapshot.csv
outputs/odds_training_features_from_snapshots_current/odds_features_match_1x2_by_snapshot.csv
```

### 9. Audit coverage

```powershell
python scripts/oddspapi_audit_backfill_coverage.py `
  --fixture-mapping outputs/oddspapi_historical_fixture_mapping_current/fixture_mapping_selected.csv `
  --historical-odds-ticks outputs/oddspapi_historical_odds_backfill_current/historical_odds_ticks.csv `
  --snapshot-odds outputs/oddspapi_snapshot_odds_current/historical_odds_snapshots.csv `
  --out-dir outputs/oddspapi_backfill_coverage_current
```

## Training policy

Recommended model separation:

1. Pure sports model: no odds features.
2. Market-aware/calibration model: odds features allowed from a fixed pre-match horizon.

If the future bot places bets at T-1h, train/evaluate with T-1h odds. If it places bets at T-10m, train/evaluate with T-10m odds.

