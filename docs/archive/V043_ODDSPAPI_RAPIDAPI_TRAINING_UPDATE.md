# v0.43 — OddsPapi RapidAPI Free Historical Odds + Training Features

This version hardens the v0.42 OddsPapi integration for a free RapidAPI pilot and adds a leakage-safe odds feature layer for model training.

## What changed

### Provider client

- `OddsPapiClient` now supports:
  - `direct` mode: `https://v5.oddspapi.io/en` with `apiKey` query parameter.
  - `rapidapi` mode: `https://odds-api1.p.rapidapi.com/en` with RapidAPI headers.
- Adds hard per-run budget: `--max-api-calls`.
- Adds monthly ledger: `data/raw/oddspapi/request_ledger.jsonl`.
- Adds optional free-tier budget guard: `--monthly-budget 250`.
- Adds raw JSON cache, so repeated calls do not burn more requests.
- Handles 401/403 access errors and 429 rate limits explicitly.

### Odds shape support

The normalizer now supports several provider shapes:

- v5 current odds: `odds -> bookmaker -> oddsId -> price`.
- v5 historical tree: `bookmakers -> markets -> outcomes -> players -> snapshots`.
- older RapidAPI/examples: `bookmakerOdds -> markets -> outcomes`.

### Market mapping

The adapter maps OddsPapi `marketId/outcomeId` into Mundialytics contract fields:

- `1x2`
- `goals`
- `btts`
- `team_goals`
- `corners` / `team_corners`
- `yellow_cards` / `team_yellow_cards`
- `shots` / `team_shots`
- `shots_on_target` / `team_shots_on_target`
- `fouls` / `team_fouls`
- `goalkeeper_saves` / `team_goalkeeper_saves`
- `player_shots`
- `player_shots_on_target`
- `player_fouls_committed`
- `player_yellow_card`

Low-confidence mappings are kept but marked in `notes` for audit.

### New scripts

- `scripts/oddspapi_fetch_current_odds.py`
- `scripts/build_training_odds_features.py`

Existing v0.42 scripts now also accept:

- `--ledger-path`
- `--monthly-budget`

## Free-plan strategy

Do not backfill thousands of fixtures immediately. Use this order:

1. Probe metadata.
2. Build model lines / odds template.
3. Fetch fixtures in date windows, not one call per match.
4. Fuzzy-match provider fixtures to internal matches.
5. Fetch current odds for coverage sanity check.
6. Fetch a tiny historical pilot.
7. Build odds features.
8. Train/evaluate model with odds features only after coverage and leakage checks.

## PowerShell flow

```powershell
cd "C:\Users\Vicente\Desktop\BetBot\mundialytics_betting_engine"

$env:ODDSPAPI_MODE="rapidapi"
$env:RAPIDAPI_KEY="TU_RAPIDAPI_KEY"
$env:RAPIDAPI_ODDSPAPI_HOST="odds-api1.p.rapidapi.com"
$env:ODDSPAPI_BASE_URL="https://odds-api1.p.rapidapi.com/en"
$env:ODDSPAPI_MONTHLY_BUDGET="250"
$env:ODDSPAPI_LEDGER_PATH="data/raw/oddspapi/request_ledger.jsonl"
```

### 1. Probe safe

```powershell
python scripts/oddspapi_probe.py `
  --mode rapidapi `
  --out-dir outputs/oddspapi_probe_current `
  --max-api-calls 3 `
  --monthly-budget 250
```

### 2. Build model shortlist / contract

```powershell
python scripts/build_odds_ready_shortlist.py `
  --line-signals outputs/event_line_backtest_current_v0391/settled_event_line_signals.csv `
  --decision-matrix outputs/market_distribution_lab_current_v0391_clean/market_side_decision_matrix.csv `
  --out-dir outputs/odds_ready_current `
  --decisions candidate `
  --min-model-probability 0.52 `
  --min-fair-odds 1.15 `
  --max-fair-odds 4.00 `
  --max-rows-per-signal-group 5000
```

### 3. Plan fixture calls

```powershell
python scripts/oddspapi_build_fixture_request_plan.py `
  --model-lines outputs/odds_ready_current/model_market_lines.csv `
  --out-dir outputs/oddspapi_request_plan_current `
  --chunk-days 14 `
  --pad-hours 12 `
  --max-planned-calls 20
```

### 4. Fetch fixtures

```powershell
python scripts/oddspapi_fetch_fixtures.py `
  --windows outputs/oddspapi_request_plan_current/fixture_search_windows.csv `
  --out-dir outputs/oddspapi_fixtures_current `
  --mode rapidapi `
  --bookmakers pinnacle `
  --max-api-calls 10 `
  --monthly-budget 250
```

### 5. Match fixtures

```powershell
python scripts/oddspapi_match_fixtures.py `
  --model-lines outputs/odds_ready_current/model_market_lines.csv `
  --fixtures outputs/oddspapi_fixtures_current/oddspapi_fixtures.csv `
  --out-dir outputs/oddspapi_fixture_mapping_current `
  --auto-threshold 0.86
```

Review:

- `outputs/oddspapi_fixture_mapping_current/fixture_mapping_candidates.csv`
- `outputs/oddspapi_fixture_mapping_current/fixture_mapping_selected.csv`

### 6. Current odds sanity check

```powershell
python scripts/oddspapi_fetch_current_odds.py `
  --fixture-mapping outputs/oddspapi_fixture_mapping_current/fixture_mapping_selected.csv `
  --markets outputs/oddspapi_probe_current/soccer_markets.csv `
  --out-dir outputs/oddspapi_current_odds_current `
  --mode rapidapi `
  --bookmakers pinnacle `
  --max-api-calls 10 `
  --monthly-budget 250
```

### 7. Historical pilot for training data

For model training, prefer `pre_kickoff`, not `closing`, until you define exactly how close to kickoff the bot is allowed to operate.

```powershell
python scripts/oddspapi_fetch_historical_odds.py `
  --fixture-mapping outputs/oddspapi_fixture_mapping_current/fixture_mapping_selected.csv `
  --markets outputs/oddspapi_probe_current/soccer_markets.csv `
  --out-dir outputs/oddspapi_historical_odds_current `
  --mode rapidapi `
  --bookmaker pinnacle `
  --snapshot-policy pre_kickoff `
  --pre-kickoff-seconds 3600 `
  --max-api-calls 10 `
  --monthly-budget 250
```

### 8. Build training odds features

```powershell
python scripts/build_training_odds_features.py `
  --historical-odds outputs/oddspapi_historical_odds_current/historical_odds_input.csv `
  --out-dir outputs/odds_training_features_current
```

Outputs:

- `odds_features_match_1x2.csv`
- `odds_features_market_lines.csv`
- `odds_training_features_summary.json`

## Strict rule for model training

Never train with post-kickoff or post-result prices. Use either:

- `pre_kickoff` snapshots, e.g. 60 minutes before kickoff.
- `closing` only if your future live/paper strategy also places bets close to kickoff.

For now, use `pre_kickoff`.
