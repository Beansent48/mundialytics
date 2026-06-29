# v0.42 — OddsPapi / RapidAPI Free Historical Odds Integration

This version adds a safe first integration layer for OddsPapi-style odds APIs, including RapidAPI mode. It does not change the models, prediction logic, calibration, or value calculation. It only helps us acquire historical bookmaker odds and normalize them into the existing `historical_odds_input.csv` contract from v0.40.

## Why this version exists

The goal is to avoid the classic mistake: paying for or burning an odds API before knowing whether it actually covers our required markets. The engine already has model-side lines and an odds-ready schema. v0.42 adds a controlled ingestion workflow:

1. probe provider coverage with only a few calls;
2. build a low-call fixture discovery plan;
3. fuzzy-match provider fixtures to internal matches locally;
4. fetch historical odds only for reviewed fixture IDs and one bookmaker at a time;
5. normalize raw odds into `historical_odds_input.csv`;
6. run the existing v0.40 `calculate_value_edges_from_odds.py` script.

## New files

### Adapter

- `src/mundialytics/data/adapters/oddspapi.py`

Includes:

- `OddsPapiClient`
- direct mode with `ODDSPAPI_API_KEY`
- RapidAPI mode with `RAPIDAPI_KEY` and `RAPIDAPI_ODDSPAPI_HOST`
- raw JSON cache
- hard max-call budget
- soccer market mapping suggestions
- fixture dataframe normalization
- historical odds flattening into Mundialytics odds schema
- fixture fuzzy matching

### Scripts

- `scripts/oddspapi_probe.py`
- `scripts/oddspapi_build_fixture_request_plan.py`
- `scripts/oddspapi_fetch_fixtures.py`
- `scripts/oddspapi_match_fixtures.py`
- `scripts/oddspapi_fetch_historical_odds.py`

### Config

- `config/odds_provider_oddspapi.example.env`

Do not commit or upload your real API key.

## Important provider notes

OddsPapi docs use:

- base URL: `https://v5.oddspapi.io/en`
- `apiKey` as a query parameter in direct mode
- soccer `sportId = 10`
- `/fixtures` for fixture discovery
- `/markets?sportId=10` for soccer market/outcome IDs
- `/fixtures/odds/historical` for historical fixture odds

For the historical endpoint, if `oddsIds` is not provided, OddsPapi requires exactly one `bookmaker`. This is good for free-tier discipline: start with one bookmaker, usually `pinnacle` if covered, otherwise `bet365` or a bookmaker with strong soccer coverage.

RapidAPI mode may use a different host/base URL. Copy it from the RapidAPI code snippet. Do not guess it.

## Recommended free-tier workflow

### 0. Set environment variables

PowerShell direct provider mode:

```powershell
$env:ODDSPAPI_MODE="direct"
$env:ODDSPAPI_API_KEY="YOUR_KEY"
$env:ODDSPAPI_BASE_URL="https://v5.oddspapi.io/en"
```

PowerShell RapidAPI mode:

```powershell
$env:ODDSPAPI_MODE="rapidapi"
$env:RAPIDAPI_KEY="YOUR_RAPIDAPI_KEY"
$env:RAPIDAPI_ODDSPAPI_HOST="HOST_FROM_RAPIDAPI_CODE_SNIPPET"
$env:ODDSPAPI_BASE_URL="BASE_URL_FROM_RAPIDAPI_CODE_SNIPPET"
```

### 1. Dry-run probe first

```powershell
python scripts/oddspapi_probe.py `
  --mode rapidapi `
  --out-dir outputs/oddspapi_probe_current `
  --max-api-calls 3 `
  --dry-run
```

Then real probe, still only 3 calls:

```powershell
python scripts/oddspapi_probe.py `
  --mode rapidapi `
  --out-dir outputs/oddspapi_probe_current `
  --max-api-calls 3
```

Expected outputs:

- `sports.csv`
- `bookmakers_player_props.csv`
- `soccer_markets.csv`
- `soccer_market_mapping_suggested.csv`
- `probe_summary.json`

Review `soccer_market_mapping_suggested.csv`. Do not assume all secondary stat markets are correctly mapped. Main markets like 1X2, BTTS and goals totals are safer. Corners/cards/shots/fouls/saves must be checked by market name/outcome IDs.

### 2. Build model shortlist first

Use v0.40/v0.41 first:

```powershell
python scripts/build_odds_ready_shortlist.py `
  --line-signals outputs/event_line_backtest_current_v0391/settled_event_line_signals.csv `
  --decision-matrix outputs/market_distribution_lab_current_v0391_clean/market_side_decision_matrix.csv `
  --out-dir outputs/odds_ready_current `
  --decisions candidate `
  --min-model-probability 0.52 `
  --min-fair-odds 1.25 `
  --max-fair-odds 3.50 `
  --max-rows-per-signal-group 5000
```

### 3. Build fixture request plan

Never call `/fixtures` once per match. Chunk by date.

```powershell
python scripts/oddspapi_build_fixture_request_plan.py `
  --model-lines outputs/odds_ready_current/model_market_lines.csv `
  --out-dir outputs/oddspapi_request_plan_current `
  --chunk-days 14 `
  --pad-hours 12 `
  --max-planned-calls 20
```

If too many windows are planned, increase `--chunk-days` or restrict the model shortlist.

### 4. Fetch fixture candidates with budget

Dry run:

```powershell
python scripts/oddspapi_fetch_fixtures.py `
  --windows outputs/oddspapi_request_plan_current/fixture_search_windows.csv `
  --out-dir outputs/oddspapi_fixtures_current `
  --mode rapidapi `
  --bookmakers pinnacle `
  --max-api-calls 10 `
  --dry-run
```

Real run:

```powershell
python scripts/oddspapi_fetch_fixtures.py `
  --windows outputs/oddspapi_request_plan_current/fixture_search_windows.csv `
  --out-dir outputs/oddspapi_fixtures_current `
  --mode rapidapi `
  --bookmakers pinnacle `
  --max-api-calls 10
```

### 5. Match internal matches to provider fixture IDs

```powershell
python scripts/oddspapi_match_fixtures.py `
  --model-lines outputs/odds_ready_current/model_market_lines.csv `
  --fixtures outputs/oddspapi_fixtures_current/oddspapi_fixtures.csv `
  --out-dir outputs/oddspapi_fixture_mapping_current `
  --auto-threshold 0.86
```

Review:

- `fixture_mapping_candidates.csv`
- `fixture_mapping_selected.csv`

Do not fetch historical odds for bad fixture matches.

### 6. Fetch historical odds for a tiny reviewed pilot

Start with one bookmaker and a very small call budget.

```powershell
python scripts/oddspapi_fetch_historical_odds.py `
  --fixture-mapping outputs/oddspapi_fixture_mapping_current/fixture_mapping_selected.csv `
  --markets outputs/oddspapi_probe_current/soccer_markets.csv `
  --out-dir outputs/oddspapi_historical_odds_current `
  --mode rapidapi `
  --bookmaker pinnacle `
  --snapshot-policy closing `
  --max-api-calls 10
```

This produces:

- `historical_odds_input.csv`
- raw historical JSON files
- `historical_odds_summary.json`

### 7. Audit coverage before calculating value

```powershell
python scripts/audit_historical_odds_coverage.py `
  --model-lines outputs/odds_ready_current/model_market_lines.csv `
  --historical-odds outputs/oddspapi_historical_odds_current/historical_odds_input.csv `
  --out-dir outputs/oddspapi_historical_odds_coverage_current
```

### 8. Calculate EV/value edges

```powershell
python scripts/calculate_value_edges_from_odds.py `
  --model-lines outputs/odds_ready_current/model_market_lines.csv `
  --historical-odds outputs/oddspapi_historical_odds_current/historical_odds_input.csv `
  --out-dir outputs/value_edges_oddspapi_current `
  --min-ev 0.03 `
  --min-edge 0.02
```

## Free-tier discipline

Start with this budget:

- probe: 3 calls
- fixture discovery: 5-10 calls
- historical pilot: 10 calls
- total pilot: 18-23 calls

Do not run full backfill until:

- fixture matching quality is good;
- `soccer_market_mapping_suggested.csv` is reviewed;
- `audit_historical_odds_coverage.py` shows useful coverage;
- historical odds actually contain the markets we care about.

## Market priorities for the first pilot

Priority 1:

- `1x2`
- `goals` totals
- `btts`

Priority 2:

- `team_yellow_cards`
- `team_fouls`
- `yellow_cards`
- `fouls`
- `team_shots_on_target`
- `goalkeeper_saves`

Priority 2 must be verified in provider market metadata. If the provider lacks these markets, keep them as `not_available`, not guessed.

## What this version still does not do

- It does not place real bets.
- It does not assume provider market mapping is perfect.
- It does not scrape bookmaker websites.
- It does not bypass API limits.
- It does not convert high hit-rate low-odds signals into ROI claims without bookmaker prices.

## Validation

The new v0.42 adapter tests pass along with the v0.40/v0.41 odds tests:

```text
12 passed
```
