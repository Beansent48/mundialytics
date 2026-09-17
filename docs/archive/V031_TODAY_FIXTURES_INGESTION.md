# v0.31 — Today fixtures ingestion + value/evidence hardening

## What changed

v0.31 adds the missing operational step before real odds integration:

```text
partidos de hoy -> run-ready fixtures.csv -> statistical matchday -> dynamic lines -> later odds/value
```

It also hardens v0.30 dynamic-line value handling.

## New script

```powershell
python scripts/build_today_matchday_inputs.py `
  --today `
  --timezone Europe/Madrid `
  --competition world_cup `
  --out-dir data/input/generated
```

Outputs:

```text
data/input/generated/today_fixtures.csv
data/input/generated/today_provider_fixtures.csv
data/input/generated/today_current_lineups.csv
data/input/generated/today_squads.csv
data/input/generated/today_matchday_audit.json
```

`today_fixtures.csv` is directly compatible with `scripts/run_statistical_matchday.py`.

## Provider modes

The builder can fetch from free/public providers:

```powershell
python scripts/build_today_matchday_inputs.py --today --provider auto
python scripts/build_today_matchday_inputs.py --today --provider sofascore
python scripts/build_today_matchday_inputs.py --today --provider espn
```

Or build from an already fetched CSV for reproducible/offline tests:

```powershell
python scripts/build_today_matchday_inputs.py `
  --fixtures-source outputs/free_today_fixtures.csv `
  --date 2026-06-21 `
  --timezone Europe/Madrid `
  --out-dir data/input/generated
```

## Run matchday after building inputs

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/generated/today_fixtures.csv `
  --lineups data/input/generated/today_current_lineups.csv `
  --squads data/input/generated/today_squads.csv `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --model-config outputs/rolling_model_lab_current/best_rolling_model_config.json `
  --event-model-config outputs/player_prop_champion_full/prediction_registry.json `
  --out-dir outputs/statistical_matchday_today `
  --clean-out-dir `
  --no-demo-picks
```

The generated lineup/squad files are empty templates. Player props require current lineups or squads to be filled or fetched separately.

## Important odds fix

Demo odds no longer create real `high_value` / `medium_value` labels. If `bookmaker = demo_book`, rows are still priced for testing, but they are labelled:

```text
value_label = demo_odds_only
value_reason_code = demo_odds_detected_not_for_real_value
```

Real odds still use the normal labels:

```text
high_value / medium_value / fair_price / no_value
```

## Evidence improvement

Dynamic lines now include evidence source columns:

```text
recent_evidence_source
similar_elo_evidence_source
h2h_recent_evidence_source
```

For player props, recent evidence can fall back from current-team history to canonical player history:

```text
current_team_recent
canonical_player_recent
canonical_player_historical
not_available
```

This avoids unnecessarily blank evidence for players who changed teams or whose current-team sample is too small.
