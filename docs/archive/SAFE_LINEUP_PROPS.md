# Safe lineup-only player props

Historical event data may include retired players and old club seasons. That is fine for training rates and calibrating probabilities, but operational inference must never choose candidates from the historical dataset.

The rule is:

> Historical data trains the model. Current lineups decide who can appear in predictions.

## Required flow

1. Build/validate event data as usual.
2. Repair missing dates in backtest predictions using `statsbomb_player_events.csv`.
3. Run a temporal calibration check with real dates.
4. Provide a current lineup CSV.
5. Generate props only for those lineup players.

## Repair dates

```powershell
python scripts/repair_player_prop_predictions.py `
  --predictions outputs/player_props_statsbomb_national/validation/player_props_backtest_predictions.csv `
  --player-events outputs/player_props_statsbomb_national/statsbomb_player_events.csv `
  --out outputs/player_props_statsbomb_national/validation/player_props_backtest_predictions_repaired.csv `
  --report outputs/player_props_statsbomb_national/validation/player_props_backtest_predictions_repair_report.json
```

## Temporal calibration check

```powershell
python scripts/temporal_calibration_check.py `
  --predictions outputs/player_props_statsbomb_national/validation/player_props_backtest_predictions_repaired.csv `
  --player-events outputs/player_props_statsbomb_national/statsbomb_player_events.csv `
  --out-dir outputs/player_props_statsbomb_national/calibration_temporal_check_real_dates `
  --calibration-fraction 0.5 `
  --min-market-rows 500
```

## Lineup-only inference

Create a file like `data/today/current_lineups.csv` with:

```csv
match_id,date,team,opponent,player,position,expected_minutes,started
ESP_URU_2026,2026-06-26,Spain,Uruguay,Lamine Yamal,RW,85,1
```

Then run:

```powershell
python scripts/run_safe_props_for_lineups.py `
  --lineups data/today/current_lineups.csv `
  --player-events outputs/player_props_statsbomb_national/statsbomb_player_events.csv `
  --calibration-predictions outputs/player_props_statsbomb_national/validation/player_props_backtest_predictions_repaired.csv `
  --calibration-results outputs/player_props_statsbomb_national/calibration/calibration_search_results.csv `
  --out outputs/today/safe_lineup_props.csv
```

Outputs include `raw_probability`, `calibrated_probability`, `safe_probability`, `sample_size`, `confidence_flag`, and warnings. The `safe_probability` column applies market caps to avoid near-certainty outputs from over-flexible calibration.
