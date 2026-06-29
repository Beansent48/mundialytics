# Rebuild clean player-prop calibration dataset

Use this when a calibrated/backtest file has missing dates or placeholder competitions such as `StatsBomb Open Data`.

The safe operational rule is:

> Historical players may train rates. Future predictions must be generated only for players supplied in the current lineup input.

## Why this rebuild exists

If rows with `date = NaN` are kept, temporal backtests can silently select unknown-date matches as the latest test set. That makes the anti-overfitting check unreliable.

If a placeholder competition such as `StatsBomb Open Data` remains, the dataset probably lost match metadata and should not be used as a final calibration set.

## Recommended flow: all valid dated data

```powershell
python scripts/filter_player_events_for_props.py `
  --player-events outputs/player_props_statsbomb_national/statsbomb_player_events.csv `
  --lineups outputs/player_props_statsbomb_national/statsbomb_lineups.csv `
  --out-player-events outputs/player_props_statsbomb_clean/statsbomb_player_events_clean.csv `
  --out-lineups outputs/player_props_statsbomb_clean/statsbomb_lineups_clean.csv `
  --report outputs/player_props_statsbomb_clean/filter_report.json `
  --require-valid-date `
  --exclude-competitions "StatsBomb Open Data"
```

Then validate and calibrate:

```powershell
python scripts/validate_player_props.py `
  --player-events outputs/player_props_statsbomb_clean/statsbomb_player_events_clean.csv `
  --lineups outputs/player_props_statsbomb_clean/statsbomb_lineups_clean.csv `
  --out-dir outputs/player_props_statsbomb_clean/validation `
  --min-train-matches 100 `
  --test-matches 300 `
  --require-valid-date `
  --exclude-competitions "StatsBomb Open Data"

python scripts/calibrate_player_props.py `
  --predictions outputs/player_props_statsbomb_clean/validation/player_props_backtest_predictions.csv `
  --out-dir outputs/player_props_statsbomb_clean/calibration `
  --calibration-fraction 0.5 `
  --min-market-rows 500

python scripts/temporal_calibration_check.py `
  --predictions outputs/player_props_statsbomb_clean/validation/player_props_backtest_predictions.csv `
  --out-dir outputs/player_props_statsbomb_clean/calibration_temporal_check `
  --calibration-fraction 0.5 `
  --min-market-rows 500
```

## Recommended flow: men's senior national-only

This is cleaner for World Cup analysis, but has fewer rows.

```powershell
python scripts/filter_player_events_for_props.py `
  --player-events outputs/player_props_statsbomb_national/statsbomb_player_events.csv `
  --lineups outputs/player_props_statsbomb_national/statsbomb_lineups.csv `
  --out-player-events outputs/player_props_statsbomb_mens_national_clean/statsbomb_player_events_clean.csv `
  --out-lineups outputs/player_props_statsbomb_mens_national_clean/statsbomb_lineups_clean.csv `
  --report outputs/player_props_statsbomb_mens_national_clean/filter_report.json `
  --require-valid-date `
  --include-competitions "FIFA World Cup" "UEFA Euro" "African Cup of Nations" "Copa America" "Copa América" `
  --set-team-scope national
```

Then run the same validation/calibration commands using `outputs/player_props_statsbomb_mens_national_clean/...`.
