# Player Props Calibration

This project separates event extraction from probability calibration:

1. `run_player_props_pipeline.py` builds real player-event tables and validates raw prop probabilities.
2. `calibrate_player_props.py` takes `player_props_backtest_predictions.csv` and searches post-model calibration methods per market.

The calibration script is designed to detect under/over-confidence by market and choose the best out-of-sample calibrator using a temporal split.

## Methods tested

- `identity`: raw model probability.
- `rate_shift`: intercept shift in logit space so the calibration split average probability matches observed rate.
- `platt_logit`: logistic calibration using `logit(raw_probability)`.
- `platt_logit_extra`: logistic calibration using `logit(raw_probability)`, `expected_minutes`, `sample_size`, and `expected_count`.
- `isotonic`: non-parametric monotonic calibration when enough data exists.

## Standalone usage

```powershell
python scripts/calibrate_player_props.py `
  --predictions outputs/player_props_statsbomb_national/validation/player_props_backtest_predictions.csv `
  --out-dir outputs/player_props_statsbomb_national/calibration `
  --calibration-fraction 0.5 `
  --min-market-rows 500
```

## Integrated usage

```powershell
python scripts/run_player_props_pipeline.py `
  --statsbomb-data data/raw/statsbomb/open-data/data `
  --team-scope national `
  --out-dir outputs/player_props_statsbomb_national `
  --min-matches 50 `
  --min-player-rows 500 `
  --min-train-matches 50 `
  --test-matches 300 `
  --run-calibration `
  --min-calibration-market-rows 500
```

## Outputs

- `calibration_search_results.csv`: all methods per market with Brier/log loss/calibration bias.
- `calibrated_player_prop_predictions.csv`: best calibrated predictions per market.
- `calibration_report.json`: best method and metrics by market.
- `incoherence_report.json`: warnings such as invalid probabilities, missing values, duplicated rows, or market-level bias.
- `reliability_raw_by_market.csv`: raw reliability bins.
- `reliability_calibrated_by_market.csv`: calibrated reliability bins.

## Important interpretation

A calibration method must be chosen by out-of-sample metrics. Do not force a calibrator just because it raises probabilities. If `identity` wins, the raw model is better for that market. If `rate_shift` or `platt_logit_extra` wins, use calibrated probabilities for paper-mode decisions.
