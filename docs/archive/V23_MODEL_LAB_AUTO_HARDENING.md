# v0.23 Model Lab Auto Hardening

This version adds an automatic experiment loop for the match prediction core. The aim is to stop tuning by intuition and run reproducible trials over conservative model variants.

## Main command

```powershell
python scripts/run_model_lab.py `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --out-dir outputs/model_lab_current `
  --clean-out-dir `
  --n-trials 10
```

For fast local iteration, use:

```powershell
python scripts/run_model_lab.py `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --out-dir outputs/model_lab_current `
  --clean-out-dir `
  --n-trials 4 `
  --max-test-matches 350
```

## Outputs

- `experiment_leaderboard.csv`
- `best_model_config.json`
- `best_calibration_model.json`
- `model_lab_report.html`
- `failed_experiments.json`
- `model_lab_audit.json`
- `trials/<trial_id>/match_evaluation_summary.json`

## Applying the best model

```powershell
python scripts/evaluate_statistical_core.py `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --out-dir outputs/evaluation_best_model_current `
  --clean-out-dir `
  --min-train-matches 50 `
  --test-fraction 0.25 `
  --model-config outputs/model_lab_current/best_model_config.json
```

Then run matchday with both the best config and the fitted calibration:

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/fixtures.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --odds data/input/odds.csv `
  --tournament-config data/input/tournament_config.csv `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --model-config outputs/model_lab_current/best_model_config.json `
  --calibration-model outputs/evaluation_best_model_current/match_calibration_model.json `
  --out-dir outputs/statistical_matchday_current `
  --clean-out-dir `
  --no-demo-picks
```

## Current real-data result from the uploaded dataset

The best fast-lab trial was `cap4_shrink6_low8_draw005`. On the full 990-match holdout it produced:

- raw 1X2 log loss: ~1.044, improved from the previous raw ~1.213
- calibrated 1X2 log loss: ~1.037
- accuracy pick max: ~47.4%
- over 2.5 raw log loss: ~0.689
- BTTS raw log loss: ~0.700

This is a real improvement in probability quality, especially by removing extreme overconfidence. It is still not a betting-ready edge engine without real odds and closing-line tracking.
