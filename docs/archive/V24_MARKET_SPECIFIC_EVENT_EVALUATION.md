# v0.24 Market-Specific Event Evaluation & Hardening

This version adds a dedicated evaluation layer for team events and player props. The goal is to stop treating every predicted market as equally trustworthy.

## New commands

### Evaluate event models

```powershell
python scripts/evaluate_event_models.py `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --out-dir outputs/event_evaluation_current `
  --clean-out-dir `
  --min-train-matches 50 `
  --test-fraction 0.25 `
  --max-test-matches 400
```

Outputs:

```text
team_event_backtest_predictions.csv
team_event_line_probabilities.csv
player_event_backtest_predictions.csv
player_event_line_probabilities.csv
event_evaluation_summary.json
event_evaluation_report.html
```

### Run event model lab

```powershell
python scripts/run_event_model_lab.py `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --out-dir outputs/event_model_lab_current `
  --clean-out-dir `
  --n-trials 5 `
  --max-test-matches 400
```

Outputs:

```text
event_experiment_leaderboard.csv
best_event_model_config.json
event_model_lab_report.html
failed_event_experiments.json
event_model_lab_audit.json
```

## What it evaluates

Team event count models:

- shots
- shots_on_target
- fouls
- yellow_cards
- corners only if real data exists

Player prop models:

- player_shots 1+/2+/3+
- player_shots_on_target 1+/2+
- player_fouls_committed 1+/2+
- player_yellow_card 1+

## New metrics

Team event metrics:

- MAE
- RMSE
- Poisson negative log loss
- bias
- baseline comparison
- line-level Brier/log loss

Player prop metrics:

- Brier score for 1+
- log loss for 1+
- baseline comparison
- count MAE
- sample-size segments
- alternative line probabilities

## Important finding

The old assumption that every event model was already good is not safe. Shots and shots-on-target generally show clearer signal. Fouls and especially player yellow cards need stricter validation before betting usage.

Yellow cards are still useful as a descriptive/paper-tracking market, but player-card props should remain blocked or caution-only until they beat baseline on a proper holdout and ideally a rolling-origin backtest.

## Policy logic

The evaluation summary creates a market policy:

- usable_with_caution
- curiosity_only_or_paper_track
- curiosity_only_needs_calibration
- not_recommended_for_value_yet

This policy is intentionally conservative. It is better to disable a market than to create fake edge from noisy event probabilities.

## Known limitations

- The player-prop evaluation is lineup-known: historical participants are known, but actual minutes are not fed into the model.
- The current implementation is still a temporal holdout, not a full rolling-origin evaluation.
- Profitability still requires real odds, closing-line tracking and paper-mode EV records.
