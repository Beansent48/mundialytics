# v0.16 — Hierarchical calibration + club-to-national player evidence

## Purpose

This version keeps v0.15's objective competition taxonomy and adds two practical betting-engine improvements:

1. **Hierarchical prop calibration**: calibrate by competition when there is enough data, otherwise fall back to broader domains.
2. **Cross-context player evidence**: for national-team player props, a player's club history can inform current national-team predictions, but only from rows dated before the target test period.

No subjective `match_importance` feature is introduced.

## Calibration hierarchy

The engine now evaluates/applies calibration in this order:

1. `market_type + competition`
2. `market_type + team_type + gender + competition_context`
3. `market_type + team_type + gender`
4. `market_type`
5. identity fallback if no group has enough rows/classes

This avoids overfitting tiny competitions while still capturing league-specific tendencies when there is enough sample.

## Cross-context player features

Backtests and inference now expose:

- `club_minutes_sample`
- `national_minutes_sample`
- `cross_context_feature_used`

For national-team predictions, `cross_context_feature_used=True` means the player had usable club history in the training feature set. This is expected for many current international players and is useful when national-team samples are small.

## Anti-leakage rule

When `--feature-player-events` is supplied in validation, feature rows are cut off before the first test match date. The model can use older club history for national players, but cannot use future club/national rows from after the prediction date range begins.

## New/updated commands

### Clean rebuild with hierarchical calibration

```powershell
python scripts/run_clean_props_rebuild.py `
  --player-events outputs/player_props_statsbomb_national/statsbomb_player_events.csv `
  --lineups outputs/player_props_statsbomb_national/statsbomb_lineups.csv `
  --feature-player-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --out-dir outputs/player_props_national_men_v16 `
  --include-competitions "FIFA World Cup" "UEFA Euro" "African Cup of Nations" "Copa America" `
  --expected-domain national `
  --min-train-matches 50 `
  --test-matches 100 `
  --min-calibration-market-rows 200 `
  --min-hierarchical-group-rows 500
```

### Standalone hierarchical calibration

```powershell
python scripts/temporal_calibration_check.py `
  --predictions outputs/player_props_national_men_v16/validation/player_props_backtest_predictions.csv `
  --out-dir outputs/player_props_national_men_v16/calibration_temporal_check `
  --hierarchical `
  --min-market-rows 200 `
  --min-hierarchical-group-rows 500 `
  --require-valid-date
```

## Audit expectations

A healthy v0.16 report should show:

- `date_null_rate = 0.0`
- `uses_observed_test_minutes = false`
- no `LEAKY` expected-minutes sources
- correct `team_type`, `team_scope`, `competition_context`, `gender`
- `hierarchical_temporal_calibration_report.json` generated
- no predictions for players outside the current lineup/squad during inference

## Remaining limitations

- Competition-specific calibration needs enough rows. If not, fallback is correct.
- Club-to-national linking depends on `player_id_global`; name-based IDs are useful but imperfect if two players share names.
- Expected minutes remain one of the highest-impact assumptions and should be improved with lineup/role sources when available.
- This is still paper-mode infrastructure, not a green light for real staking.
