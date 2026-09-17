# v0.14 Operational Hardening

This release recenters the project on the actual betting-analysis workflow:

1. Current fixtures -> ELO + Poisson match probabilities.
2. Current official/expected lineups -> eligible player candidates.
3. Historical player events -> player event rates.
4. Calibration -> calibrated probabilities.
5. Safety layer -> capped `safe_probability`, confidence flags, low-sample warnings.
6. Optional odds -> paper-mode value candidates.

## Important fixes

### 1. No observed-minutes leakage in props backtests

Previous prop backtests used the player\'s observed minutes from the target match as `expected_minutes`. That is not available pre-match and can leak future information. v0.14 changes the default:

- `expected_minutes` is estimated from historical training data.
- observed target-match minutes are saved separately as `actual_minutes`.
- `expected_minutes_source` records where the prediction minutes came from.
- audit fails if `expected_minutes_source` contains `LEAKY` or `observed_test_minutes`.

Diagnostic leaky mode exists only with `--allow-observed-test-minutes` and is blocked by strict audit/calibration.

### 2. Prediction metadata contract

Every prop prediction CSV must preserve:

- `match_id`, `date`, `competition`, `team_scope`
- `team`, `opponent`, `team_id`, `opponent_id` when available
- `player`, `player_id_global`, `player_context_id`
- `position`, `started`
- `market_type`, `line`
- `probability`, `raw_probability`
- `expected_minutes`, `expected_minutes_source`, `actual_minutes`
- `sample_size`, `actual_count`, `actual`

This prevents the previous bug where `competition` disappeared before calibration.

### 3. Current-lineup-only inference

Historical players may train the model, but `run_safe_props_for_lineups.py` and `run_matchday_analysis.py` only output players supplied in the current lineup CSV. Retired historical players cannot appear unless they are explicitly present in the current lineup input.

### 4. Match context now flows into props

`run_matchday_analysis.py` attaches fixture-level ELO/Poisson context to lineup rows, including team-perspective `elo_diff` and a bounded `expected_possession` proxy. Player props can therefore react to match strength/context.

## Main commands

### Rebuild clean props

```powershell
python scripts/run_clean_props_rebuild.py `
  --player-events outputs/player_props_statsbomb_national/statsbomb_player_events.csv `
  --lineups outputs/player_props_statsbomb_national/statsbomb_lineups.csv `
  --out-dir outputs/player_props_statsbomb_clean_v14 `
  --exclude-competitions "StatsBomb Open Data" `
  --expected-domain mixed `
  --min-train-matches 100 `
  --test-matches 300 `
  --min-calibration-market-rows 500
```

### Matchday analysis

```powershell
python scripts/run_matchday_analysis.py `
  --goal-bundle outputs/validation_national_elite_recent/final_national_poisson_model.pkl `
  --fixtures data/today/fixtures.csv `
  --lineups data/today/current_lineups.csv `
  --player-events outputs/player_props_statsbomb_clean_v14/statsbomb_player_events_clean.csv `
  --calibration-predictions outputs/player_props_statsbomb_clean_v14/validation/player_props_backtest_predictions.csv `
  --calibration-results outputs/player_props_statsbomb_clean_v14/calibration/calibration_search_results.csv `
  --out-dir outputs/today_analysis
```

The output includes:

- `match_predictions.csv`
- `lineups_with_match_context.csv`
- `safe_lineup_props.csv`
- `matchday_analysis_report.json`
