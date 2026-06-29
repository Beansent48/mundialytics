# v0.49.3 — Statistical Model Calibration Roadmap and First Implementation

Status: Implemented for goals/scoreline evaluation enhancements; Planned for corners/cards negative-binomial models.

## 1. Context

The user clarified that Mundialytics is primarily a statistical football engine and simulator, not a system whose purpose is to bet every 1X2 edge.

After v0.49.2, historical validation showed that the Poisson baseline is consistently stronger than `random_forest_lambda` across national teams and several club datasets. The next improvement should therefore strengthen the statistical model around Poisson instead of replacing it blindly with a more complex model.

The user also clarified that Elo was one of the original design pillars. The repository already had an internal Elo implementation and used it in walk-forward backtests, but the documentation needed to make this explicit.

## 2. Objective

Improve the statistical-engine foundation around:

1. detailed calibration diagnostics for totals and BTTS,
2. a calibration layer for 1X2, totals and BTTS,
3. Dixon-Coles scoreline adjustment diagnostics,
4. time decay and shrinkage for goal-lambda training,
5. explicit Elo/ClubElo role in the model design,
6. a documented next step for corners/cards count models, probably Negative Binomial.

## 3. Scope Implemented

### Detailed line calibration

`run_historical_validation.py` now writes:

```text
statistical_engine_line_calibration_<model_type>.csv
```

This file contains probability-bin diagnostics for:

```text
total_goals over/under 0.5, 1.5, 2.5, 3.5, 4.5
BTTS yes/no
```

### Calibration layer

`src/mundialytics/evaluation/statistical_engine.py` now evaluates a temporal holdout calibration layer:

```text
statistical_engine_calibration_layer_<model_type>.csv
```

The first part of the backtest window is used as a calibration window and the later part as an evaluation window.

Implemented calibration diagnostics:

```text
1X2: one-vs-rest isotonic calibration with row renormalization
total goals lines: binary isotonic calibration when enough data exists
BTTS: binary isotonic calibration when enough data exists
```

When there is not enough calibration data, the calibration layer falls back to identity and marks the method clearly.

### Dixon-Coles diagnostics

`src/mundialytics/evaluation/statistical_engine.py` now estimates a Dixon-Coles low-score adjustment parameter on the calibration window and evaluates it on the later holdout window.

Output:

```text
statistical_engine_dixon_coles_scorelines_<model_type>.csv
```

The summary reports:

```text
dixon_coles.rho
independent_scoreline_metrics
dixon_coles_scoreline_metrics
```

This is a statistical scoreline-distribution diagnostic, not a betting feature.

### Time decay

`GoalLambdaModel` now supports recency sample weights:

```text
GoalModelConfig.time_decay_half_life_days
```

Default:

```text
365 days
```

This means recent training rows can matter more than older rows without breaking temporal ordering.

### Rolling-feature shrinkage

`build_goal_training_frame()` now shrinks noisy rolling team features toward dataset medians when a team has few prior matches.

Default prior:

```text
rolling_shrinkage_prior_matches = 10
```

This stabilizes newly promoted clubs, rare national teams, or teams with very little recent sample.

### Elo / ClubElo support

The historical validation already used internal Elo through:

```text
src/mundialytics/ratings/elo.py
```

The goal model now documents and exposes Elo features more clearly:

```text
team_elo
opponent_elo
elo_diff
```

The feature contract also supports optional external Elo / ClubElo columns:

```text
home_external_elo
away_external_elo
home_clubelo
away_clubelo
home_elo
away_elo
```

If present in canonical match data, these become:

```text
external_team_elo
external_opponent_elo
external_elo_diff
```

Internal Elo remains the default always-available feature.

## 4. Scope Not Yet Implemented

Corners/cards count models are documented as the next modelling slice, but this phase does not yet train a new negative-binomial corners/cards model.

Reason:

- goals/scoreline calibration needed to be stabilized first,
- corners/cards require coverage checks by league/source,
- Negative Binomial should be validated against Poisson and simple rolling-rate baselines before becoming default.

## 5. Outputs

For each model type, `run_historical_validation.py` now writes:

```text
statistical_engine_<model_type>_summary.json
statistical_engine_goal_errors_<model_type>.csv
statistical_engine_goal_lines_<model_type>.csv
statistical_engine_line_calibration_<model_type>.csv
statistical_engine_scorelines_<model_type>.csv
statistical_engine_calibration_layer_<model_type>.csv
statistical_engine_dixon_coles_scorelines_<model_type>.csv
```

The operational report stores these paths under:

```text
backtests.<model_type>.statistical_engine_evaluation
```

## 6. CLI Parameters Added

```text
--poisson-alpha
--time-decay-half-life-days
--rolling-shrinkage-prior-matches
```

Defaults:

```text
--poisson-alpha 1.0
--time-decay-half-life-days 365.0
--rolling-shrinkage-prior-matches 10.0
```

## 7. Acceptance Criteria

- [x] Existing v0.49.2 statistical evaluation still works.
- [x] Historical validation writes detailed line calibration diagnostics.
- [x] Historical validation writes calibration-layer diagnostics.
- [x] Historical validation writes Dixon-Coles scoreline diagnostics.
- [x] Goal model supports time decay sample weighting.
- [x] Team rolling features support low-sample shrinkage.
- [x] Internal Elo usage is documented as a core model feature.
- [x] Optional external Elo/ClubElo feature columns are supported.
- [x] Corners/cards Negative Binomial modelling is documented as the next count-model slice, not falsely claimed as complete.

## 8. Validation Performed

Focused validation:

```bash
python -m compileall -q src scripts/run_historical_validation.py
python -m pytest tests/test_v0492_statistical_engine_evaluation.py tests/test_match_value.py tests/test_value_backtest.py -q
```

Result:

```text
12 passed
```

The full suite was not run in this phase.

## 9. Known Limitations

- Calibration layer is currently diagnostic/offline; it does not yet replace production probabilities.
- Dixon-Coles rho is estimated from the backtest window and reported as a diagnostic.
- Negative Binomial corners/cards modelling remains planned.
- External ClubElo is supported if canonical match rows include the relevant columns, but no new downloader/provider flow was added in this phase.
