# v0.49.2 — Statistical Engine Evaluation Foundation Spec

## Status

Implemented — documentation updated and historical validation now emits statistical-engine evaluation outputs for goals, goal lines, BTTS and scorelines.

## Date

2026-06-25

## 1. Context

Recent historical validation runs showed that the match-level Poisson baseline is useful as a probabilistic football model across national teams and several club datasets. The same conversation also clarified an important product boundary:

Mundialytics is primarily a statistical football engine and simulator. It is not a system whose main objective is to bet every 1X2 signal.

The value-pick layer remains a later, separate, selective research layer.

## 2. Problem

The existing historical validation mostly reported 1X2 probability metrics and optionally ran a broad 1X2 value backtest when odds were available. That optional value backtest can be useful diagnostically, but it should not drive model development for the simulator.

The core engine must instead be evaluated as a statistical model that predicts football distributions:

- result probabilities,
- expected goals,
- total-goals lines,
- BTTS,
- exact scoreline distribution,
- later corners, cards, shots, goalkeeper saves, player events, awards and tournament outcomes.

## 3. Objective

Reorient validation around statistical-engine quality and make this explicit in code and documentation.

The phase adds historical statistical evaluation outputs for:

- goal errors,
- total-goals line calibration,
- BTTS calibration,
- exact-score probability and top-k coverage.

It keeps model logic unchanged.

## 4. Current Behavior Before This Phase

`scripts/run_historical_validation.py` could run walk-forward validation for `poisson` and `random_forest_lambda`, emit 1X2 metrics and optionally run a historical value backtest if odds were supplied.

The optional value backtest was too easy to overinterpret as the main objective.

## 5. Desired Behavior

Historical validation should clearly separate:

```text
Statistical Engine Evaluation
→ measures predictive quality and distribution quality

Value Pick Evaluation
→ optional later layer for selective market opportunities
```

The operational validation report should now contain a `statistical_engine_evaluation` section for every model type.

## 6. Scope

Included:

- New reusable statistical-engine evaluation module.
- Goal-error metrics for lambdas vs actual goals.
- Total-goals over/under line calibration.
- BTTS yes/no calibration.
- Exact-score probability, rank and top-k coverage.
- Documentation clarifying that ROI/profit does not drive simulator model selection.
- Focused tests.
- A small defensive fix in the 1X2 value helper so full-season odds files can be used with shorter backtest prediction windows.

## 7. Non-Goals

Not included:

- No model training changes.
- No new features.
- No calibration changes.
- No new APIs.
- No live betting.
- No staking logic.
- No player-prop model changes.
- No corners/cards/player-event modelling changes yet.
- No dashboard rewrite.

## 8. Inputs

The new statistical evaluation consumes walk-forward prediction rows with:

```text
match_id
home_goals
away_goals
lambda_home
lambda_away
p_home_win
p_draw
p_away_win
```

Optional descriptive columns:

```text
date
competition
home_team
away_team
most_likely_score
```

## 9. Outputs

For each model type in `run_historical_validation.py`:

```text
statistical_engine_<model_type>_summary.json
statistical_engine_goal_errors_<model_type>.csv
statistical_engine_goal_lines_<model_type>.csv
statistical_engine_scorelines_<model_type>.csv
```

The operational report includes:

```text
backtests.<model_type>.statistical_engine_evaluation
```

## 10. Constraints

- Offline only.
- No external API calls.
- No new dependencies.
- Missing/unsafe data must not be invented.
- ROI/profit must not be used for statistical model selection.
- Club and national-team models remain separate.

## 11. Architecture Impact

New files:

```text
src/mundialytics/evaluation/statistical_engine.py
tests/test_v0492_statistical_engine_evaluation.py
docs/V0492_STATISTICAL_ENGINE_EVALUATION_SPEC.md
```

Updated files:

```text
scripts/run_historical_validation.py
src/mundialytics/reports/match_value.py
tests/test_match_value.py
README.md
CHANGELOG.md
docs/PROJECT_CONTINUITY.md
docs/DECISIONS.md
docs/NEXT_VALIDATION_STEPS.md
docs/MODEL_DESIGN.md
pyproject.toml
src/mundialytics/__init__.py
```

## 12. Edge Cases

- Full-season odds file contains fixtures outside the backtest prediction window.
- Odds files contain descriptive columns that overlap with prediction columns.
- Actual high scorelines exceed the default scoreline grid.
- Backtest predictions exist but no odds are supplied.
- Value backtest exists but should not be interpreted as core engine validation.

## 13. Acceptance Criteria

- [x] Historical validation still emits normal 1X2 backtest outputs.
- [x] Historical validation emits statistical-engine outputs for each model.
- [x] Statistical evaluation includes goals, total-goals lines, BTTS and scorelines.
- [x] Documentation states that the statistical engine and value-pick engine are separate layers.
- [x] ROI/profit is not used for model ranking.
- [x] Focused tests validate the new outputs.
- [x] Existing value helper handles prediction-window vs full-season odds mismatch.

## 14. Validation Plan

Automated validation:

```bash
python -m compileall -q src scripts/run_historical_validation.py tests/test_v0492_statistical_engine_evaluation.py tests/test_match_value.py tests/test_value_backtest.py
python -m pytest tests/test_v0492_statistical_engine_evaluation.py tests/test_match_value.py tests/test_value_backtest.py -q
```

Manual validation:

- Run national historical validation.
- Run club historical validation.
- Confirm `statistical_engine_evaluation` exists in `operational_validation_report.json`.
- Confirm outputs are used for statistical interpretation, not profit claims.

## 15. Documentation Updates

Updated:

- `README.md`
- `CHANGELOG.md`
- `docs/PROJECT_CONTINUITY.md`
- `docs/DECISIONS.md`
- `docs/NEXT_VALIDATION_STEPS.md`
- `docs/MODEL_DESIGN.md`

## 16. Risks

- Users may still overinterpret optional value backtests if odds are supplied.
- Statistical evaluation currently covers goals/scorelines/BTTS, not corners/cards/player props yet.
- Retrospective backtests still require temporal care; forward evaluation remains the cleanest evidence path.

## 17. Rollback Plan

Revert:

```text
src/mundialytics/evaluation/statistical_engine.py
tests/test_v0492_statistical_engine_evaluation.py
```

and remove the `evaluate_statistical_engine` integration from `scripts/run_historical_validation.py`.

The underlying model, adapters and existing 1X2 validation remain unchanged.
