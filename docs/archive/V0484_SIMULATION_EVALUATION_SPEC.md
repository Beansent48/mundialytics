# Feature / Change Spec: v0.48.4 — Simulation Evaluation Report

## 1. Context

Mundialytics already generates simulator-first outputs: match probabilities, scoreline distributions, dynamic market-style lines, matchday rankings and tournament reports. The next professional step is to evaluate those predictions against known results before changing models or investing in a larger visual interface.

This version keeps the engine offline and conservative. It does not retrain models, change calibration, add APIs, create picks or claim betting profitability.

## 2. Problem

The simulator can generate rich predictions, but the project needs an auditable way to answer:

- Are 1X2 probabilities accurate and calibrated?
- Does the simulator improve over simple baselines?
- Are expected goals close to actual goals?
- Do the top scoreline probabilities cover real outcomes?
- Which dynamic goal lines have observable signal when actual results exist?
- Which metrics are not available because the required data is missing?

## 3. Objective

Add a separate offline evaluation workflow that compares already-generated simulator outputs with actual match results and produces CSV, JSON and HTML reports.

## 4. Current Behavior

`run_statistical_matchday.py` generates predictions and simulator reports, but there is no separate evaluation command for comparing those predictions against real outcomes.

## 5. Desired Behavior

A new evaluation command should accept prediction outputs plus actual results and generate:

- `simulation_metrics.json`
- `simulation_evaluation.csv`
- `calibration_1x2.csv`
- `goal_error_metrics.csv`
- `scoreline_evaluation.csv`
- `baseline_comparison.csv`
- `line_evaluation.csv`
- `simulation_evaluation_report.html`

If actual results are missing, the workflow must complete and mark the evaluation as `not_available` instead of inventing metrics.

## 6. Scope

Included:

- Offline evaluation script.
- Evaluation module for 1X2, goals, scorelines and dynamic goal lines.
- Simple diagnostic baselines.
- Sample smoke actuals clearly marked as non-real sample data.
- HTML evaluation report.
- Data requirements guide for the next data-foundation phase.
- Focused tests.

## 7. Non-Goals

Not included:

- No model retraining.
- No model calibration changes.
- No OddsPapi/API calls.
- No live betting or execution.
- No ROI/yield/CLV evaluation.
- No deep player-prop evaluation.
- No visual dashboard rewrite.
- No data-audit phase yet.

## 8. Inputs

Required:

- `match_predictions.csv`

Optional but recommended:

- `actual_results.csv`
- `scoreline_distribution.csv`
- `dynamic_market_lines.csv`

Minimum actual results schema:

```text
match_id
date
competition
home_team
away_team
home_goals
away_goals
status
```

Accepted aliases:

```text
fixture_id -> match_id
home_score -> home_goals
away_score -> away_goals
home_goals_actual -> home_goals
away_goals_actual -> away_goals
```

## 9. Outputs

```text
simulation_metrics.json
simulation_evaluation.csv
calibration_1x2.csv
goal_error_metrics.csv
scoreline_evaluation.csv
baseline_comparison.csv
line_evaluation.csv
simulation_evaluation_report.html
```

## 10. Constraints

- Offline only.
- No external APIs.
- No new dependencies.
- No model behavior changes.
- Missing data must be represented as `not_available`.
- Evaluation mode must be explicitly labelled:
  - `sample_smoke_evaluation`
  - `retrospective_backtest`
  - `forward_evaluation`

## 11. Architecture Impact

New files:

```text
src/mundialytics/statistical_core/simulation_evaluation.py
scripts/run_simulation_evaluation.py
docs/V0484_SIMULATION_EVALUATION_SPEC.md
docs/NEXT_DATA_FOUNDATION_REQUIREMENTS.md
data/sample/sample_actual_results_for_evaluation.csv
tests/test_v0484_simulation_evaluation.py
```

Updated files:

```text
pyproject.toml
src/mundialytics/__init__.py
src/mundialytics/statistical_core/__init__.py
README.md
CHANGELOG.md
```

## 12. Edge Cases

- Predictions exist but actual results are missing.
- Actual results exist but `match_id` does not match predictions.
- Scoreline distribution is missing.
- Dynamic lines are missing.
- Actual scoreline falls outside the generated scoreline grid.
- Very small evaluation sample creates unreliable calibration bins.
- Retrospective predictions may not be forward-safe.

## 13. Acceptance Criteria

- [x] The new command runs offline.
- [x] Missing actual results produce `status = not_available`.
- [x] Sample actual results produce `status = completed`.
- [x] Outputs are written as CSV/JSON/HTML.
- [x] Baseline comparison is generated.
- [x] Calibration bins are generated.
- [x] The report states that this is not betting performance.
- [x] The data requirements guide documents what is needed next.

## 14. Validation Plan

Automated validation:

```bash
python -m compileall -q src scripts/run_simulation_evaluation.py tests/test_v0484_simulation_evaluation.py
pytest tests/test_v0484_simulation_evaluation.py -q
```

Smoke flow:

```bash
python scripts/run_statistical_matchday.py \
  --fixtures data/input/fixtures.csv \
  --lineups data/input/current_lineups.csv \
  --squads data/input/squads.csv \
  --odds data/input/odds.csv \
  --tournament-config data/input/tournament_config.csv \
  --historical-events data/sample/sample_player_events.csv \
  --out-dir outputs/smoke_matchday \
  --n-simulations 50 \
  --seed 42 \
  --clean-out-dir

python scripts/run_simulation_evaluation.py \
  --predictions outputs/smoke_matchday/match_predictions.csv \
  --actual-results data/sample/sample_actual_results_for_evaluation.csv \
  --scorelines outputs/smoke_matchday/scoreline_distribution.csv \
  --dynamic-lines outputs/smoke_matchday/dynamic_market_lines.csv \
  --out-dir outputs/smoke_evaluation \
  --evaluation-mode sample_smoke_evaluation \
  --clean-out-dir
```

## 15. Documentation Updates

- README updated with evaluation workflow.
- CHANGELOG updated.
- Next data phase guide added.

## 16. Risks

- Small sample metrics are not meaningful for model selection.
- Retrospective evaluations can suffer leakage if predictions were generated using future information.
- The empirical-frequency baseline is diagnostic/in-sample unless supplied from a separate training window.
- Player props and Golden Boot modelling require much stronger player/minute/squad data before deep evaluation.

## 17. Rollback Plan

Remove the new evaluation module/script/docs/test and restore version metadata to the previous release.
