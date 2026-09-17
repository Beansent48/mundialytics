# v0.49.0 — Data Audit Report Spec

## 1. Context

Mundialytics now has a simulator-first pipeline, advanced reports, tournament Monte Carlo outputs and an offline simulation evaluation layer. The next risk is data quality: model quality depends on whether fixtures, results, lineups, squads, player events, odds snapshots and prediction snapshots are complete, temporally valid and entity-safe.

## 2. Problem

The project needs an offline audit that shows what data is present, what is missing, what is low quality and which future features must remain `not_available` until reliable data exists.

## 3. Objective

Add a conservative, offline data-audit layer that inspects supplied CSV files and generates machine-readable and human-readable reports without changing model logic.

## 4. Current Behavior

The project can generate predictions and evaluate them against actual results, but there is no single audit entrypoint that summarizes dataset schemas, coverage, gaps, entity issues and feature availability.

## 5. Desired Behavior

A user can run `scripts/run_data_audit.py` with local CSV inputs and get:

- dataset schema/health report
- coverage report
- data gaps report
- entity quality report
- feature availability matrix
- next data requirements guide
- HTML summary report

Missing datasets must be marked as `not_available`, not fabricated.

## 6. Scope

Included:

- Offline CSV audit only.
- Dataset-level schema checks.
- Coverage checks against supplied fixtures.
- Basic team/player entity guardrails.
- Conservative player-prop and Golden Boot future requirements.
- Outputs in CSV, JSON and HTML.
- Focused smoke test.

## 7. Non-Goals

Not included:

- No model changes.
- No retraining.
- No new calibration.
- No external API calls.
- No OddsPapi/Betfair work.
- No live betting or picks.
- No dashboard/UI framework.
- No deep player-prop evaluation yet.

## 8. Inputs

Optional CSV files:

- fixtures
- actual_results
- lineups
- squads
- player_events
- odds
- predictions
- scorelines
- dynamic_lines
- matchday_summary
- tournament_simulation
- tournament_report

## 9. Outputs

Generated files:

- `data_audit_summary.json`
- `data_audit_report.csv`
- `coverage_report.csv`
- `data_gaps_report.csv`
- `entity_quality_report.csv`
- `feature_availability_matrix.csv`
- `next_data_requirements.csv`
- `data_audit_report.html`

## 10. Constraints

- Offline only.
- No invented data.
- No API calls.
- No model behavior changes.
- Player props remain conservative unless current squads/lineups and historical player events are available.
- Historical data may be used for training/evaluation, but current player inference needs current eligibility evidence.

## 11. Architecture Impact

New module:

- `src/mundialytics/data_quality/data_audit.py`

New script:

- `scripts/run_data_audit.py`

New test:

- `tests/test_v0490_data_audit.py`

## 12. Edge Cases

- Missing fixtures.
- Missing actual results.
- Empty CSV.
- Missing required columns.
- Duplicate keys.
- Lineup player not found in current squads.
- Historical teams/players appearing in training data but not current fixtures.
- Optional odds absent.

## 13. Acceptance Criteria

- [x] Audit runs offline from local CSV files.
- [x] Missing datasets are marked `not_available`.
- [x] Outputs include JSON, CSV and HTML.
- [x] Coverage and entity-quality reports are generated.
- [x] Feature availability matrix is generated.
- [x] Golden Boot/player-prop future data requirements are documented.
- [x] Focused test passes.
- [x] Existing simulation/evaluation scripts remain separate.

## 14. Validation Plan

Automated validation:

```bash
python -m compileall -q src scripts/run_data_audit.py tests/test_v0490_data_audit.py
pytest tests/test_v0490_data_audit.py -q
```

Manual smoke test:

```powershell
python scripts/run_data_audit.py `
  --fixtures data/input/fixtures.csv `
  --actual-results data/sample/sample_actual_results_for_evaluation.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --player-events data/sample/sample_player_events.csv `
  --odds data/input/odds.csv `
  --out-dir outputs/data_audit_current `
  --run-label sample_data_audit `
  --clean-out-dir
```

## 15. Documentation Updates

- README updated with the v0.49.0 audit command.
- CHANGELOG updated.
- `docs/NEXT_DATA_FOUNDATION_REQUIREMENTS.md` remains the broader guide for the next data collection phase.

## 16. Risks

- This is a basic audit, not a full data warehouse validation suite.
- Provider identity resolution remains a later phase.
- Coverage checks depend on stable `match_id` joins.
- Historical player data is not enough for current player-prop inference.

## 17. Rollback Plan

Remove:

- `src/mundialytics/data_quality/`
- `scripts/run_data_audit.py`
- `tests/test_v0490_data_audit.py`
- this spec and changelog/README edits
