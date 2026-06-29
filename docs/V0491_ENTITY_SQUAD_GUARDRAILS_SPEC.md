# v0.49.1 — Entity & Squad Guardrails Spec

## Status

Implemented — v0.49.1 guardrail layer added and focused validation completed.

## Date

2026-06-25

## 1. Context

v0.49.0 introduced an offline data audit that reports dataset schemas, coverage, gaps, entity quality and feature availability. The next risk is stricter identity and squad safety before current player-prop inference, tournament award modelling or higher-confidence reporting.

The project already treats data quality conservatively: missing or unsafe data should become `not_available`, `warning`, `blocked` or `needs_review`, not fabricated model confidence.

## 2. Problem

The engine needs stronger guardrails around:

- stable match identity across datasets,
- provider fixture/team/player IDs,
- team scope compatibility,
- current squad eligibility,
- lineup validity,
- historical-only player leakage into current predictions,
- ambiguous team/player names,
- unsafe joins between fixtures, events, odds and predictions.

Without these guardrails, future player props and tournament-player features can look valid while using stale, mismatched or context-leaking entities.

## 3. Objective

Add a conservative offline guardrail layer that detects and reports unsafe entity and squad conditions before downstream inference or evaluation treats the data as reliable.

The phase should make the project safer for future features, not more aggressive.

## 4. Current Behavior

v0.49.0 can generate an entity-quality report as part of the audit, but it does not yet provide a dedicated, stricter v0.49.1 guardrail contract for provider identity consistency and current squad eligibility.

## 5. Desired Behavior

A user can run the existing data-audit flow, or a dedicated guardrail path if needed, and receive clear outputs showing whether the supplied datasets are safe for:

- match-level prediction/evaluation,
- team-level event modelling,
- current player-prop candidate generation,
- tournament-player features such as top scorer/Golden Boot in a later phase.

Unsafe player or match records must be flagged with explicit reason codes.

## 6. Scope

Included:

- Stable `match_id` checks across supplied datasets.
- Provider ID consistency checks where provider columns exist.
- Team scope guardrails for `club` vs `national`.
- Current squad eligibility checks for lineup/player-prop candidates.
- Historical-only player detection for current inference.
- Duplicate or ambiguous team/player identity warnings.
- Conservative reason-code outputs.
- Focused tests for guardrail behavior.
- Documentation updates.

## 7. Non-Goals

Not included:

- No model changes.
- No new training logic.
- No new external API calls.
- No automatic provider switch.
- No live betting/trading.
- No dashboard.
- No Golden Boot model implementation.
- No broad repository restructure.
- No new dependencies unless approved separately.

## 8. Inputs

Likely existing CSV inputs:

```text
fixtures
actual_results
lineups
squads
player_events
odds
predictions
scorelines
dynamic_lines
matchday_summary
tournament_simulation
tournament_report
```

Important optional columns:

```text
match_id
provider
provider_fixture_id
provider_team_id
provider_player_id
team
home_team
away_team
player
player_id
team_scope
team_type
competition_context
current_squad_flag
lineup_status
availability_status
expected_minutes
kickoff_utc
generated_at_utc
```

## 9. Outputs

Implemented outputs:

```text
entity_guardrails_report.csv
squad_guardrails_report.csv
guardrail_summary.json
```

Possible additions to existing v0.49.0 outputs:

```text
entity_quality_report.csv
feature_availability_matrix.csv
data_audit_summary.json
data_audit_report.html
```

Reason-code examples:

```text
missing_match_id
duplicate_match_id
provider_fixture_id_conflict
team_scope_mismatch
team_not_in_fixture
player_not_in_current_squad
lineup_player_without_player_id
historical_only_player_for_current_inference
ambiguous_player_name
ambiguous_team_name
missing_current_eligibility
unsafe_for_player_props
unsafe_for_forward_evaluation
```

## 10. Constraints

- Offline first.
- No invented data.
- No API calls.
- No model behavior changes.
- No betting or live actions.
- Preserve existing v0.49.0 audit behavior.
- Prefer warnings and blocked statuses over silent filtering.
- Guardrails should be deterministic and testable.
- Use existing dependencies unless a separate decision approves otherwise.

## 11. Architecture Impact

Affected areas:

```text
src/mundialytics/data_quality/data_audit.py
src/mundialytics/data_quality/entity_guardrails.py
src/mundialytics/data_quality/__init__.py
scripts/run_data_audit.py
tests/test_v0490_data_audit.py
tests/test_v0491_entity_squad_guardrails.py
README.md
CHANGELOG.md
docs/PROJECT_CONTINUITY.md
```

Implementation decision:

- Keep the existing data-audit entrypoint.
- Add a focused pure guardrail module instead of growing `data_audit.py` further.
- Do not restructure the wider project.
- Do not add dependencies.

## 12. Edge Cases

- Fixture exists without `match_id`.
- Same `match_id` maps to multiple fixtures.
- Same provider fixture ID maps to multiple `match_id` values.
- Club and national records appear in the same current inference input.
- Lineup team is neither fixture home nor away team.
- Player appears in lineup but not in current squad.
- Player appears in historical events but not in current eligibility input.
- Same player name appears for multiple teams without stable `player_id`.
- Current squad lacks `current_squad_flag` or `availability_status`.
- Predictions have `generated_at_utc >= kickoff_utc`.
- Odds snapshot lacks timestamp or line identity.
- Player has valid global identity but invalid current context.

## 13. Acceptance Criteria

Implementation is complete only when:

- [x] v0.49.1 guardrail spec is implemented or updated to match the final implementation.
- [x] Stable `match_id` checks are performed for relevant supplied datasets.
- [x] Team scope mismatches are reported.
- [x] Lineup players not present in current squads are flagged.
- [x] Historical-only players are blocked or marked unsafe for current player-prop inference.
- [x] Guardrail outputs contain explicit reason codes.
- [x] Existing v0.49.0 audit behavior is preserved.
- [x] Focused tests cover safe and unsafe cases.
- [x] Smoke command generates expected reports.
- [x] README/CHANGELOG are updated only after implementation is validated.

## 14. Validation Plan

Automated validation completed:

```bash
python -m compileall -q src scripts/run_data_audit.py tests/test_v0490_data_audit.py tests/test_v0491_entity_squad_guardrails.py
python -m pytest tests/test_v0490_data_audit.py tests/test_v0491_entity_squad_guardrails.py -q
```

Focused result:

```text
2 passed
```

Smoke validation completed:

```powershell
python scripts/run_data_audit.py `
  --fixtures data/input/fixtures.csv `
  --actual-results data/sample/sample_actual_results_for_evaluation.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --player-events data/sample/sample_player_events.csv `
  --odds data/input/odds.csv `
  --out-dir outputs/data_audit_v0491_guardrails `
  --run-label sample_data_audit_v0491_guardrails `
  --clean-out-dir
```

Manual checks:

- Confirm unsafe rows are visible in CSV outputs.
- Confirm HTML report surfaces guardrail warnings.
- Confirm no player-prop candidate is treated as safe without current squad evidence.
- Confirm missing optional datasets are still reported as `not_available`, not errors.

## 15. Documentation Updates

Updated after implementation:

- [x] `README.md`
- [x] `CHANGELOG.md`
- [x] `docs/PROJECT_CONTINUITY.md`
- [ ] `docs/DECISIONS.md`, if scope changes
- [x] this spec

## 16. Risks

- Over-blocking valid players when source data lacks IDs.
- Under-blocking ambiguous names when IDs are absent.
- Scope creep into provider ingestion or player-prop modelling.
- Duplicating v0.49.0 audit logic instead of extending it cleanly.
- Treating historical data as current eligibility evidence.

## 17. Rollback Plan

Because this spec is documentation-only until implementation, rollback is safe:

- remove or revise `docs/V0491_ENTITY_SQUAD_GUARDRAILS_SPEC.md`;
- remove related entries from `docs/PROJECT_CONTINUITY.md` and `docs/DECISIONS.md`.

After implementation, rollback should remove only the v0.49.1 guardrail code/tests and restore v0.49.0 audit behavior.


## 18. Implementation Summary

Implemented in v0.49.1:

- Added `src/mundialytics/data_quality/entity_guardrails.py`.
- Integrated guardrails into `audit_data_sources` and `write_data_audit_outputs`.
- Added CSV outputs for entity and squad guardrails.
- Added `guardrail_summary.json`.
- Added guardrail sections to the HTML audit report.
- Added focused regression coverage in `tests/test_v0491_entity_squad_guardrails.py`.

Validation completed locally:

```text
compileall: passed
pytest focused v0.49.0 + v0.49.1: 2 passed
smoke data audit: generated 11 outputs
```
