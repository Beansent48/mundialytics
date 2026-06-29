# v0.16 audit report

## Audit loop summary

The v0.16 work was reviewed through repeated implementation/audit passes.

### Round 1 — Static compatibility

- Ran `python -m compileall -q src scripts tests`.
- Result: pass.
- Focus: syntax/API compatibility after adding hierarchical calibration and cross-context player fields.

### Round 2 — Regression test suite

- Ran full suite with deterministic numerical thread caps:
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`.
- Result: `47 passed`.
- Reason for thread caps: local CI-style environment can hang after sklearn subprocess tests without BLAS thread limits. This is an environment/runtime issue, not a failed test assertion.

### Round 3 — Hierarchical calibration smoke

- Ran `scripts/calibrate_player_props.py` on the v09 sample with `--hierarchical`.
- Result: pass.
- Output generated:
  - `hierarchical_calibration_results.csv`
  - `hierarchical_calibrated_player_prop_predictions.csv`
  - `hierarchical_calibration_report.json`
- Sample report selected competition-level calibration where enough rows existed.

### Round 4 — Temporal hierarchical calibration smoke

- Ran `scripts/temporal_calibration_check.py` with `--hierarchical`.
- Result: pass.
- Output generated:
  - `hierarchical_temporal_calibration_results.csv`
  - `hierarchical_temporal_calibrated_predictions.csv`
  - `hierarchical_temporal_calibration_report.json`
- Purpose: verify that the anti-overfitting temporal check still works with the hierarchy.

### Round 5 — Safe lineup inference smoke

- Ran `scripts/run_safe_props_for_lineups.py` using the current-lineups template and sample historical events.
- Result: pass.
- Confirmed output fields:
  - `calibration_level`
  - `calibration_group_key`
  - `calibration_rows`
  - `club_minutes_sample`
  - `national_minutes_sample`
  - `cross_context_feature_used`
- Confirmed output is restricted to supplied lineup players.

## Issues found and handled

1. **Potential overfitting from per-competition models**
   - Decision: do not train fully separate league models by default.
   - Fix: hierarchical calibration with row/class thresholds and fallback.

2. **National-team props lacking current player evidence**
   - Fix: added `--feature-player-events` and club-to-national sample exposure.
   - Guard: feature rows are cut off before the first test date.

3. **Invisible club-to-national feature use**
   - Fix: predictions now include `club_minutes_sample`, `national_minutes_sample`, and `cross_context_feature_used`.

4. **Inference calibration too coarse**
   - Fix: safe lineup inference can now fit/select hierarchical calibrators when calibration predictions are provided.

5. **Audit schema did not require cross-context fields**
   - Fix: `audit_props_pipeline.py` now expects the cross-context sample fields and validates the cross-context flag.

## Remaining risks

- `player_id_global` is still only as good as the identity resolution source. If it is name-derived, two players with the same name can collide.
- Club-to-national transfer is useful but not always football-perfect: a player may have a different role for country than club.
- Competition-specific calibration should be used only when sample thresholds are met.
- Some markets remain more fragile than others: fouls and cards are especially sensitive to referee/competition behavior.
