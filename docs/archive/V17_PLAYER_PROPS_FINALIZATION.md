# v0.17 Player Props Finalization

This version closes the player-props layer for paper-mode use before moving to team/match stats.

## What changed

### 1. Adaptive hierarchical calibration

Previous v0.16 hierarchy used a fixed narrowest-available selection. In practice, a high `min_hierarchical_group_rows` could make every club-men prediction fall back to `domain_context`, and a low threshold could force competition-level calibration even when the broader parent was safer.

v0.17 adds `selection_mode="adaptive"`:

1. Fit eligible calibration groups for each level:
   - `competition`
   - `domain_context`
   - `team_type_gender`
   - `market_global`
2. Evaluate each candidate on the temporal validation split.
3. Select the available level with the best validation score for that row, using log loss plus a small bias penalty.

This lets league/competition calibration add value without blindly overfitting.

### 2. Competition diagnostics

The hierarchical report now includes `competition_diagnostics`, with raw and calibrated metrics by `market_type + competition`. This is meant to reveal whether a market problem comes from a specific league or from the whole domain.

### 3. Final market policy

New script:

```powershell
python scripts/finalize_player_props_policy.py `
  --temporal-report outputs/<run>/calibration_temporal_check/temporal_calibration_report.json `
  --hierarchical-report outputs/<run>/calibration_temporal_check/hierarchical_temporal_calibration_report.json `
  --out outputs/<run>/player_props_policy.json
```

The policy decides, by market, whether operational inference should use:

- simple market-level calibration, or
- hierarchical/adaptive calibration.

It also stores:

- readiness status,
- reason for the decision,
- chosen metrics,
- safe probability caps/floors.

### 4. Safe inference can use the policy

`run_safe_props_for_lineups.py` now accepts:

```powershell
--calibration-policy outputs/<run>/player_props_policy.json
```

The output includes:

- `calibration_policy_source`
- `calibration_policy_status`
- `calibration_policy_reason`

So matchday reports can explain whether each market used simple or hierarchical calibration.

## Defaults changed

`min_hierarchical_group_rows` now defaults to `200` instead of `800`, because earlier validation showed that `800` was too restrictive for competition-level diagnostics on available StatsBomb Open Data.

## What did not change

No `match_importance` feature was added. The model still uses objective labels only:

- `team_type`
- `team_scope`
- `competition_context`
- `gender`
- `competition`

## Validation

Local audit run:

```text
compileall OK
50 tests passed
```
