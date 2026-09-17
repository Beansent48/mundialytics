# v0.17 Audit Report

Scope: finalize player props for paper mode before starting team/match stat props.

## Audit loop 1 — code integration

- Added adaptive hierarchical calibration selection.
- Added policy generator.
- Added policy-aware safe lineup inference.
- Checked compile of changed modules.

Result: passed.

## Audit loop 2 — regression tests

Command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m pytest -q
```

Result:

```text
50 passed
```

## Audit loop 3 — adaptive hierarchy logic

Concern: v0.16 could fail to use `competition` when thresholds were too high, or force it when too low.

Fix: `selection_mode="adaptive"` compares eligible levels using validation score. It can choose competition, domain context, team-type/gender, or market global per row.

Result: synthetic test verifies diagnostics and eligible-level selection.

## Audit loop 4 — policy logic

Concern: hierarchical calibration should not be used blindly if simple market-level calibration performs better.

Fix: `finalize_player_props_policy.py` chooses simple vs hierarchical per market using log loss, Brier and bias heuristics.

Result: policy smoke test produced a mixed policy: simple for one market and hierarchical for another.

## Audit loop 5 — operational inference contract

Concern: safe matchday inference needs to explain which calibration source was used and must keep caps/floors.

Fix: `run_safe_props_for_lineups.py` supports `--calibration-policy`. Output includes policy source/status/reason.

Result: test verifies `calibration_policy_source`, `calibration_policy_status` and policy-controlled simple-market fallback.

## Remaining limitations

- This version does not add team/match stat props yet.
- It does not claim live betting readiness.
- It does not add subjective `match_importance`.
- League-level calibration remains data-limited in open-data samples; adaptive selection is designed to avoid overusing it.
- Referee-level features are still missing, so cards/fouls remain caution markets.
