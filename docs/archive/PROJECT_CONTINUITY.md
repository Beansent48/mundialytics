# Project Continuity Handoff

## Purpose

This document is the first file to read when continuing the project in a new chat.

The user wants important decisions, planned phases and implementation status to be saved inside the repository documentation even before code is implemented. This makes the project portable: a future chat should be able to receive the ZIP and continue without relying on hidden conversation history.

## Current Repository State

Current package version in `pyproject.toml`:

```text
0.50.0
```

Latest implemented phase:

```text
v0.50.0 — Advanced Football Data Acquisition Layer
```

Latest implemented documentation/spec:

```text
docs/V0500_ADVANCED_FOOTBALL_DATA_LAYER_SPEC.md
```

Latest proposed next phase:

```text
v0.50.1 — Advanced Feature Snapshot Validation & Hybrid Model Evaluation
```

Proposed next-phase guide:

```text
docs/NEXT_DATA_FOUNDATION_REQUIREMENTS.md
```

Decision log:

```text
docs/DECISIONS.md
```

## User Continuity Preference

Accepted on `2026-06-25`:

- Keep planning decisions in `docs/` even if they are not implemented yet.
- Mark planned work clearly as `Proposed`, `Accepted`, `Implemented` or `Deferred`.
- Avoid relying on prior chat memory.
- Keep enough context in the ZIP so a new chat can continue from the repository alone.
- Do not silently mix unrelated projects.
- Do not update package version or changelog as if a feature is implemented before code and validation exist.



## Latest v0.49.5 Planning Decisions

Accepted on `2026-06-25`:

```text
Club model direction:
global Big 5 model
+ league features
+ league-level diagnostics/calibration
+ team rolling features
+ internal Elo
+ optional ClubElo/external Elo
+ optional xG/event enrichments
```

The user explicitly prefers the hybrid approach because broad training data adds value, while league context must still be visible in analysis.

Data-source levels:

```text
Level 1: goals + shots + shots on target + corners + fouls/cards + internal Elo
Level 2: Level 1 + ClubElo/external Elo
Level 3: Level 2 + xG/xA/event data
```

xG should be pursued, but it must be optional so the engine remains usable when xG coverage is absent.

Club/national usage:

- Use club data heavily for player/event evidence, expected minutes, player form and award/player-prop modelling.
- Use national results, national Elo and national context for national-team match outcomes.
- Do not mix club and national rows in one base model unless a later explicit cross-context model is designed and validated.

New data contract:

```text
canonical_matches.csv
→ model_ready_match_snapshots.csv
→ model_ready_feature_contract.csv
→ model_ready_snapshot_report.json
```

The model may consume only columns marked as pre-match features. Targets are included for training/evaluation but must not be passed as input features.

## Core Product Rule

Do not mix club and national-team contexts in the same model or live inference path.

Relevant fields and concepts:

```text
team_scope
team_type
competition_context
match_id
team_id
player_id
provider
provider_fixture_id
current_squad_flag
lineup_status
availability_status
```

## Current Implemented Capability Summary

Implemented before this handoff:

- v0.49.5 model-ready snapshot builder for leakage-safe hybrid Big 5 data contracts.
- v0.49.5 `model_ready_feature_contract.csv` that separates identity, pre-match features and post-match targets.
- v0.49.5 accepted architecture: global Big 5 club model with league/context features and by-league diagnostics/calibration.
- v0.49.5 xG policy: optional enrichment, never required for baseline operation.
- v0.49.5 club/national policy: club data is strongest for player/event evidence; national results/Elo remain primary for national-team outcomes.

- v0.49.4 data-foundation builder for cleaned canonical multi-file match datasets.
- v0.49.4 feature-coverage, competition/season quality, anomaly and dropped-row reports.
- Historical validation now writes a `data_foundation` section before backtesting.
- Simulator-first match prediction/reporting pipeline.
- Offline simulation evaluation against actual results when available.
- v0.49.0 offline data audit for local CSVs.
- v0.49.1 entity and squad guardrails integrated into the data-audit flow.
- Data audit outputs for schema, coverage, gaps, entity quality, feature availability, next requirements, entity guardrails and squad guardrails.
- Conservative `not_available`, `warning`, `blocked`, `needs_review` and `unsafe_for_current_player_props` behavior when required data is missing or unsafe.

## Latest Local Validation Notes

From the reconstructed ZIP state, the focused v0.49.0 + v0.49.1 path was validated with:

```bash
python -m compileall -q src scripts/run_data_audit.py tests/test_v0490_data_audit.py tests/test_v0491_entity_squad_guardrails.py
python -m pytest tests/test_v0490_data_audit.py tests/test_v0491_entity_squad_guardrails.py -q
```

Focused result:

```text
2 passed
```

The data-audit smoke command generated the expected 11 outputs and returned status:

```text
v0.49.0_data_audit_report: status=warning outputs=11
```

Guardrail smoke summary:

```text
guardrail version: v0.49.1_entity_squad_guardrails
guardrail status: warning
reason codes: historical_only_player_for_current_inference, lineup_player_without_player_id, match_id_not_in_fixtures, team_not_in_fixture
```

Known warning reasons:

```text
low_coverage
predictions_not_available
historical_only_player_for_current_inference
lineup_player_without_player_id
match_id_not_in_fixtures
team_not_in_fixture
```

The full test suite was **not** confirmed in this handoff because only the focused data-audit/guardrail path was executed.

## Recommended Next Step

Start `v0.49.5 — Model Selection on Data Foundation / Corners Cards Count Models` after validating v0.49.4 outputs across multiple leagues/seasons.

Recommended first implementation slice:

1. Keep v0.49.1 guardrails unchanged as the safety contract.
2. Choose one real source path to normalize first, without adding model changes.
3. Produce stable `match_id`, `team_scope`, provider fixture/team/player IDs and current eligibility columns.
4. Run the v0.49.1 audit against that normalized data.
5. Only then consider player-prop or tournament-player feature expansion.

## Important Non-Goals For Next Phase

Do not implement these as part of the next dataset-foundation slice unless a later decision changes scope:

- No new model training.
- No betting execution.
- No live Betfair trading.
- No external API provider switch.
- No dashboard framework.
- No Golden Boot model.
- No new dependencies unless explicitly justified.
- No version bump beyond `0.49.1` until the next implementation and validation are complete.

## Files A Future Chat Should Read First

```text
README.md
CHANGELOG.md
docs/PROJECT_CONTINUITY.md
docs/DECISIONS.md
docs/V0490_DATA_AUDIT_REPORT_SPEC.md
docs/V0491_ENTITY_SQUAD_GUARDRAILS_SPEC.md
docs/V0494_DATA_FOUNDATION_SPEC.md
docs/NEXT_DATA_FOUNDATION_REQUIREMENTS.md
docs/NEXT_VALIDATION_STEPS.md
```

## Suggested Opening Prompt For Future Chats

```text
This ZIP is the current state of Mundialytics.
Please read docs/PROJECT_CONTINUITY.md, docs/DECISIONS.md and docs/V0491_ENTITY_SQUAD_GUARDRAILS_SPEC.md first.
Do not implement broad changes yet.
Reconstruct the current state and recommend the smallest safe next step.
```


## v0.49.2 Product Direction Clarification

Accepted on `2026-06-25`:

Mundialytics must be developed as a **statistical football engine first**, not as a broad 1X2 betting bot.

Separate layers:

```text
1. Statistical Engine
   Purpose: predict football outcomes and distributions.
   Examples: 1X2, goals, exact scorelines, BTTS, corners, cards, shots,
   player events, tournament simulations and individual awards.

2. Value Pick Engine
   Purpose: later selective market research for a small number of high-quality opportunities.
   Examples: under 10.5 corners, card lines, player shots, team totals or other market-specific edges.
```

Rules:

- Statistical model development is driven by log loss, RPS, Brier, calibration, MAE/RMSE, line calibration and scoreline diagnostics.
- Profit/ROI must not drive simulator model selection.
- Optional value backtests are diagnostic only until a separate value-pick validation phase exists.
- Player props and awards require current squad/lineup eligibility and careful club-to-national context handling.

## Latest v0.49.2 Implemented Capability Summary

Implemented in v0.49.2:

- New statistical-engine evaluation module for historical validation.
- Historical validation now writes goal-error, total-goals line, BTTS and scoreline diagnostics for each model type.
- Operational reports include `backtests.<model_type>.statistical_engine_evaluation`.
- Defensive 1X2 value-helper fixes for full-season odds files with shorter backtest windows.
- Documentation updated to separate the statistical engine from the value-pick engine.

## Latest v0.49.2 Local Validation Notes

Focused validation completed in the build handoff:

```bash
python -m compileall -q src scripts/run_historical_validation.py tests/test_v0492_statistical_engine_evaluation.py tests/test_match_value.py tests/test_value_backtest.py
python -m pytest tests/test_v0492_statistical_engine_evaluation.py tests/test_match_value.py tests/test_value_backtest.py -q
```

The full suite was not rerun in this handoff.

User-run historical validations during planning showed that Poisson outperformed `random_forest_lambda` on the main 1X2 metrics for:

```text
national teams
EPL 2025/26
EPL 2024/25
LaLiga 2024/25
Serie A 2024/25
```

Interpretation:

```text
Poisson is the current baseline statistical engine.
RandomForest lambda remains a secondary experiment.
The value-pick layer is not validated and should not guide simulator development.
```

## Recommended Next Step After v0.49.5

Build and compare Big 5 model-ready snapshots:

```text
foundation_laliga_multi_season/canonical_matches.csv
foundation_epl_multi_season/canonical_matches.csv
foundation_seriea_multi_season/canonical_matches.csv
foundation_bundesliga_multi_season/canonical_matches.csv
foundation_ligue1_multi_season/canonical_matches.csv
→ model_ready_*_v0495/model_ready_match_snapshots.csv
```

Then evaluate:

```text
global Big 5 baseline
vs by-league diagnostics
vs optional league-level calibration
```

Do not open corners/cards models until the snapshot contract is confirmed across Big 5. Keep value picks as a later, separate and selective layer.


## v0.49.4 Continuity Note — Statistical model improvements

Implemented in v0.49.4:

- Detailed calibration diagnostics for total-goals lines and BTTS.
- Offline calibration-layer diagnostics for 1X2, totals and BTTS.
- Dixon-Coles low-score scoreline diagnostics.
- Time-decay sample weighting for the goal-lambda model.
- Low-sample shrinkage for rolling team features.
- Explicit documentation that Elo is a core model feature.
- Optional external Elo/ClubElo feature columns in canonical match rows.

Important interpretation:

- These are statistical-engine improvements, not value-pick or staking changes.
- Calibration and Dixon-Coles outputs are diagnostics until validated across multiple leagues/seasons.
- Corners/cards count models remain the next modelling slice and should use separate count distributions, probably Negative Binomial where overdispersion exists.


## v0.49.5 Local Validation Notes

Focused validation completed in this build:

```bash
python -m compileall -q src scripts/build_model_ready_dataset.py tests/test_v0495_hybrid_model_ready_snapshots.py
python -m pytest tests/test_v0492_statistical_engine_evaluation.py tests/test_match_value.py tests/test_value_backtest.py tests/test_v0494_match_dataset_foundation.py tests/test_v0495_hybrid_model_ready_snapshots.py -q
python scripts/build_model_ready_dataset.py --matches <sample_matches.csv> --out-dir <tmp_out> --dataset-name smoke
```

Result:

```text
16 passed
script smoke passed
```

The full suite was not rerun in this build. The known `artifact_tool` spreadsheet warmup warning may appear on stderr, but the focused commands completed with exit code `0`.


## Latest v0.49.6 Implementation Notes

Implemented on `2026-06-25`:

```text
v0.49.6 — External Data Enrichment & Feature Expansion
```

Purpose:

```text
Turn the v0.49.5 hybrid data contract into a practical enrichment pipeline.
```

Implemented components:

```text
scripts/build_team_registry.py
scripts/download_clubelo.py
scripts/enrich_matches_with_clubelo.py
scripts/download_understat_xg.py
scripts/enrich_matches_with_xg.py
src/mundialytics/data_quality/team_registry.py
src/mundialytics/enrichment/clubelo.py
src/mundialytics/enrichment/xg.py
src/mundialytics/enrichment/understat.py
```

Important rules:

- ClubElo is downloaded/cached under `data/external/clubelo`.
- xG is optional and may come from Understat research downloads, paid APIs or manual provider CSVs.
- The generated `team_registry.csv` must be reviewed when provider aliases are uncertain.
- Current-match xG is a post-match observation; only prior rolling xG is a model feature.
- Missing external data does not block Level 1 baseline operation.
- The model logic is not claimed improved until local validation compares metrics.

Recommended next local action:

```text
Build Big 5 registry → download ClubElo → enrich one league → optional xG → build model-ready snapshots → rerun historical validation.
```

## 2026-06-25 — v0.49.7 ClubElo team-history download fix

Status: Implemented

The v0.49.6 ClubElo downloader used daily full-table snapshots by match date, which is too slow for multi-season Big 5 datasets. v0.49.7 makes team-history download the default.

New default:

```text
one ClubElo API request per team alias
```

instead of:

```text
one ClubElo API request per unique match date
```

The legacy daily snapshot mode remains available through `--mode daily-snapshot`, but normal enrichment should use `--mode team-history` and `--source-mode auto`.



## Latest v0.49.8 xG ingestion decision

Accepted on `2026-06-25`:

Direct Understat scraping is best-effort. If Understat no longer exposes inline `datesData`, the pipeline must not crash or block ClubElo/snapshot generation.

The supported xG contract is now provider-agnostic:

```text
provider/manual xG CSV
→ canonical understat_xg_matches.csv contract
→ xG enrichment
→ model-ready snapshots
```

`scripts/import_xg_csv.py` is the preferred fallback when direct Understat scraping is blocked.
`enrich_matches_with_xg.py --allow-missing-xg` can be used for batch jobs where xG coverage is temporarily zero.


## v0.49.9 continuity note

The project now has a free official xG/event import path via `scripts/import_statsbomb_open_xg.py`. It reads the local StatsBomb Open Data checkout and writes canonical match-level and shot-level xG files. This does not replace a future paid/full-coverage provider, but it unblocks free xG/event enrichment without relying on broken Understat scraping.
