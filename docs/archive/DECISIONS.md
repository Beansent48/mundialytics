# Decision Log

This file records important project decisions so future chats can continue from the ZIP without relying on hidden conversation history.

## 2026-06-25 — Keep planning and decisions inside `docs/`

Status: Accepted

### Context

The user reached a chat limit and wants the project ZIP to contain enough context for a new chat to continue safely.

### Decision

Important decisions, planned phases and specs should be documented inside `docs/` even before implementation.

### Consequences

Positive:

- Future chats can reconstruct project state from the repository.
- Planned work can be reviewed before implementation.
- The project avoids hidden assumptions from previous conversations.

Trade-offs:

- Some docs may describe proposed work that is not implemented yet.
- Every proposed document must clearly mark status to avoid false claims.

### Impacted Files / Areas

```text
docs/PROJECT_CONTINUITY.md
docs/DECISIONS.md
docs/V0491_ENTITY_SQUAD_GUARDRAILS_SPEC.md
README.md
```

### Validation Needed

- Confirm future specs are marked `Proposed`, `Accepted`, `Implemented` or `Deferred`.
- Do not update package version or changelog as implemented before code and tests exist.

## 2026-06-25 — v0.49.1 next phase should be Entity & Squad Guardrails

Status: Implemented

### Context

v0.49.0 implemented an offline data audit. The next recommended phase from the previous chat was `v0.49.1 — Entity & Squad Guardrails`.

### Decision

Implement `v0.49.1 — Entity & Squad Guardrails` as the next phase.

### Recommendation

Keep the conservative guardrail layer as the safety contract before adding new model features, dashboards or player-award predictions.

### Consequences

Positive:

- Reduces risk of stale/historical players entering current inference.
- Improves safety of player props and future Golden Boot modelling.
- Strengthens match and provider identity before more complex outputs.

Trade-offs:

- Delays feature expansion.
- May initially block outputs when source data lacks IDs or current squad evidence.

### Impacted Files / Areas

Likely:

```text
src/mundialytics/data_quality/
scripts/run_data_audit.py
tests/
docs/
```

### Validation Needed

- Focused guardrail tests.
- v0.49.0 audit regression test.
- Smoke audit command.
- Manual inspection of reason codes and reports.


## 2026-06-25 — Keep v0.49.1 guardrails inside the existing data-audit flow

Status: Accepted

### Context

The v0.49.1 guardrails strengthen the v0.49.0 offline data audit. Creating a separate runner would add another operational path before the project has a normalized real-data foundation.

### Decision

Integrate entity and squad guardrails into `scripts/run_data_audit.py`, while keeping reusable guardrail logic in `src/mundialytics/data_quality/entity_guardrails.py`.

### Consequences

Positive:

- One offline audit command now produces schema, coverage, gap, entity-quality and guardrail outputs.
- Guardrails are pure functions and focused tests can validate them without API calls.
- Model, training, odds and paper-mode behavior remain unchanged.

Trade-offs:

- The data-audit output contract now has three additional files.
- The audit status may remain `warning` more often because unsafe entity/squad conditions are surfaced instead of ignored.

### Impacted Files / Areas

```text
src/mundialytics/data_quality/entity_guardrails.py
src/mundialytics/data_quality/data_audit.py
scripts/run_data_audit.py
tests/test_v0491_entity_squad_guardrails.py
README.md
CHANGELOG.md
docs/V0491_ENTITY_SQUAD_GUARDRAILS_SPEC.md
```

### Validation Needed

Completed focused validation:

```text
compileall passed
pytest focused v0.49.0 + v0.49.1 passed
smoke data audit generated 11 outputs
```


## 2026-06-25 — Separate Statistical Engine from Value Pick Engine

Status: Accepted / Implemented in v0.49.2 documentation and evaluation outputs

### Context

During historical validation, the project produced useful 1X2 metrics and optional broad 1X2 value-backtest results. The user clarified that the main objective is not to bet every 1X2 signal. The objective is a statistical football engine similar in spirit to football analytics/reporting systems: predict outcomes, goals, corners, cards, player events, tournament simulations and awards.

### Decision

Mundialytics will explicitly separate two layers:

```text
Statistical Engine
→ core football prediction and simulation quality

Value Pick Engine
→ later selective market-opportunity layer
```

The Statistical Engine must be evaluated with statistical metrics:

```text
log loss
RPS
Brier score
calibration
MAE/RMSE
line calibration
scoreline probability and top-k coverage
```

The Value Pick Engine must be evaluated separately later and should produce a small number of selective picks, not broad bets on all 1X2 edges.

### Consequences

Positive:

- Model development is aligned with simulator/reporting quality.
- Profit/ROI cannot accidentally drive core model design.
- Future corners/cards/player-prop work can be evaluated statistically before market use.
- Value picks remain conservative and selective.

Trade-offs:

- Optional value backtests remain diagnostic and should not be overinterpreted.
- More evaluation outputs are needed before expanding to corners, cards and player props.
- Player props require both club evidence and national-team context guardrails.

### Impacted Files / Areas

```text
docs/MODEL_DESIGN.md
docs/NEXT_VALIDATION_STEPS.md
docs/PROJECT_CONTINUITY.md
docs/V0492_STATISTICAL_ENGINE_EVALUATION_SPEC.md
scripts/run_historical_validation.py
src/mundialytics/evaluation/statistical_engine.py
```

### Validation Needed

- Confirm historical validation writes statistical-engine outputs.
- Confirm optional value backtest does not affect model ranking.
- Confirm future docs do not describe ROI/profit as the objective of simulator validation.


## 2026-06-25 — Keep Poisson as the current statistical baseline

Status: Accepted

### Context

User-run historical validations during v0.49.2 planning compared Poisson and `random_forest_lambda` on national teams and multiple club datasets. Poisson consistently outperformed `random_forest_lambda` on the main 1X2 metrics across the shared runs.

### Decision

Use Poisson as the current baseline statistical engine for match-level probabilities and goal distributions. Keep `random_forest_lambda` as a secondary experiment, not as the primary model.

### Consequences

Positive:

- Simpler and more interpretable baseline.
- Consistent performance across national and club examples.
- Better fit for simulator-first reporting and tournament simulation.

Trade-offs:

- This does not yet validate corners, cards, shots, player props or award models.
- This does not prove betting profitability.
- Future models may still improve specific statistical targets, but must beat Poisson on agreed metrics.

### Validation Needed

- Continue evaluating goals, scorelines, over/under lines and BTTS.
- Add domain-specific metrics before expanding to corners/cards/player events.


## 2026-06-25 — Improve Poisson baseline with calibration, Dixon-Coles, time decay, shrinkage and Elo

Status: Accepted / Implemented in v0.49.3 for goals and scoreline diagnostics

### Context

Historical validation showed that the Poisson baseline outperformed `random_forest_lambda` across national and club examples. The user clarified that the goal is a statistical football engine, not a broad 1X2 betting system, and asked to clearly document Elo and the statistical improvements.

### Decision

Keep Poisson as the current baseline statistical engine and improve it with:

```text
detailed total-goals and BTTS calibration diagnostics
offline calibration layer for 1X2, totals and BTTS
Dixon-Coles scoreline diagnostics
time decay sample weighting
low-sample rolling-feature shrinkage
explicit internal Elo features
optional external Elo / ClubElo feature support
```

### Consequences

Positive:

- Improves the model around the baseline that already has evidence.
- Keeps the system interpretable.
- Separates probability/statistical quality from betting ROI.
- Makes Elo a visible first-class part of the model contract.

Trade-offs:

- Calibration and Dixon-Coles outputs are diagnostic until validated across multiple leagues/seasons.
- Time decay and shrinkage can change backtest metrics and must be compared before being called an improvement.
- Corners/cards require a separate modelling phase.

### Impacted Files / Areas

```text
src/mundialytics/evaluation/statistical_engine.py
src/mundialytics/models/goal_model.py
src/mundialytics/features/team_features.py
src/mundialytics/data/loaders.py
src/mundialytics/evaluation/backtest_runner.py
scripts/run_historical_validation.py
docs/V0493_STATISTICAL_MODEL_CALIBRATION_SPEC.md
docs/MODEL_DESIGN.md
docs/NEXT_VALIDATION_STEPS.md
```

### Validation Needed

- Focused unit/integration tests.
- Multi-league comparison of baseline vs v0.49.3 parameters.
- Review calibration-layer pre/post metrics.
- Review Dixon-Coles vs independent scoreline metrics.


## 2026-06-25 — Maximize data foundation before further model complexity

Status: Implemented

### Context

The user clarified that the next priority is not to add another model immediately. The goal is to improve the quality and treatment of the underlying football data first, then evaluate how the model reacts.

Recent validation showed that Poisson remains a solid baseline, while calibration/time-decay/shrinkage changes need robust evidence before becoming defaults. This suggests that the next high-leverage improvement is data foundation, not model complexity.

### Decision

Implement `v0.49.4 — Data Foundation and Match Dataset Treatment`.

The project should treat cleaned canonical match datasets as first-class artifacts before model training or validation.

### Consequences

Positive:

- Multi-season and multi-league datasets can be built consistently.
- Feature coverage for goals, shots, corners, cards, xG and Elo is visible before modelling.
- Unsafe rows, duplicates and anomalies are audited instead of hidden.
- Future corners/cards/player-event phases can depend on actual coverage rather than assumptions.
- Model changes can be compared on stronger data.

Trade-offs:

- This phase does not guarantee immediate metric improvements.
- It adds another preprocessing command before serious validation.
- Users must inspect data reports before interpreting model results.

### Impacted Files / Areas

```text
src/mundialytics/data_quality/match_dataset_foundation.py
scripts/build_match_dataset.py
scripts/run_historical_validation.py
docs/V0494_DATA_FOUNDATION_SPEC.md
```

### Validation Needed

- Focused tests for dataset cleaning/profiling.
- Smoke build of a Football-Data canonical dataset.
- Historical validation using the generated canonical dataset.
- Compare one-season vs multi-season metrics before changing model defaults.


## 2026-06-25 — Use a hybrid Big 5 club data/model architecture

Status: Implemented in v0.49.5 as data contract; model training still future work

### Context

The user wants to improve the statistical engine substantially by improving data quality and data treatment first. Local foundation runs showed strong Big 5 coverage for goals, shots, shots on target, corners, fouls and yellow cards.

The user explicitly chose the hybrid approach:

```text
one broad club model trained with Big 5 volume
+ league-aware features and diagnostics
```

instead of fully separate league models or a blindly pooled model.

### Decision

Adopt a hybrid club modelling direction:

```text
global Big 5 club model
+ league/context features
+ league-level diagnostics/calibration
+ team rolling features
+ internal Elo
+ optional ClubElo/external Elo
+ optional xG/event enrichments
```

### Consequences

Positive:

- Uses much more data than one-league models.
- Keeps league behaviour visible to the model and evaluation.
- Provides a natural path to league-level calibration when one league is over/under-estimated.
- Allows xG and ClubElo to improve the engine without becoming hard dependencies.

Trade-offs:

- Requires stricter data contracts.
- Requires by-league metrics so global averages do not hide league-level failures.
- Requires careful leakage controls for rolling and league-rate features.

### Impacted Files / Areas

```text
src/mundialytics/data_quality/model_ready_snapshots.py
scripts/build_model_ready_dataset.py
docs/V0495_HYBRID_BIG5_DATA_MODEL_SPEC.md
docs/MODEL_DESIGN.md
docs/DATA_SOURCE_STRATEGY.md
docs/NEXT_VALIDATION_STEPS.md
```

### Validation Needed

- Build model-ready snapshots for each Big 5 foundation dataset.
- Compare global and by-league metrics.
- Validate whether league-level lambda calibration improves Premier overestimation without damaging other leagues.


## 2026-06-25 — Keep xG optional and additive

Status: Accepted

### Context

The user wants to obtain more data such as xG because it likely adds football value. Current Football-Data CSV foundations show `xG` coverage as unavailable, while goals/shots/SOT/corners/fouls/yellows are available.

### Decision

xG is an optional enrichment, not a required baseline dependency.

Use staged data levels:

```text
Level 1: goals + shots + SOT + corners + fouls/cards + internal Elo
Level 2: Level 1 + ClubElo/external Elo
Level 3: Level 2 + xG/xA/event data
```

### Consequences

Positive:

- The engine works without xG.
- xG can improve the engine when coverage exists.
- Provider outages or partial xG coverage do not block baseline validation.

Trade-offs:

- Model selection must report which data level was used.
- xG-enriched models may not be directly comparable to Level 1 baselines unless evaluation reports the data contract.

### Validation Needed

- Add provider/caching/provenance checks before using xG in training.
- Verify xG coverage by competition/season.
- Confirm xG rolling features are shifted and leakage-safe.


## 2026-06-25 — Club data is primary for player evidence; national data remains primary for national-team results

Status: Accepted

### Context

The project covers tournament simulation and individual awards. Club datasets contain far more player/event evidence than national-team datasets, but national-team outcomes should not be predicted by blindly substituting club results.

### Decision

Use:

```text
Team result engine:
national-first for national matches using national results, national Elo and international context.

Player/event/award engine:
club evidence + national squad eligibility + expected minutes + tournament progression.
```

### Consequences

Positive:

- Avoids mixing club and national contexts in unsafe ways.
- Preserves the value of rich club player evidence.
- Keeps national-team outcome modelling grounded in national-team history.

Trade-offs:

- Requires a later explicit bridge from club player form to national squad/player projections.
- Requires guardrails around current squad eligibility and expected minutes.

### Validation Needed

- Do not train club and national rows in one base model.
- Player-award models must enforce squad/eligibility/current-player guardrails.


## 2026-06-25 — v0.49.6 external data enrichment pipeline

Status: Implemented

### Context

After the hybrid Big 5 data model was accepted, the user asked to obtain and use the additional data discussed during planning: ClubElo/external ratings, xG, richer rolling features and model-ready snapshots.

### Decision

Implement external data enrichment as an additive, cached pipeline:

```text
team_registry.csv
→ ClubElo cached daily snapshots
→ canonical_matches_with_clubelo.csv
→ optional xG provider/manual CSV
→ canonical_matches_with_xg.csv
→ enriched model_ready_match_snapshots.csv
```

### Consequences

Positive:

- The project can now enrich Big 5 foundations with ClubElo and xG when coverage exists.
- Team/provider aliases are explicit and editable.
- xG remains optional rather than a hard dependency.
- Model-ready snapshots now include rest days, season progress, external ratings and conversion features.

Trade-offs:

- External downloads can fail due to internet/provider/API issues.
- Understat scraping is marked as optional research and must be reviewed for provider terms/licensing.
- Provider aliases may need manual correction before joins reach full coverage.

### Impacted Files / Areas

```text
docs/V0496_EXTERNAL_DATA_ENRICHMENT_SPEC.md
scripts/build_team_registry.py
scripts/download_clubelo.py
scripts/enrich_matches_with_clubelo.py
scripts/download_understat_xg.py
scripts/enrich_matches_with_xg.py
src/mundialytics/data_quality/team_registry.py
src/mundialytics/enrichment/
src/mundialytics/data_quality/model_ready_snapshots.py
```

### Validation Needed

- Run the enrichment pipeline locally with internet access.
- Review `team_registry.csv`.
- Review `clubelo_join_report.csv` and `xg_join_report.csv`.
- Compare model metrics before/after enriched snapshots.

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



## 2026-06-25 — Treat Understat direct xG scraping as best-effort and support provider/manual CSV fallback

Status: Implemented in v0.49.8

### Context

A Big 5 xG download attempt returned `status=blocked` because the scraper could not find `datesData` in Understat league pages. This is a provider markup/source availability issue and should not block the rest of the data foundation.

### Decision

Keep xG as an optional Level 3 enrichment, but do not depend on direct Understat scraping. The canonical xG contract can be filled from any provider/manual CSV using `scripts/import_xg_csv.py`.

When direct scraping is blocked:

```text
understat_xg_matches.csv
```

is still written with canonical headers and zero rows so downstream scripts do not fail with `FileNotFoundError`.

### Consequences

Positive:

- Big 5 ClubElo/snapshot jobs can continue when xG is unavailable.
- Provider/manual xG sources can be added without changing model code.
- The project remains honest about xG coverage.

Trade-offs:

- Direct Understat scraping may remain unavailable depending on provider markup.
- A real xG provider/export is required for Level 3 coverage.
- Imported xG requires team/date matching review.


## v0.49.9 — Free xG provider decision

Decision: use **StatsBomb Open Data** as the preferred free xG/event source when no paid API key is available. Understat remains a best-effort research fallback only because direct scraping can be blocked or markup can change.

Important caveat: StatsBomb Open Data coverage is partial. The project must not treat it as complete Big 5 xG. Missing coverage must be explicit and model-ready snapshots must only use pre-match rolling xG features, never current-match xG as an input.

## 2026-06-25 — v0.50.0 advanced data layer

Status: Implemented

The user wants maximum statistical value from available football data. xG is considered strategically important, but no single free provider gives complete, stable Big 5 modern coverage. Therefore the accepted architecture is a multi-provider advanced data layer with canonical contracts, provider priority, local cache, coverage audit and leakage rules.

Implemented provider paths:

```text
FBref/soccerdata best-effort downloader
provider/manual advanced CSV import
Kaggle Understat CSV import
StatsBomb Open Data advanced event import
provider-priority merge
advanced match enrichment
advanced data coverage audit
```

Current-match advanced stats are post-match observations. Only prior rolling derived features may be used as model inputs.
