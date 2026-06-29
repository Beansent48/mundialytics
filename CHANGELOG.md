# Changelog

## v0.50.3 — FBref Team-Match Normalization Repair

- Fixed soccerdata/FBref team-match raw normalization for flattened columns such as `Standard.1` and `Performance.2`.
- Added conservative FBref mappings for:
  - shooting: shots and shots on target
  - misc: yellow/red cards, fouls, interceptions and tackles won
  - keeper: goals against, saves and save percentage
  - schedule: possession
- Kept FBref xG/npxG empty when the raw export does not contain xG columns.
- Added tests for FBref shooting, misc and keeper raw normalization.
- Added spec and next-chat handoff docs for continuing into Understat/Kaggle xG ingestion.


## 0.50.2 — Advanced Join Alias Repair

### Added
- Added manual team-alias support to advanced enrichment via `--manual-aliases`.
- Added alias-aware join logic that maps both Football-Data names and provider names to one canonical join key.
- Added `config/team_aliases/provider_team_aliases_manual.csv` as a seed alias file.
- Added regression coverage for `paris sg` ↔ `Paris Saint-Germain` joins.

### Changed
- Advanced enrichment now computes join keys from alias-normalized home/away team names.
- Enrichment reports include `manual_alias_rows` and `team_join_alias_count`.

### Validation
- `python -m compileall -q src scripts`: passed.
- `python -m pytest -q tests/test_v0500_advanced_football_data_layer.py`: 4 passed.

### Known Limitations
- Current available advanced coverage is still partial; the user diagnostic found 84 alias-joinable matches, mostly StatsBomb Open Data.
- Full Big 5 xG still requires better FBref coverage and/or Kaggle Understat import.
- Fuzzy matching is not enabled in production to avoid false-positive joins.



## 0.50.1 — Maximum Useful Data Ingestion

### Added
- Expanded advanced match contract with neutral venue context, shot location/body-part metrics, set-piece xG, territory/progression/defensive/keeper fields.
- Added generic player-match and shot-event CSV normalization.
- Added StatsBomb lineup export from Open Data.
- Added optional FBref player match stats, lineups and events downloads.
- Added documentation for advanced metrics and player lineup strength rating.

### Changed
- Advanced source merge now fills each metric by provider priority instead of selecting one whole provider row per match.
- Advanced enrichment now preserves existing canonical stats when provider advanced values are missing.
- Model-ready snapshots dynamically roll expanded numeric team metrics while preserving the previous snapshot version contract.

### Known Limitations
- xG coverage still depends on successful Understat/FBref/StatsBomb joins.
- Player lineup strength rating is documented but not implemented yet.
- Market/odds features are intentionally excluded from this ingestion version.


## v0.49.9 — Free StatsBomb Open Data xG Provider

- Added `scripts/import_statsbomb_open_xg.py` to import official free StatsBomb Open Data xG/events from a local open-data checkout.
- Added `src/mundialytics/enrichment/statsbomb_open.py`.
- Produces canonical match-level xG and shot-level xG/event CSVs:
  - `statsbomb_xg_matches.csv`
  - `statsbomb_xg_shots.csv`
  - `statsbomb_xg_import_report.json`
- Keeps Understat as best-effort research fallback only; StatsBomb Open Data is now the preferred free stable xG/event path when no paid API key is available.
- Documents that StatsBomb Open Data coverage is partial and must not be treated as complete Big 5 xG.
- No model logic changed.


## v0.49.8 — xG Provider Fallbacks and Understat Block Handling

- Updated the optional Understat xG downloader to support multiple historic inline JSON formats.
- When direct Understat scraping is blocked, the downloader now still writes an empty canonical `understat_xg_matches.csv` so batch enrichment does not fail with `FileNotFoundError`.
- Added provider/manual CSV import mode to `scripts/download_understat_xg.py` via `--input-csv`.
- Added `scripts/import_xg_csv.py` as a clearer alias for importing any provider/manual xG CSV into the canonical xG contract.
- Updated `scripts/enrich_matches_with_xg.py` with `--allow-missing-xg` so Big 5 batch jobs can continue while xG is unavailable.
- Documented that Understat direct scraping is best-effort and that xG should be sourced via provider/manual CSV when provider markup blocks scraping.
- Added focused tests for blocked Understat downloads, provider CSV import and missing-xG enrichment fallback.
- No model logic changed.


## v0.49.7 — ClubElo Team-History Download Fix

- Changed `scripts/download_clubelo.py` default mode to `team-history`.
- Added one-history-per-team ClubElo cache under `data/external/clubelo/teams/`.
- Kept legacy daily snapshot downloads behind `--mode daily-snapshot`.
- Updated ClubElo enrichment to prefer team histories in `--source-mode auto`.
- Added temporal lookup using ClubElo `From`/`To` intervals.
- Added focused tests for team-history download/enrichment.
- No model logic changed.


## 0.49.6 — External Data Enrichment & Feature Expansion

Implemented:

- Added editable Big 5/team provider registry:
  - `src/mundialytics/data_quality/team_registry.py`
  - `scripts/build_team_registry.py`
- Added ClubElo external rating pipeline:
  - `src/mundialytics/enrichment/clubelo.py`
  - `scripts/download_clubelo.py`
  - `scripts/enrich_matches_with_clubelo.py`
- Added xG enrichment pipeline:
  - `src/mundialytics/enrichment/xg.py`
  - `src/mundialytics/enrichment/understat.py`
  - `scripts/download_understat_xg.py`
  - `scripts/enrich_matches_with_xg.py`
- Expanded model-ready snapshots:
  - rest days,
  - season progress,
  - external Elo/ClubElo features,
  - rolling conversion features,
  - optional xG targets and prior rolling xG features.
- Added spec:
  - `docs/V0496_EXTERNAL_DATA_ENRICHMENT_SPEC.md`
- Added tests:
  - `tests/test_v0496_external_data_enrichment.py`

Validation:

- `compileall` passed for new scripts/modules.
- Focused pytest suite passed.


## v0.49.5 - Hybrid Big 5 data model and model-ready snapshots

- Updated package version metadata to `0.49.5`.
- Added `docs/V0495_HYBRID_BIG5_DATA_MODEL_SPEC.md`.
- Documented the accepted hybrid club architecture:
  - global Big 5 club model,
  - league/context features,
  - league-level diagnostics/calibration,
  - team rolling features,
  - internal Elo,
  - optional ClubElo/external Elo,
  - optional xG/event enrichments.
- Added `src/mundialytics/data_quality/model_ready_snapshots.py`.
- Added `scripts/build_model_ready_dataset.py`.
- Added `model_ready_match_snapshots.csv` output contract with one row per match.
- Added `model_ready_feature_contract.csv` to mark every column as `identity`, `feature` or `target`.
- Added `model_ready_snapshot_report.json`.
- Added leakage-safe league prior features:
  - prior league goal rate,
  - home/away goal rates,
  - draw rate,
  - BTTS rate,
  - over 2.5 rate.
- Added pre-match internal Elo columns to snapshots.
- Added rolling xG support to team rolling features when xG columns are present.
- Kept xG optional and additive; baseline operation still works without xG.
- Documented club/national separation:
  - club data is primary for player/event evidence,
  - national results/Elo remain primary for national-team outcomes.
- Added `tests/test_v0495_hybrid_model_ready_snapshots.py`.

Focused validation:

```text
compileall passed
pytest focused statistical/value/foundation/snapshot tests: 16 passed
script smoke passed
```



## v0.49.4 - Data foundation and match dataset treatment

- Updated package version metadata to `0.49.4`.
- Added `docs/V0494_DATA_FOUNDATION_SPEC.md`.
- Added `src/mundialytics/data_quality/match_dataset_foundation.py`.
- Added `scripts/build_match_dataset.py` to build cleaned, profiled canonical match datasets from one or many input files.
- Added multi-file / glob input support for Football-Data.co.uk, canonical, international-results and OpenFootball sources.
- Added data-foundation outputs:
  - `canonical_matches_raw_combined.csv`
  - `canonical_matches.csv`
  - `match_dataset_foundation_report.json`
  - `match_dataset_feature_coverage.csv`
  - `match_dataset_quality_by_competition_season.csv`
  - `match_dataset_anomalies.csv`
  - `match_dataset_dropped_rows.csv`
- Integrated data-foundation profiling into `scripts/run_historical_validation.py` through the `data_foundation` section in `operational_validation_report.json`.
- Added feature-coverage profiling for goals, shots, shots on target, corners, fouls, yellow cards, red cards, xG, external Elo and ClubElo.
- Added conservative anomaly detection for invalid rows, duplicate identities, impossible/unsafe stat relationships and mixed scopes.
- Preserved model logic: this phase improves data treatment and observability, not model parameters or value-pick behavior.
- Added `tests/test_v0494_match_dataset_foundation.py`.

Focused validation:

```text
compileall passed
pytest focused statistical/value/foundation tests: 14 passed
```



## v0.49.3 - Statistical model calibration foundation

- Updated package version metadata to `0.49.3`.
- Added `docs/V0493_STATISTICAL_MODEL_CALIBRATION_SPEC.md`.
- Documented Elo as a first-class model pillar; internal Elo is already implemented and used in walk-forward validation.
- Added optional external Elo / ClubElo feature support for canonical match rows containing `home_external_elo`, `away_external_elo`, `home_clubelo`, `away_clubelo`, `home_elo` or `away_elo`.
- Added detailed total-goals and BTTS calibration-bin output:
  - `statistical_engine_line_calibration_<model_type>.csv`
- Added offline calibration-layer diagnostics for:
  - 1X2 probabilities,
  - total-goals lines,
  - BTTS.
- Added Dixon-Coles low-score scoreline diagnostics:
  - `statistical_engine_dixon_coles_scorelines_<model_type>.csv`
- Added recency weighting to the goal-lambda model through `time_decay_half_life_days`.
- Added low-sample shrinkage for rolling team features through `rolling_shrinkage_prior_matches`.
- Added CLI parameters:
  - `--poisson-alpha`
  - `--time-decay-half-life-days`
  - `--rolling-shrinkage-prior-matches`
- Documented corners/cards count models, likely Negative Binomial, as the next modelling slice; not falsely marked as complete.

Focused validation:

```text
compileall passed
pytest focused statistical/value tests: 12 passed
```


## v0.49.2 - Statistical engine evaluation foundation

- Updated package version metadata to `0.49.2`.
- Added `src/mundialytics/evaluation/statistical_engine.py` for offline statistical-engine quality evaluation.
- Integrated statistical-engine outputs into `scripts/run_historical_validation.py` for each model type.
- Added outputs for goal errors, total-goals line calibration, BTTS calibration and exact-score diagnostics.
- Added `docs/V0492_STATISTICAL_ENGINE_EVALUATION_SPEC.md`.
- Updated `docs/MODEL_DESIGN.md`, `docs/NEXT_VALIDATION_STEPS.md`, `docs/PROJECT_CONTINUITY.md`, `docs/DECISIONS.md` and `README.md` to document the separation between Statistical Engine and Value Pick Engine.
- Added `tests/test_v0492_statistical_engine_evaluation.py`.
- Added regression coverage for full-season odds files overlapping prediction columns or containing fixtures outside the backtest window.
- Preserved model logic, training logic, calibration logic, API behavior, player-prop behavior and live-betting behavior.
- Clarified that ROI/profit is diagnostic for a later value-pick layer and must not drive core simulator model development.



## v0.49.1 - Entity and squad guardrails

- Updated package version metadata to `0.49.1`.
- Added `src/mundialytics/data_quality/entity_guardrails.py` for offline match identity, provider fixture ID, team scope, fixture-team, squad eligibility and historical-player guardrails.
- Integrated guardrail outputs into `scripts/run_data_audit.py` through `entity_guardrails_report.csv`, `squad_guardrails_report.csv` and `guardrail_summary.json`.
- Added guardrail sections to `data_audit_report.html` and embedded guardrail summary details in `data_audit_summary.json`.
- Added `tests/test_v0491_entity_squad_guardrails.py` and preserved the focused v0.49.0 data-audit regression path.
- Preserved model logic, training logic, API behavior, odds handling, player-prop model behavior and betting/paper-mode behavior.
- Kept unsafe player-prop candidates conservative with explicit reason codes such as `player_not_in_current_squad`, `missing_current_eligibility` and `historical_only_player_for_current_inference`.

## v0.49.0 - Offline data audit report

- Updated package version metadata to `0.49.0`.
- Added `src/mundialytics/data_quality/data_audit.py` for offline schema, coverage, gap and entity-quality audits.
- Added `scripts/run_data_audit.py` as a separate data-quality entrypoint, keeping data audit separate from prediction and simulation evaluation.
- Added `data_audit_summary.json`, `data_audit_report.csv`, `coverage_report.csv`, `data_gaps_report.csv`, `entity_quality_report.csv`, `feature_availability_matrix.csv`, `next_data_requirements.csv` and `data_audit_report.html` outputs.
- Added `docs/V0490_DATA_AUDIT_REPORT_SPEC.md`.
- Added `tests/test_v0490_data_audit.py`.
- Preserved model logic, simulation logic, evaluation logic, odds handling, player-prop logic and betting/paper-mode behavior.
- Kept player props and Golden Boot development conservative by documenting required current squads, expected minutes and player event history before future inference.

## v0.48.4 - Simulation evaluation and baseline report

- Updated package version metadata to `0.48.4`.
- Added `src/mundialytics/statistical_core/simulation_evaluation.py` for offline simulator evaluation.
- Added `scripts/run_simulation_evaluation.py` as a separate evaluation entrypoint, keeping inference and evaluation responsibilities separate.
- Added `simulation_metrics.json`, `simulation_evaluation.csv`, `calibration_1x2.csv`, `goal_error_metrics.csv`, `scoreline_evaluation.csv`, `baseline_comparison.csv`, `line_evaluation.csv` and `simulation_evaluation_report.html` outputs.
- Added diagnostic baselines: uniform 1X2 baseline and empirical-frequency baseline.
- Added missing-data policy: if actual results are unavailable, evaluation outputs are generated with `status = not_available` instead of invented metrics.
- Added `data/sample/sample_actual_results_for_evaluation.csv` as clearly marked smoke-test sample data, not real future results.
- Added `docs/V0484_SIMULATION_EVALUATION_SPEC.md`.
- Added `docs/NEXT_DATA_FOUNDATION_REQUIREMENTS.md` to preserve the required data formats for the upcoming data-foundation phase.
- Added `tests/test_v0484_simulation_evaluation.py`.
- Preserved model logic, calibration logic, odds handling, player-prop logic and tournament simulation behavior.

## v0.48.3 - Tournament visual report

- Updated package version metadata to `0.48.3`.
- Updated the statistical matchday runner audit version to `v0.48.3_tournament_visual_report`.
- Added `src/mundialytics/statistical_core/tournament_report.py` to build a tournament visual report from existing Monte Carlo outputs.
- Added `tournament_report.csv` and `tournament_report.json` outputs.
- Added `Tournament Visual Report` to `daily_report.html` with championship race, qualification race, group winner race, expected group tables, knockout path views, attacking projection, uncertainty watchlist and optional competition/player award context.
- Added lightweight HTML probability bars without new dependencies.
- Added tournament report validation to `simulation_contract_report.json`.
- Added `docs/V0483_TOURNAMENT_VISUAL_REPORT_SPEC.md`.
- Added `tests/test_v0483_tournament_report.py`.
- Preserved model logic, CLI compatibility, optional odds behavior and paper-only betting context.


## v0.48.2 - Matchday summary rankings

- Updated package version metadata to `0.48.2`.
- Updated the statistical matchday runner audit version to `v0.48.2_matchday_summary_rankings`.
- Added `src/mundialytics/statistical_core/matchday_summary.py` to build simulator-first daily rankings from existing outputs.
- Added `matchday_summary.csv` and `matchday_summary.json` outputs.
- Added `Matchday Summary Rankings` to `daily_report.html` with categories for high goal expectation, low goal environment, most balanced matches, strongest favorites, highest uncertainty, BTTS lean, dynamic statistical signals and data-quality watchlist.
- Added matchday summary validation to `simulation_contract_report.json`.
- Added `docs/V0482_MATCHDAY_SUMMARY_RANKINGS_SPEC.md`.
- Added `tests/test_v0482_matchday_summary.py`.
- Preserved existing model logic, CLI compatibility, optional odds behavior and paper-only betting context.

## v0.48.1 - Advanced match report

- Updated package version metadata to `0.48.1`.
- Updated the statistical matchday runner audit version to `v0.48.1_advanced_match_report`.
- Added `docs/V0481_ADVANCED_MATCH_REPORT_SPEC.md` as the accepted v0.48.1 reporting spec.
- Reworked `daily_report.html` into an advanced simulator-first report with executive summary, per-match cards, top scorelines, dynamic goal lines, team/player statistics, data-quality flags and simulation metadata.
- Kept optional paper-value context separated from the statistical report.
- Added HTML section validation to `simulation_contract_report.json`.
- Added `tests/test_v0481_advanced_match_report.py` and updated focused simulator contract validation.
- Preserved the existing CLI flow and avoided new dependencies, API calls or model-training changes.


## v0.48.0 - Statistical simulator upgrade

- Shifted the active roadmap to simulator-first development before expanding betting-pick features.
- Updated package version metadata to `0.48.0`.
- Updated the statistical matchday runner audit version to `v0.48_statistical_simulator_upgrade`.
- Added `docs/V048_STATISTICAL_SIMULATOR_SPEC.md` as the accepted v0.48 simulator contract/spec.
- Added `src/mundialytics/statistical_core/simulation_contract.py` to define expected simulator outputs and key schemas.
- Added `simulation_contract_report.json` as a machine-readable output contract/audit artifact.
- Added `--detail-sample-simulations` to control retained sample rows in `tournament_details.csv`.
- Optimised tournament score sampling by precomputing scoreline distributions per fixture, making large Monte Carlo runs such as 50,000 simulations more practical.
- Updated the HTML report header and audit section for the simulator-first workflow.
- Added a focused v0.48 smoke test for the simulator contract.
- Kept odds comparison optional and kept OddsPapi historical bulk as experimental.

## v0.47.0 - Prediction and simulation consolidation

- Consolidated Mundialytics Betting Engine as a prediction, simulation, dynamic market-line and paper-value engine.
- Kept OddsPapi/RapidAPI as an optional experimental odds layer; historical bulk fixture discovery must not block the statistical core.
- Updated package version metadata to `0.47.0`.
- Updated the statistical matchday runner audit version to `v0.47_prediction_simulation_consolidation`.
- Added `docs/V047_PREDICTION_SIMULATION_CONSOLIDATION.md` as the accepted v0.47 scope/spec.
- Added a focused smoke test for `scripts/run_statistical_matchday.py`.
- Clarified README guidance so the recommended workflow is prediction/simulation-first, not Exchange/live betting.

## v0.43 - OddsPapi RapidAPI training update

- Hardened OddsPapi/RapidAPI client with monthly ledger, cache hits, explicit 429/401/403 errors and free-tier budget guard.
- Added current odds fetcher for mapped fixtures.
- Improved v5 historical odds tree flattening and older RapidAPI bookmakerOdds shape support.
- Added training odds feature builder for 1X2 and market-line devig features.
- Added v0.43 tests for RapidAPI config, historical tree flattening and feature construction.


## v0.40.0 - Odds-ready contract layer

- Added provider-agnostic odds-ready schema for future odds APIs.
- Added compact shortlist/template generation from market-side candidates.
- Added value-edge calculator for historical/bookmaker odds CSVs.
- Added docs/ODDS_READY_CONTRACT.md.
- Does not fetch odds, place bets, or change existing model performance logic.

## v0.38.2 - Event-line backtest robustness fix

- Fixes `TypeError: sequence item 0: expected str instance, float found` when combined team stats contain mixed/NaN `saves_data_quality_flag` values.
- Cleans and joins audit flags safely across Football-Data, StatsBomb raw and derived-save rows.
- Skips dry-run/download summary JSON files when building StatsBomb raw extra stats and goalkeeper stats.
- Reads large combined market stats CSVs with `low_memory=False` to avoid noisy mixed-type warnings.

## v0.36.2 - Full bookmaker-style market side policy

- Generalises pick-policy signals beyond goals/BTTS/1X2 to any settled over/under market.
- Supports typical bookmaker sides for corners, team/player shots, shots on target, cards, fouls and goalkeeper saves once real targets exist.
- Adds `--line-signals` to `scripts/backtest_pick_policy.py` so settled event-line predictions can be evaluated alongside match markets.
- Updates market coverage audit to require both sides for over/under markets and BTTS Yes/No.
- Keeps corners and goalkeeper saves blocked until real targets are present; no proxy saves/corners are invented.


## v0.36.0 - Market model audit and missing-market guardrails

- Added `scripts/audit_market_coverage.py` to explicitly show which markets are trainable/evaluable from the current data.
- Added per-market and per-line model performance outputs to `scripts/backtest_pick_policy.py`:
  - `market_model_performance.csv`
  - `market_line_performance.csv`
  - `market_model_takeaways.json`
- Reinforced that corners and goalkeeper saves must remain `not_available` unless real target data exists.
- Added `goalkeeper_saves_not_available_not_invented` audit warning in the statistical matchday runner.
- Kept console pick viewer showing both overs and unders.


## v0.34.1 - Console matchday viewer

- Added `scripts/show_matchday_console.py` to inspect matchday predictions directly in PowerShell.
- Supports filtering by team or match ID.
- Prints 1X2 probabilities, xG, scorelines, team event projections, dynamic market lines and player prop candidates.


## v0.34.0 - Squad candidate filtering for player props

- Adds candidate ranking for squad/roster fallback player inputs.
- Adds candidate_policy, candidate_reason, candidate_rank_team and candidate_score to player prop outputs.
- Excludes unresolved, zero-sample and very-low-confidence squad players from available player prop lines.
- Keeps confirmed/manual lineups as high-priority candidates while marking squad fallback as lower confidence.
- Blocks low-confidence squad candidates from SOT and higher dynamic lines to reduce noisy player markets before confirmed lineups exist.

## v0.33 - historical position fallback and player input confidence
- Uses matched players' historical most-frequent position when provider roster position is generic (D/M/F).
- Adds input/resolved position columns plus position_source for auditability.
- Adds player_input_source and player_selection_confidence for squad-vs-lineup transparency.
- Keeps goalkeeper attacking prop guardrails and unresolved-player blocks.


## v0.32.2 - player input guardrails

- Fix ESPN roster position parsing so dict positions do not become `unknown_outfield`.
- Map provider position abbreviations (`G`, `F`, `M`, `D`) to role groups.
- Mark unresolved/zero-sample player prop dynamic lines as `not_available` instead of available.
- Add tests for ESPN compact positions, provider dict-string positions and unresolved squad prop blocking.

## v0.32

- Added event/user/UTC date filtering for automatic matchday fixtures via `--date-mode`.
- Added event-local kickoff columns and host-city timezone inference for World Cup fixtures.
- Expanded national-team alias mapping, including Ivory Coast/Côte d'Ivoire -> `cote d ivoire`.
- Added best-effort current lineup/squad ingestion from SofaScore and ESPN public endpoints.
- Added audit fields for player input fetching and lineup/squad fallback quality.
- Added tests for team identity mapping, event-local date filtering, and lineup/squad parsers.


## v0.31

- Added `scripts/build_today_matchday_inputs.py` to generate run-ready `today_fixtures.csv` from today's public/free fixture providers or an already fetched fixture CSV.
- Added `mundialytics.matchday.today_builder` with status filtering, local-date filtering, provider IDs and audit output.
- Demo odds are now labelled `demo_odds_only` in dynamic lines instead of producing real `high_value` labels.
- Added evidence-source columns for recent, similar-Elo and H2H evidence.
- Player prop evidence now falls back from current-team recent sample to canonical-player history when needed.


## v0.26 - Statistical Model Upgrade

- Added optional Dixon-Coles low-score correction to the match scoreline distribution.
- Added Dixon-Coles challenger configs to the automatic match model lab.
- Added starter/substitute-role expected minutes to the player prop champion lab.
- Added optional negative-binomial count probability link for overdispersed prop markets.
- Added optional shots-on-target conditional layer from expected shots and historical SOT conversion.
- Added v0.26 docs and tests for Dixon-Coles and player prop statistical upgrades.


## v0.25.0 - Champion model selection and recovered player-prop architecture

- Added `scripts/run_player_prop_champion_lab.py` to choose the best player-prop architecture per market.
- Added `src/mundialytics/statistical_core/player_prop_champion.py` with v0.16-style expected-minutes, player-rate, team-environment, player-share and hierarchical calibration logic.
- Added `prediction_registry.json` output so markets can be promoted, downgraded or blocked based on validation against baseline.
- Added segment diagnostics by team_type, gender, competition_context, competition, position, sample-size bucket and probability bucket.
- Fixed production matchday runner so `--event-model-config` is actually applied to `PlayerEventModel`.
- Added v0.25 docs and tests.


## v0.24.0 - Market-specific event evaluation and hardening

- Added temporal holdout evaluation for team events and player props.
- Added event model lab to compare team/player event configurations.
- Added configurable TeamStatsModel and PlayerEventModel parameters.
- Added event evaluation reports, market policies and worst-miss audits.
- Added optional --event-model-config support to run_statistical_matchday.py.

## v0.23 - Model Lab Auto Hardening

- Added `scripts/run_model_lab.py` for automatic experiment sweeps over match-model caps, shrinkage, recency and overconfidence controls.
- Added `src/mundialytics/statistical_core/model_lab.py` with leaderboard, best config, calibration export and HTML report.
- Made `MatchOutcomeModel` configurable through auditable kwargs (`goal_cap`, `profile_shrinkage_k`, `low_sample_blend_k`, rating controls, draw lambda blending).
- Added fast precomputed team-goal evaluation path for model-lab iterations.
- Added `--model-config` support to `evaluate_statistical_core.py` and `run_statistical_matchday.py`.
- Ran a real-data model lab on the uploaded StatsBomb event dataset and selected a hardened config that reduced full-holdout raw 1X2 log loss from ~1.213 to ~1.044.


## v0.18.3 - Free current fixture fallback

- Added SofaScore public scheduled-events adapter for free/keyless fixture discovery.
- Added `scripts/fetch_sofascore_fixtures.py` and `scripts/fetch_world_cup_fixtures_free.py`.
- Added local-date post-filtering in an IANA timezone to avoid yesterday/tomorrow slate drift.
- Added senior World Cup convenience filter that excludes Club/Women/U-age/qualifier competitions by default.
- Added tests for SofaScore fixture parsing, World Cup filtering, and local timezone filtering.


## v0.18 - Provider Identity Layer

- Added API-Football/API-Sports adapter for free operational fixtures and lineups.
- Added provider identity mapping: `api_football:<player_id>` -> historical `player_id_global`.
- Added `build_provider_identity_map.py` and `diagnose_provider_identity.py`.
- Safe lineup props now accepts `--identity-map` and preserves provider/canonical/historical identity columns.
- Added tests for provider IDs, identity maps and safe-props identity resolution.


## v0.17.1 - Player identity resolver hardening

- Added conservative fuzzy player identity resolver for current-lineup inference.
- Added resolved identity output columns in safe player props.
- Added `scripts/diagnose_player_identity.py` for operational matching audits.
- Added regression tests for short-name to full-name matching (Morata/Valverde style failures).
- Fixed test path setup so scripts can be imported during full-suite pytest.


## v0.17 - Player props finalization

- Added adaptive hierarchical calibration selection for player props.
- Added per-competition diagnostics in hierarchical calibration reports.
- Added `scripts/finalize_player_props_policy.py` to generate market-level operational policy.
- Added support for `--calibration-policy` in safe lineup inference.
- Lowered default hierarchical group threshold to 200 for available open-data coverage.
- Added tests for adaptive selection, policy generation, and safe inference policy usage.


## v0.16 - Hierarchical prop calibration and club-to-national player evidence

- Added hierarchical player-prop calibration by competition, domain context, team type/gender, then market fallback.
- Added optional `--feature-player-events` for leakage-safe cross-context player features.
- Added `club_minutes_sample`, `national_minutes_sample`, and `cross_context_feature_used` to prop predictions/inference outputs.
- Updated safe lineup inference to select hierarchical calibrators when calibration predictions are supplied.
- Strengthened audits for cross-context feature flags and sample metadata.
- Added v0.16 regression tests and audit documentation.


## v0.14 - Operational hardening

- Fixed prop backtest leakage: target-match observed minutes are no longer used as pre-match expected minutes by default.
- Prop prediction CSVs now preserve competition/team_scope/player identity/position/started metadata.
- Strict audits now fail on missing prediction metadata, date nulls, placeholder competitions, and observed-minute leakage.
- Added current-lineup-only matchday analysis combining ELO+Poisson fixture probabilities with safe calibrated props.
- Added regression tests for metadata preservation, no-minute-leakage, and lineup-only inference.


## v0.11 - Safe lineup-only props and date repair

- Added repair script to fill missing prop prediction dates/competition from player event metadata.
- Added real-date temporal calibration check.
- Added safe lineup-only player prop inference so retired historical players can train the model but cannot appear in current predictions unless supplied in the lineup input.
- Added market-specific probability caps and low-sample warnings.


## v0.10 - Player prop calibration search

- Added `scripts/calibrate_player_props.py` to search calibration methods per market.
- Added `src/mundialytics/evaluation/prop_calibration.py` with identity, rate-shift, Platt/logit, Platt+extra features, and isotonic calibration.
- Added incoherence diagnostics for prop prediction files.
- Added reliability-bin CSV outputs before and after calibration.
- Added optional `--run-calibration` to `scripts/run_player_props_pipeline.py`.
- Added regression tests for calibration and CLI outputs.


## v0.7 - historical validation runner

- Added `scripts/run_historical_validation.py`, a one-command operational check for real historical datasets.
- The script converts raw public datasets when needed, runs data diagnostics, walk-forward backtests, readiness gates, optional historical odds value backtests, and trains a final scoped model bundle.
- Added `docs/NEXT_VALIDATION_STEPS.md` with the recommended local verification flow before using upcoming fixtures.
- Fixed validation readiness to evaluate completed historical matches separately from future fixture rows, so upcoming matches in the same file do not falsely fail the data-quality gate.


## v0.5-data-sources

- Added Wyscout public event-data adapter for player events, team events and lineup/substitution extraction.
- Expanded StatsBomb adapter with lineups, substitutions, tactical shifts, team event aggregation and extra player metrics.
- Added `scripts/build_event_datasets.py` for StatsBomb and Wyscout event ingestion.
- Added event-data sample JSONs and regression tests.
- Added `docs/DATA_SOURCE_STRATEGY.md` and `docs/EVENT_DATA_PIPELINE.md`.


## v0.49.0 - Offline data audit report

- Updated package version metadata to `0.49.0`.
- Added `src/mundialytics/data_quality/data_audit.py` for offline schema, coverage, gap and entity-quality audits.
- Added `scripts/run_data_audit.py` as a separate data-quality entrypoint, keeping data audit separate from prediction and simulation evaluation.
- Added `data_audit_summary.json`, `data_audit_report.csv`, `coverage_report.csv`, `data_gaps_report.csv`, `entity_quality_report.csv`, `feature_availability_matrix.csv`, `next_data_requirements.csv` and `data_audit_report.html` outputs.
- Added `docs/V0490_DATA_AUDIT_REPORT_SPEC.md`.
- Added `tests/test_v0490_data_audit.py`.
- Preserved model logic, simulation logic, evaluation logic, odds handling, player-prop logic and betting/paper-mode behavior.
- Kept player props and Golden Boot development conservative by documenting required current squads, expected minutes and player event history before future inference.

## v0.48.4 - Simulation evaluation and baseline report

- Updated package version metadata to `0.48.4`.
- Added `src/mundialytics/statistical_core/simulation_evaluation.py` for offline simulator evaluation.
- Added `scripts/run_simulation_evaluation.py` as a separate evaluation entrypoint, keeping inference and evaluation responsibilities separate.
- Added `simulation_metrics.json`, `simulation_evaluation.csv`, `calibration_1x2.csv`, `goal_error_metrics.csv`, `scoreline_evaluation.csv`, `baseline_comparison.csv`, `line_evaluation.csv` and `simulation_evaluation_report.html` outputs.
- Added diagnostic baselines: uniform 1X2 baseline and empirical-frequency baseline.
- Added missing-data policy: if actual results are unavailable, evaluation outputs are generated with `status = not_available` instead of invented metrics.
- Added `data/sample/sample_actual_results_for_evaluation.csv` as clearly marked smoke-test sample data, not real future results.
- Added `docs/V0484_SIMULATION_EVALUATION_SPEC.md`.
- Added `docs/NEXT_DATA_FOUNDATION_REQUIREMENTS.md` to preserve the required data formats for the upcoming data-foundation phase.
- Added `tests/test_v0484_simulation_evaluation.py`.
- Preserved model logic, calibration logic, odds handling, player-prop logic and tournament simulation behavior.

## v0.4-agent

- Added match-winner value betting pipeline.
- Added Football-Data.co.uk 1X2 odds extraction.
- Added value backtesting for historical predictions + odds.
- Added readiness/quality gate for cautious production assessment.
- Added reliability bins to walk-forward backtest summaries.
- Fixed `shrink_probability` handling of `pd.NA` and zero-strength shrinkage.
- Fixed sample match odds fixture IDs.
- Added regression tests for all new flows.

## v0.3.1-agent-audit

- Fixed canonical lineup/player replacement normalization.
- Fixed team context mismatch in the demo.
- Fixed identity helper truncation when `team_scope` was absent.
- Fixed paper tracking CLI and ledger stake summaries.
- Fixed Football-Data.co.uk day-first date parsing.
- Fixed stale feature/ELO state in walk-forward backtests with cached models.

## v0.6.0 - Operational audit

- Fixed lineup minutes precedence: observed lineup/substitution minutes now override event-adapter fallback minutes.
- Removed duplicate Wyscout `date` assignment in lineup conversion.
- Added bookmaker-compatible event rules documentation for shots on target, fouls/cards and Sustituto+.
- Added live operation checklist for real fixtures/odds paper-mode workflow.
- Updated model artifact code version to `0.6-operational-audit`.
- Re-ran compile, unit tests, scoped prediction, mismatch guard, backtesting, quality gate and event dataset builds.

## v0.8 - Operational Runtime & Event Validation

- Reworked historical validation to avoid full-history national-team crawls by default.
  - `international-results` now defaults to a modern window from 2010 unless `--full-history` is explicitly passed.
  - Added `--start-date`, `--end-date`, `--max-completed-matches`, and `--max-backtest-predictions`.
  - Backtesting now runs in chunked expanding windows instead of rebuilding ELO/features before every single match.
- Kept both Poisson and Random Forest available in the validation flow; Random Forest is no longer deferred, but its runtime is bounded.
- Goal model now drops all-null event features at fit time, so result-only datasets do not pretend to use shots/corners/cards.
- Added player-prop historical validation:
  - `scripts/validate_player_props.py`
  - `src/mundialytics/evaluation/player_props.py`
  - validates 1+ shots, 1+ shots on target, 1+ fouls committed, 1+ yellow card from processed player-events data.
- Added `data/sample/player_events_synthetic.csv` as a local smoke-test dataset for player-prop validation.
- Added regression tests for chunked backtesting, all-null feature handling, and player-prop backtesting.


## v0.9 - Real event-source gate

- Added a strict event-data readiness diagnostic so empty player-prop columns are no longer silently accepted.
- Added StatsBomb Open Data setup/download helper.
- Added one-command player props pipeline for StatsBomb Open Data.
- Improved StatsBomb event adapter with match metadata scanning and Bad Behaviour card extraction.
- Clarified bookmaker-like SOT rule: Goal, Saved, Saved to Post; ordinary blocks are excluded.
- Added regression tests for event coverage and metadata scanning.


## v0.18.1 — Today fixtures command

Added a real fixture discovery step so examples no longer require an invented `fixture_id`. Use:

```powershell
$env:API_FOOTBALL_KEY="YOUR_KEY"
python scripts/fetch_today_fixtures.py `
  --timezone America/New_York `
  --out outputs/api_football_today_fixtures_et.csv `
  --raw-out outputs/api_football_today_fixtures_et.json
```

The command prints a compact table with real `fixture_id` values. Use one of those IDs in `fetch_api_football_lineups.py`. See `docs/V18_1_TODAY_FIXTURES.md`.

## v0.18.2 — World Cup fixtures command

- Added `scripts/fetch_world_cup_fixtures.py`.
- Added `--world-cup` shortcut to `scripts/fetch_api_football_fixtures.py`, mapping to API-Football `league=1`, `season=2026`.
- Added local calendar-date post-filtering in the requested timezone to avoid yesterday/tomorrow fixture leakage.
- Added tests for the World Cup shortcut and local-date filtering.

## v0.19 — Daily MVP flow, free fixtures and team props baseline

- Replaced the API-Football-only daily fixture path with a free/keyless provider stack:
  - SofaScore scheduled-events as primary.
  - ESPN public scoreboard as fallback.
- Rewrote `scripts/fetch_today_fixtures.py` to support `--competition world_cup`, `--provider auto`, local timezone filtering and raw JSON caching.
- Added ESPN scoreboard adapter.
- Added SofaScore lineup parser and `scripts/fetch_fixture_lineups_free.py` preserving provider player/team IDs.
- Added team/match stats layer:
  - `src/mundialytics/features/team_match_stats.py`
  - `scripts/build_team_match_stats.py`
  - `scripts/validate_team_props.py`
  - `scripts/predict_team_props.py`
  - `scripts/calibrate_team_props.py`
- Added `scripts/build_daily_report.py` for a simple HTML daily product report.
- Integrated optional team props output into `scripts/run_matchday_analysis.py`.
- Added v0.19 docs and audit report.
- Added tests for ESPN fixtures, SofaScore lineups, timezone filtering and team props.

## v0.22.0 - Evaluation, Calibration & Competition Forecast

- Added temporal holdout evaluation for match probabilities.
- Added calibration bins and shrinkage calibration model export.
- Added optional calibration application in `run_statistical_matchday.py`.
- Added `--clean-out-dir` to avoid stale accumulated outputs.
- Added `--no-demo-picks` to block recommendations from demo odds.
- Added football.meets.data-style outputs: top scorer, awards approximation, competition summary.
- Added tests for evaluation/calibration, competition forecast, clean output directory and demo-pick blocking.

## v0.27.0 - Rolling & Segment Hardening
- Added rolling-origin match model validation with train -> calibration -> future test folds.
- Added `scripts/run_rolling_model_lab.py` and rolling reports/leaderboards.
- Added segment-level deployment policies to `prediction_registry.json` for player props.
- Added additional v0.27 player-prop challenger configs.
- Added regression tests for rolling validation and segment policy registry.

## v0.28.0 - Position Groups & Role Guardrails
- Added normalized tactical position keys and coarse position groups for player-prop modelling.
- Added group-level position priors and role-aware calibration levels: competition+position_group, domain+position_group and position_group.
- Added goalkeeper guardrails for attacking player props: goalkeepers are blocked for shots and shots-on-target value.
- Added v0.28 player-prop challenger configs focused on group calibration, winger/forward softening and conservative card behaviour.
- Added regression tests for position grouping and goalkeeper attacking-prop blocking.

## v0.29.0 - Dynamic Lines & Structured Evidence
- Added `src/mundialytics/statistical_core/dynamic_lines.py`.
- Added `dynamic_market_lines.csv` to the statistical matchday output.
- Generates dynamic over/under lines for goals, shots, shots on target, fouls, yellow cards, corners and player props.
- Supports match, team and player scopes instead of forcing one fixed line such as Over 2.5 goals.
- Adds structured evidence columns instead of free-form justifications: recent hit rate, similar-Elo hit rate, recent H2H hit rate, evidence tags, reason codes, data quality flags and value labels.
- Adds configurable recency cutoffs for H2H and similar-Elo evidence.
- Keeps corners as `not_available` unless reliable corner data exists.
- Adds report section for dynamic market lines and tests for structured evidence and goalkeeper attacking-prop guardrails.

## v0.46 - OddsPapi Historical Odds Backfill
- Added historical fixture planning, fetching, matching, raw odds backfill, snapshot odds, training features, and coverage audit scripts.
- Enforced leakage-safe pre-match snapshots before model training/backtesting.
- Added target-market filtering for 1x2, goals, corners, cards, shots, SOT, fouls, saves, and player props.

## v0.50.0 — Advanced Football Data Acquisition Layer

- Added a multi-provider advanced football data layer for xG, npxG, xA, shot quality, possession, progression, defensive actions, goalkeeper fields, player match stats and shot events.
- Added canonical contracts for `canonical_advanced_match_stats.csv`, `canonical_player_match_stats.csv` and `canonical_shot_events.csv`.
- Added best-effort FBref/soccerdata downloader, provider/manual CSV importer, Kaggle Understat CSV importer, StatsBomb Open Data advanced event importer, provider-priority merge, match enrichment and coverage audit scripts.
- Added leakage policy documentation: current-match advanced fields are post-match observations; only prior rolling features may be model inputs.
- Added compatibility data helpers so the ZIP remains importable as a standalone project snapshot.
- Added tests for the v0.50.0 advanced data layer.
