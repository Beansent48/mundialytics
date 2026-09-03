# Next Data Foundation Requirements

This guide captures the data needed after v0.48.4 and the first offline data-audit layer introduced in v0.49.0. The v0.49.0 audit can identify schema, coverage, gap and entity-quality issues, but real data collection, provider normalization, deeper entity resolution and reproducible dataset building remain the next data-foundation phases.

## 1. Core Principle

Mundialytics should not invent data. If a market, stat, player prop or tournament output lacks reliable data, the output must be marked as:

```text
not_available
```

The next data phase should make this explicit by measuring coverage, gaps, entity quality and leakage risk.

## 2. Match Results Dataset

Minimum required columns:

```text
match_id
date
kickoff_utc
competition
season
stage
group
home_team
away_team
home_goals
away_goals
status
neutral
team_scope
team_type
competition_context
gender
source
```

Notes:

- `match_id` must be stable across predictions, results, odds and events.
- `status` should distinguish finished, postponed, abandoned, awarded and pending.
- `kickoff_utc` is needed to prove whether a prediction was generated before kickoff.
- `team_scope` should separate `club` and `national`.

## 3. Forward Prediction Snapshots

To evaluate the engine properly, store every prediction made before kickoff.

Recommended columns:

```text
run_id
generated_at_utc
model_version
config_version
data_version
match_id
kickoff_utc
home_team
away_team
competition
p_home_win
p_draw
p_away_win
lambda_home
lambda_away
p_over_05
p_over_15
p_over_25
p_over_35
p_btts
most_likely_score
evaluation_mode_candidate
```

Rules:

- `generated_at_utc < kickoff_utc` is required for `forward_evaluation`.
- If this is not true, label the evaluation as `retrospective_backtest`.
- Store the exact model/config/data versions used.

## 4. Scoreline Distribution Snapshots

Recommended columns:

```text
run_id
generated_at_utc
match_id
home_goals
away_goals
probability
model_version
data_version
```

Needed for:

- top-1 scoreline accuracy
- top-3 / top-5 coverage
- probability assigned to actual scoreline
- scoreline distribution diagnostics

## 5. Team Event Stats

Recommended actual/stat columns:

```text
match_id
team
opponent
is_home
minutes_context
shots
shots_on_target
corners
fouls
yellow_cards
red_cards
goalkeeper_saves
xg
source
```

Needed for:

- team stat model evaluation
- dynamic lines by team
- availability matrix
- identifying `not_available` markets

## 6. Player Events and Player Props

Minimum historical player-event columns:

```text
match_id
date
team
opponent
player_id
player
position
minutes
started
substitute
goals
assists
shots
shots_on_target
fouls_committed
fouls_drawn
yellow_cards
red_cards
source
```

Minimum current eligibility columns:

```text
match_id
team
player_id
player
position
current_squad_flag
lineup_status
expected_minutes
availability_status
source
```

Rules:

- Historical players can train models.
- Current player predictions require current squad/lineup eligibility.
- Retired/inactive players must not appear as live player-prop candidates unless present in current input.

## 7. Golden Boot / Top Scorer Requirements

Golden Boot modelling is a future objective and should be treated as player-prop/tournament hybrid modelling.

Required data:

```text
player_id
player
team
current_squad_flag
position
age_optional
club_optional
historical_goals
historical_shots
historical_xg_if_available
minutes_rate
expected_minutes_by_match
team_progression_probabilities
fixture_difficulty
penalty_taker_status_if_reliable
actual_goals_by_match
```

Key modelling need:

```text
player scoring probability per match
× expected minutes
× team attacking strength
× team progression probability
```

Do not predict Golden Boot seriously without current squads, expected minutes and tournament progression probabilities.

## 8. Odds and Market Data — Later Phase

Odds are not required for v0.48.4 evaluation. For future value betting evaluation, collect:

```text
snapshot_id
collected_at_utc
bookmaker
match_id
market
selection
line
odds_decimal
suspended
source
```

Needed for:

- edge
- EV
- CLV
- paper betting ROI/yield

Rules:

- Prediction quality and market value are separate.
- No ROI/yield claims without odds snapshots and paper logs.

## 9. Dataset Manifest

Every processed dataset should include:

```text
dataset_name
dataset_version
created_at_utc
source_files
row_count
date_min
date_max
competitions
team_scope
entity_counts
target_columns
feature_columns
known_limitations
leakage_warnings
```

## 10. Deferred v0.49 Data Phase

Recommended next data roadmap:

```text
v0.49.0 Data Audit Report
v0.49.1 Entity & Squad Guardrails
v0.49.2 Feature Availability Matrix
v0.49.3 Dataset Builder & Manifest
```

This should happen before a major dashboard/interface release.


## 11. v0.49.1 Entity & Squad Guardrails — Proposed

The next proposed phase is documented in:

```text
docs/V0491_ENTITY_SQUAD_GUARDRAILS_SPEC.md
```

Goal:

- strengthen stable `match_id` and provider identity checks;
- block unsafe club/national scope mixing;
- verify current squad and lineup eligibility before player-prop inference;
- detect historical-only players appearing as current candidates;
- emit explicit reason codes for unsafe records.

This phase should remain offline and conservative. It should not add new model logic, API calls, live betting, dashboard work or Golden Boot modelling.


## 12. v0.49.0 Audit Outputs

The first offline audit layer writes:

```text
data_audit_summary.json
data_audit_report.csv
coverage_report.csv
data_gaps_report.csv
entity_quality_report.csv
feature_availability_matrix.csv
next_data_requirements.csv
data_audit_report.html
```

These outputs are diagnostics. They do not replace the need to collect real provider data, stable IDs, forward prediction snapshots and current squad/lineup evidence.

Recommended next implementation phases:

```text
v0.49.1 entity/provider identity hardening
v0.49.2 feature availability and coverage by competition/market
v0.49.3 reproducible dataset builder and dataset manifests
```


## v0.49.5 Update — Model-Ready Snapshot Contract

After the v0.49.4 foundation step, the accepted next data artifact is:

```text
model_ready_match_snapshots.csv
```

with companion contract:

```text
model_ready_feature_contract.csv
```

This contract must separate:

```text
identity columns
pre-match feature columns
post-match target columns
```

The future hybrid Big 5 club model should consume only columns marked as `feature`.
Targets may live in the same file for training/evaluation convenience, but they must never be passed as model inputs.

Accepted hybrid data direction:

```text
global Big 5 club model
+ league/context features
+ team rolling features
+ internal Elo
+ optional external Elo/ClubElo
+ optional xG/event enrichments
```

xG should be pursued as an optional enrichment, not as a mandatory dependency.
Club data should be used heavily for player/event evidence. National-team match outcomes should remain national-first, using national results, national Elo and international context.


## v0.49.6 Data enrichment requirements now implemented

The repository now includes practical scripts for:

```text
team registry generation
ClubElo daily snapshot caching
ClubElo match enrichment
optional Understat xG research download
provider/manual xG enrichment
enriched model-ready snapshots
```

Before training with enriched data:

1. review `team_registry.csv`,
2. inspect `clubelo_join_report.csv`,
3. inspect `xg_join_report.csv`,
4. confirm coverage in the enrichment reports,
5. rebuild `model_ready_match_snapshots.csv`,
6. validate against the Level 1 baseline.


## v0.49.8 xG acquisition rule

xG is valuable but optional. The data foundation must distinguish:

```text
xG desired
xG unavailable
xG imported from provider/manual CSV
xG matched to canonical fixtures
```

Direct Understat scraping can be blocked by provider markup changes. The canonical fallback is:

```text
provider/manual CSV → scripts/import_xg_csv.py → understat_xg_matches.csv
```

The model-ready pipeline must continue with xG coverage `0` when no reliable xG source exists.


## v0.49.9 xG data requirement

The model-ready layer now accepts free StatsBomb Open Data xG. Required checks before modelling with xG:

- `statsbomb_xg_import_report.json` must show match and shot rows.
- Each league enrichment must report matched coverage.
- xG features must be rolling/pre-match only.
- Current-match xG remains target/diagnostic data.

## v0.50.0 advanced data requirements

Advanced data is now a first-class data foundation layer.

Minimum requirements before using an advanced provider in modelling:

```text
provider report exists
canonical_advanced_match_stats.csv exists
advanced_data_coverage_report.json exists
coverage by league/season is reviewed
current-match advanced fields are not used directly as pre-match features
provider license/terms are acceptable for the intended use
```

Preferred feature groups:

```text
xG/npxG
xA/creative quality
shot quality
possession/control
territory/progression
defensive resistance
goalkeeping
player match form
shot events
```
