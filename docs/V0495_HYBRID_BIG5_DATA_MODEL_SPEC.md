# v0.49.5 — Hybrid Big 5 Data Model & Model-Ready Snapshots Spec

Status: Implemented  
Date: 2026-06-25  
Version: 0.49.5

## 1. Context

After v0.49.4, local data-foundation runs showed that the Big 5 Football-Data.co.uk datasets are strong enough to become the first serious club-data foundation.

Observed local foundation results from the user:

```text
LaLiga:      1900 rows, 5 seasons, goals/shots/SOT/corners/fouls/yellows available
Premier:     1900 rows, 5 seasons, goals/shots/SOT/corners/fouls/yellows available
Serie A:     1900 rows, 5 seasons, goals/shots/SOT/corners/fouls/yellows available
Bundesliga:  1530 rows, 5 seasons, almost complete shots/SOT/corners/fouls/yellows
Ligue 1:     1677 rows, 5 seasons, goals/shots/SOT/corners/fouls/yellows available
```

The data foundation confirmed that model improvement should begin with better data treatment and feature snapshots, not with blind model-family changes.

The user selected a hybrid club modelling direction:

```text
global Big 5 club model
+ explicit league/context features
+ league-level diagnostics/calibration
+ team rolling form/stat features
+ internal Elo
+ optional ClubElo/external Elo
+ optional xG/event enrichments
```

## 2. Core Decision

Use the hybrid option for club football:

```text
Train with broad Big 5 data because volume adds value.
Still make the league visible to the model and evaluation because leagues behave differently.
```

This means:

- do not train one narrow model per league as the default first path,
- do not blindly pool leagues without league features,
- do not mix club and national rows,
- always report global metrics and by-league metrics,
- make optional enrichments such as xG and ClubElo additive, not mandatory.

## 3. Scope

This phase implements the next data layer:

```text
canonical_matches.csv
→ model_ready_match_snapshots.csv
→ model_ready_feature_contract.csv
→ model_ready_snapshot_report.json
```

The new snapshot layer is designed to be the model input contract for future hybrid Big 5 experiments.

## 4. Non-Goals

This phase does not:

- replace the Poisson baseline,
- train a new global Big 5 model,
- activate xG as a required model input,
- download xG/ClubElo automatically,
- implement league calibration gates,
- implement corners/cards Negative Binomial models,
- optimize profit, ROI, staking or value picks,
- mutate raw files.

## 5. Implemented Files

New:

```text
src/mundialytics/data_quality/model_ready_snapshots.py
scripts/build_model_ready_dataset.py
tests/test_v0495_hybrid_model_ready_snapshots.py
docs/V0495_HYBRID_BIG5_DATA_MODEL_SPEC.md
```

Updated:

```text
src/mundialytics/features/team_features.py
src/mundialytics/data_quality/__init__.py
docs/PROJECT_CONTINUITY.md
docs/DECISIONS.md
docs/MODEL_DESIGN.md
docs/DATA_SOURCE_STRATEGY.md
docs/NEXT_VALIDATION_STEPS.md
README.md
CHANGELOG.md
pyproject.toml
src/mundialytics/__init__.py
```

## 6. Data Architecture

Accepted architecture:

```text
data/raw/*
  ↓
canonical_matches.csv
  ↓
model_ready_match_snapshots.csv
  ↓
historical validation / model selection / simulation
```

Future enrichments fit between canonical and model-ready:

```text
canonical_matches.csv
  ↓
optional enrichments:
  - team registry / provider aliases
  - ClubElo or external Elo
  - xG / event data
  - player/squad summaries
  ↓
model_ready_match_snapshots.csv
```

## 7. Snapshot Contract

`model_ready_match_snapshots.csv` has one row per match.

It deliberately contains:

### Identity/context columns

```text
match_id
date
competition
season
stage
team_scope
home_team
away_team
neutral
```

### Pre-match features

Examples:

```text
league_match_count_pre
league_goal_rate_pre
league_home_goal_rate_pre
league_away_goal_rate_pre
league_draw_rate_pre
league_btts_rate_pre
league_over25_rate_pre
home_elo_pre
away_elo_pre
elo_diff_pre
expected_home_score_elo_pre
home_goals_for_last5
away_goals_for_last5
home_shots_for_last5
away_shots_for_last5
home_corners_for_last5
away_corners_for_last5
home_yellow_cards_for_last5
away_yellow_cards_for_last5
home_xg_for_last5
away_xg_for_last5
```

### Post-match targets

Targets are included for training/evaluation convenience but must never be passed as model features:

```text
target_home_goals
target_away_goals
target_total_goals
target_1x2
target_btts
target_home_shots
target_away_shots
target_home_sot
target_away_sot
target_home_corners
target_away_corners
target_home_fouls
target_away_fouls
target_home_yellow_cards
target_away_yellow_cards
```

The companion `model_ready_feature_contract.csv` marks each column as:

```text
identity
feature
target
```

and tags target columns as:

```text
post_match_target_not_feature
```

## 8. Leakage Policy

This phase establishes a hard leakage rule:

```text
The model may only consume feature columns that were known before kickoff.
```

Rolling features are shifted by one prior team match. League context features are cumulative prior rates only. Elo columns are pre-match Elo values.

Examples:

- A match on `2025-03-01` may use a team's previous matches.
- It may not use season-end averages that include future matches.
- It may not use the current match's shots, corners, goals or cards as input.
- It may keep current match targets in the same dataset only if the feature contract excludes them.

## 9. Hybrid Big 5 Model Direction

The accepted future model direction is:

```text
global club model trained on Big 5 snapshots
+ league indicators/context
+ league-level diagnostics
+ optional league-level calibration
```

This is preferred over:

- isolated per-league models as the default first path,
- a pooled model with no league awareness.

The Premier League local validation showed why this matters: the model overestimated total goals more strongly in EPL than in LaLiga. Therefore, league context is not optional in evaluation.

## 10. xG Policy

xG is useful but optional.

The system should support three data levels:

```text
Level 1:
goals + shots + SOT + corners + cards + internal Elo

Level 2:
Level 1 + ClubElo/external Elo

Level 3:
Level 2 + xG/xA/event features
```

Rules:

- xG must never be required for baseline operation.
- If xG coverage exists, rolling xG features should be added as pre-match features.
- If xG coverage is missing, the snapshot builder still works and reports `xg_features_available=false`.
- xG providers must be cached/provenanced and should not silently overwrite raw provider data.

## 11. Club vs National Split

Accepted interpretation:

```text
Club data:
- strongest for player/event evidence,
- useful for player form, minutes, shots, goals, xG and props,
- useful for club-level statistical models.

National data:
- primary source for national-team results,
- primary source for national-team Elo/results modelling,
- primary source for international context such as tournament/friendly/neutral.
```

For World Cup/Mundialytics use:

```text
team result engine:
national-first

player/award engine:
club evidence + national squad eligibility + expected minutes + tournament progression
```

Club and national rows must not be trained in the same base model unless a later explicit cross-context model is designed and validated.

## 12. Corners and Cards Direction

Big 5 foundations now show enough coverage for corners, fouls and yellow cards to open separate count models later.

Direction:

```text
corners:
- separate count model
- likely Poisson / Negative Binomial comparison
- outlier policy for extreme corners
- evaluate lines such as 8.5, 9.5, 10.5, 11.5

cards:
- separate count model
- likely Negative Binomial if overdispersed
- referee data should be added if available
- yellow cards usable now; red-card coverage currently absent in Football-Data CSV foundation
```

Corners/cards should first be used as pre-match rolling features, then as targets for dedicated models.

## 13. Command

Build model-ready snapshots from a canonical foundation dataset:

```powershell
python scripts/build_model_ready_dataset.py `
  --matches data/processed/foundation_laliga_multi_season/canonical_matches.csv `
  --out-dir data/processed/model_ready_laliga_multi_season_v0495 `
  --dataset-name model_ready_laliga_multi_season_v0495
```

Expected outputs:

```text
model_ready_match_snapshots.csv
model_ready_feature_contract.csv
model_ready_snapshot_report.json
```

## 14. Validation

Focused validation performed during implementation:

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

The known `artifact_tool` spreadsheet warmup warning may appear on stderr; the commands completed with exit code `0`.

## 15. Next Intended Work

Recommended next phase:

```text
v0.49.6 — Big 5 Snapshot Evaluation & Hybrid Model Selection
```

Planned goals:

- build model-ready snapshots for each Big 5 league,
- optionally concatenate them into a Big 5 global snapshot dataset,
- validate Poisson/RF baselines using the snapshot contract,
- compare global vs by-league metrics,
- decide whether to add league-level lambda calibration,
- decide when to use xG/ClubElo if coverage is added.
