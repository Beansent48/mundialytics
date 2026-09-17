# Spec v0.50.3 — FBref Team-Match Normalization Repair

## Objective

Repair normalization of soccerdata/FBref team-match raw exports so that one-row-per-team match files are converted into the canonical home/away advanced match contract.

## Context

The local raw FBref files contain useful team/opponent/date/venue data and metric columns, but the prior normalized `fbref_shooting_advanced_match_stats.csv` contained empty `home_team`, `away_team`, `home_shots`, `home_sot` and xG fields. The root cause is that soccerdata flattens FBref multi-index metric columns into names such as `Standard.1` and `Performance.2`, which the generic importer did not understand.

## Scope

Included:

- Map FBref shooting raw columns:
  - `Standard.1` -> shots
  - `Standard.2` -> shots on target
  - `Standard.6` -> free-kick shot count helper
  - `Standard.7` -> penalty goal count helper
- Map FBref misc raw columns:
  - `Performance` -> yellow cards
  - `Performance.1` -> red cards
  - `Performance.3` -> fouls
  - `Performance.7` -> interceptions
  - `Performance.8` -> tackles won
- Map FBref keeper raw columns:
  - `Performance.1` -> keeper goals against
  - `Performance.2` -> keeper saves
  - `Performance.3` -> keeper save percentage
- Map FBref schedule raw `Poss` -> possession.
- Keep xG/npxG empty when the raw FBref export does not actually contain xG columns.
- Preserve existing Football-Data shots/SOT in enriched datasets.

Excluded:

- Claiming FBref xG coverage when current raw exports do not contain xG.
- Player ratings or player Elo.
- Understat/Kaggle import implementation.
- Optimizing pandas fragmentation warnings.

## Acceptance criteria

- `fbref_shooting_advanced_match_stats.csv` should have `home_team`, `away_team`, `home_shots`, `away_shots`, `home_sot`, `away_sot` populated after re-import.
- `fbref_keeper_advanced_match_stats.csv` should have keeper saves/goals against/save pct populated after re-import.
- `fbref_misc_advanced_match_stats.csv` should have discipline/fouls/interceptions/tackles populated after re-import.
- xG remains null for FBref shooting unless a future raw export includes semantic xG columns.
- Existing tests pass.

## Validation run

Validated with:

```bash
python -m compileall -q src scripts
python -m pytest -q tests/test_v0495_hybrid_model_ready_snapshots.py tests/test_v0496_external_data_enrichment.py tests/test_v0500_advanced_football_data_layer.py
```

Observed result: `14 passed`.

Warnings remain for pandas fragmentation and are tracked as technical debt.
