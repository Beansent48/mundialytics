# v0.50.0 — Advanced Football Data Acquisition Layer

Status: Implemented  
Date: 2026-06-25  
Version: 0.50.0

## 1. Context

The user decided that xG is strategically important because it measures chance quality and match process better than final-score finishing variance. The search for a stable free source showed that a single free provider is not enough:

```text
Understat direct scraping: useful historically, but fragile.
StatsBomb Open Data: official/free/high quality, but partial coverage.
FBref/soccerdata: promising free Big 5 advanced stats path.
worldfootballR: useful FBref/Understat/Transfermarkt export path.
Kaggle Understat datasets: useful historical backfill when licensing allows it.
RapidAPI Football xG Statistics: useful freemium/smoke/incremental path.
Manual/provider CSV import: required universal fallback.
```

The project therefore needs an advanced data layer, not only a single xG downloader.

## 2. Core decision

Implement a multi-provider advanced-football-data layer with canonical contracts for:

```text
advanced match stats
player match stats
shot events
```

The statistical engine should eventually learn from:

```text
chance quality: xG, npxG, xG per shot
chance creation: xA, shot-creating actions, key passes
territory/control: possession, touches in box, progressive passes/carries
defensive resistance: tackles, interceptions, blocks, clearances, pressures
goalkeeping: saves, post-shot xG where available
player form: minutes, xG, xA, shots, assists, cards
```

## 3. Provider priority

Default priority:

```text
1. FBref / soccerdata / worldfootballR exports
2. Kaggle Understat backfill
3. RapidAPI Football xG Statistics export
4. StatsBomb Open Data
5. Understat direct fallback
6. Manual/provider CSV
```

This order is configurable through `scripts/merge_advanced_sources.py`.

## 4. Leakage policy

Current-match advanced stats are **post-match observations**.

Examples:

```text
home_xg
away_xg
home_possession
home_progressive_passes
home_keeper_psxg
```

They may be used as:

```text
targets
diagnostics
historical evidence
```

They may **not** be used directly as pre-match features for the same match.

Allowed model features are rolling/prior snapshots such as:

```text
rolling_xg_for_last5
rolling_xg_against_last5
rolling_xg_per_shot_last5
rolling_xa_for_last5
rolling_progressive_passes_last5
rolling_keeper_psxg_minus_ga_last5
```

## 5. Implemented contracts

### `canonical_advanced_match_stats.csv`

Core fields include:

```text
provider
provider_match_id
date
competition
season
home_team
away_team
home_xg / away_xg
home_npxg / away_npxg
home_xa / away_xa
home_shots / away_shots
home_sot / away_sot
home_avg_shot_distance / away_avg_shot_distance
home_possession / away_possession
home_touches_box / away_touches_box
home_progressive_passes / away_progressive_passes
home_progressive_carries / away_progressive_carries
home_tackles / away_tackles
home_interceptions / away_interceptions
home_blocks / away_blocks
home_keeper_psxg / away_keeper_psxg
source_confidence
join_method
```

### `canonical_player_match_stats.csv`

Core fields include:

```text
provider
provider_match_id
match_id
date
team
opponent
player
player_id
position
started
minutes
goals
assists
xg
npxg
xa
xg_chain
xg_buildup
shots
sot
key_passes
sca
gca
progressive_passes
progressive_carries
touches
touches_box
tackles
interceptions
blocks
pressures
yellow_cards
red_cards
saves
psxg
```

### `canonical_shot_events.csv`

Core fields include:

```text
provider
provider_match_id
match_id
date
team
opponent
player
minute
second
xg
psxg
outcome
body_part
situation
shot_type
assisted_by
x
y
is_penalty
is_goal
```

## 6. Added modules

```text
src/mundialytics/enrichment/advanced.py
```

## 7. Added scripts

```text
scripts/download_fbref_advanced.py
scripts/import_advanced_csv.py
scripts/import_kaggle_understat.py
scripts/import_statsbomb_open_advanced.py
scripts/merge_advanced_sources.py
scripts/enrich_matches_with_advanced_stats.py
scripts/audit_advanced_data_coverage.py
```

## 8. Intended usage

### 8.1 Best-effort FBref download with optional soccerdata

```powershell
python scripts/download_fbref_advanced.py `
  --league "ENG-Premier League" `
  --season 2021 2022 2023 2024 2025 `
  --stat-type schedule shooting keeper misc `
  --out-dir data/external/advanced/fbref/epl
```

If `soccerdata` is unavailable or FBref blocks access, the script writes a blocked report instead of pretending success.

### 8.2 Import provider/manual advanced CSV

```powershell
python scripts/import_advanced_csv.py `
  --input data/external/advanced/fbref/epl/fbref_team_match_stats_shooting.csv `
  --provider fbref `
  --out-dir data/external/advanced/fbref/epl_normalized
```

### 8.3 Import Kaggle/Understat backfill CSV

```powershell
python scripts/import_kaggle_understat.py `
  --input data/external/advanced/kaggle_understat/games.csv `
  --out-dir data/external/advanced/kaggle_understat
```

### 8.4 Import StatsBomb Open Data advanced aggregates

```powershell
python scripts/import_statsbomb_open_advanced.py `
  --data-dir data/raw/statsbomb/open-data/data `
  --out-dir data/external/advanced/statsbomb
```

### 8.5 Merge providers

```powershell
python scripts/merge_advanced_sources.py `
  --source fbref=data/external/advanced/fbref/epl_normalized/fbref_advanced_match_stats.csv `
           kaggle_understat=data/external/advanced/kaggle_understat/kaggle_understat_advanced_match_stats.csv `
           statsbomb_open_data=data/external/advanced/statsbomb/statsbomb_advanced_match_stats.csv `
  --provider-priority fbref kaggle_understat rapidapi_football_xg statsbomb_open_data understat provider_csv `
  --out-dir data/external/advanced/canonical
```

### 8.6 Enrich canonical matches

```powershell
python scripts/enrich_matches_with_advanced_stats.py `
  --matches data/processed/enriched/epl_clubelo/canonical_matches_with_clubelo.csv `
  --advanced data/external/advanced/canonical/canonical_advanced_match_stats.csv `
  --registry data/processed/entities/team_registry.csv `
  --out-dir data/processed/enriched/epl_clubelo_advanced `
  --dataset-name epl_clubelo_advanced
```

### 8.7 Audit coverage

```powershell
python scripts/audit_advanced_data_coverage.py `
  --matches data/processed/enriched/epl_clubelo_advanced/canonical_matches_with_advanced_stats.csv `
  --out-dir data/processed/enriched/epl_clubelo_advanced `
  --dataset-name epl_clubelo_advanced
```

## 9. What this phase does not do

This phase does not claim that every free provider covers every Big 5 match. It creates the professional acquisition, merge and audit layer so every provider can be used when available and honestly reported when coverage is partial.

The next modelling phase should connect advanced enriched canonical matches to model-ready snapshots and then evaluate whether rolling advanced features improve:

```text
1X2 log loss
RPS
goal MAE/RMSE
scoreline log loss
BTTS/over-under calibration
corner/card models
player/award/event models
```

## 10. Validation

Implemented tests:

```text
tests/test_v0500_advanced_football_data_layer.py
```

Validated behaviours:

```text
StatsBomb Open Data advanced import creates match/player/shot outputs.
Provider CSV import canonicalizes xG/xA/possession-style fields.
Provider merge selects canonical rows by priority.
Enrichment joins advanced stats to canonical matches.
Coverage audit reports xG, npxG, xA, shot quality, possession, progression, territory, defence and goalkeeping groups.
```
