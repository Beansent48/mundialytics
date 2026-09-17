# v0.49.9 — Free xG Provider: StatsBomb Open Data

## Purpose

Use a free and more stable xG/event source when direct Understat scraping is blocked.

The selected free source is **StatsBomb Open Data** because it is official open data and is already present in many local Mundialytics data folders under:

```text
data/raw/statsbomb/open-data/data
```

This phase does **not** pretend that StatsBomb Open Data covers every Big 5 match. Its purpose is to add the best free, stable xG/event path available without API keys, while keeping the project honest about coverage.

## Decisions

### 1. Understat is no longer the primary free xG path

Understat direct scraping remains a research fallback only. If it reports blocked markup or cannot find inline JSON, the pipeline should continue without xG rather than fail.

### 2. StatsBomb Open Data is the preferred free xG/event import path

StatsBomb Open Data is used offline from a local checkout. The importer does not call the paid StatsBomb API and does not require credentials.

### 3. xG coverage is partial by design

StatsBomb Open Data coverage depends on the competitions and matches available in the local open-data folder. It is valid enrichment when matched, but missing coverage must be explicit through `xg_available = false`.

### 4. xG from the current match is target/post-match data

The match-level `home_xg`, `away_xg`, `home_npxg`, and `away_npxg` values describe the match after it happened. They may be used as targets, diagnostics and historical evidence. The model may only consume rolling/pre-match xG features derived from earlier matches.

## Added scripts

```text
scripts/import_statsbomb_open_xg.py
```

Reads:

```text
data/raw/statsbomb/open-data/data/competitions.json
data/raw/statsbomb/open-data/data/matches/<competition_id>/<season_id>.json
data/raw/statsbomb/open-data/data/events/<match_id>.json
```

Writes:

```text
data/external/xg/statsbomb/statsbomb_xg_matches.csv
data/external/xg/statsbomb/statsbomb_xg_shots.csv
data/external/xg/statsbomb/statsbomb_xg_import_report.json
```

## Canonical match-level xG contract

```text
provider
provider_match_id
date
competition
season
home_team
away_team
home_xg
away_xg
home_npxg
away_npxg
xg_match_confidence
```

## Canonical shot-level event contract

```text
provider
provider_match_id
date
competition
season
team
opponent
is_home_team
player
minute
second
xg
npxg
shot_type
body_part
outcome
technique
under_pressure
x
y
```

## Operational flow

```powershell
python scripts/import_statsbomb_open_xg.py `
  --data-dir data/raw/statsbomb/open-data/data `
  --out-dir data/external/xg/statsbomb
```

Then enrich a canonical match dataset:

```powershell
python scripts/enrich_matches_with_xg.py `
  --matches data/processed/enriched/epl_clubelo/canonical_matches_with_clubelo.csv `
  --xg data/external/xg/statsbomb/statsbomb_xg_matches.csv `
  --registry data/processed/entities/team_registry.csv `
  --provider statsbomb_open_data `
  --provider-alias-column statsbomb_name `
  --out-dir data/processed/enriched/epl_clubelo_statsbomb_xg `
  --dataset-name epl_clubelo_statsbomb_xg `
  --allow-missing-xg
```

## Validation

The importer is covered by focused tests with local synthetic StatsBomb-style data. It validates:

- match-level xG aggregation from shot events;
- non-penalty xG exclusion for penalties;
- canonical match-level output;
- shot-level output;
- downstream xG enrichment compatibility.

## Non-goals

- No paid provider integration.
- No live API calls.
- No guarantee of full Big 5 coverage.
- No model logic changes.
