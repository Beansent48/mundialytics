# Data source strategy for Mundialytics

The prediction engine needs more than final scores. The practical data stack is layered by use case.


## Data foundation rule

Before model improvements, source files should be converted into a cleaned canonical match dataset and profiled with:

```bash
python scripts/build_match_dataset.py
```

The foundation layer writes:

```text
canonical_matches.csv
match_dataset_foundation_report.json
match_dataset_feature_coverage.csv
match_dataset_quality_by_competition_season.csv
match_dataset_anomalies.csv
match_dataset_dropped_rows.csv
```

Use these reports to decide which model families are valid:

- goals/1X2 require reliable goal coverage,
- scoreline simulation benefits from multi-season goal history,
- corners models require corners coverage,
- cards models require yellow/red-card coverage,
- shot/player models require shots, shots-on-target, lineups and event data,
- external Elo/ClubElo should be used only when the foundation report confirms coverage.

Raw files in `data/raw` should remain immutable. Cleaned/profiled datasets belong in `data/processed` or `outputs`.


## v0.49.5 — Data levels for the hybrid Big 5 model

The accepted club-data architecture is hybrid:

```text
global Big 5 club model
+ league/context features
+ league-level diagnostics/calibration
```

Use data in levels so the engine can improve without becoming brittle:

```text
Level 1: Football-Data foundation
- goals
- shots
- shots on target
- corners
- fouls
- yellow cards
- internal Elo

Level 2: ratings enrichment
- ClubElo or other external Elo/rating snapshots

Level 3: event/xG enrichment
- xG
- non-penalty xG
- xA/shot assist proxies
- player event rates
```

Rules:

- Level 1 must remain sufficient for baseline operation.
- Level 2 and Level 3 are optional enrichments.
- xG is valuable and should be pursued, but missing xG must not block baseline training/validation.
- Provider enrichments must be cached/provenanced and must not mutate `data/raw` source files.
- Every enrichment must enter through the model-ready snapshot contract and be marked as pre-match safe.

Current local Big 5 foundations showed strong Level 1 coverage:

```text
LaLiga, Premier League, Serie A, Bundesliga, Ligue 1
→ goals, shots, shots on target, corners, fouls and yellow cards are available or nearly complete.
```

`xG`, `external_elo` and `clubelo` are currently unavailable in the Football-Data CSV foundations and should be added through explicit enrichment steps, not assumed.

## v0.49.5 — Model-ready snapshot layer

After the data foundation step, build leakage-safe snapshots:

```powershell
python scripts/build_model_ready_dataset.py `
  --matches data/processed/foundation_laliga_multi_season/canonical_matches.csv `
  --out-dir data/processed/model_ready_laliga_multi_season_v0495 `
  --dataset-name model_ready_laliga_multi_season_v0495
```

Outputs:

```text
model_ready_match_snapshots.csv
model_ready_feature_contract.csv
model_ready_snapshot_report.json
```

The feature contract is mandatory for future training code: targets may exist in the same file but must not be used as features.


## Tier A — match results, odds and match-level stats

### Football-Data.co.uk
Best free source for club match-result modelling and value-betting backtests.

Use it for:
- full-time result,
- home/away goals,
- shots, shots on target, corners, fouls and cards where available,
- historical bookmaker odds for 1X2, totals and handicaps.

Limitations:
- mostly club football,
- weak for player props,
- no event-level tactical sequence.

### international-results
Best free source for national-team ELO and goal models.

Use it for:
- national-team historical results,
- tournament/competition context,
- neutral-ground flags,
- long-run ELO training.

Limitations:
- no player-event data,
- no betting odds,
- no lineups/minutes.

## Tier B — event-level player data

### StatsBomb Open Data
Best high-quality open source for player-event modelling.

Use it for:
- shots and shot outcomes,
- passes, shot assists and goal assists,
- fouls committed/won,
- cards,
- pressures, duels, dribbles, interceptions and recoveries,
- Starting XI events,
- Substitution events,
- Tactical Shift events,
- 360 freeze-frame context where available.

Limitations:
- coverage is open but selective, not full daily coverage.

Implemented in this project:

```bash
python scripts/build_event_datasets.py statsbomb \
  --input data/raw/statsbomb/open-data/data/events \
  --competition "StatsBomb Open Data" \
  --team-scope club \
  --player-events-out data/processed/statsbomb_player_events.csv \
  --team-events-out data/processed/statsbomb_team_events.csv \
  --lineups-out data/processed/statsbomb_lineups.csv \
  --tactical-out data/processed/statsbomb_tactical_shifts.csv
```

### Wyscout public event dataset
Best open large event dataset for broad player-event modelling across several competitions.

Use it for:
- passes,
- shots,
- fouls,
- cards,
- duels,
- interceptions/recoveries,
- players and teams metadata,
- lineups/substitutions when match metadata is provided.

Limitations:
- historical snapshot, not current daily feed,
- provider schema differs from StatsBomb,
- fouls drawn/opponent attribution is limited compared with richer provider feeds.

Implemented in this project:

```bash
python scripts/build_event_datasets.py wyscout \
  --events data/raw/wyscout/events_England.json \
  --matches data/raw/wyscout/matches_England.json \
  --players data/raw/wyscout/players.json \
  --teams data/raw/wyscout/teams.json \
  --competition "Premier League" \
  --season 2017-2018 \
  --team-scope club \
  --player-events-out data/processed/wyscout_player_events_england.csv \
  --team-events-out data/processed/wyscout_team_events_england.csv \
  --lineups-out data/processed/wyscout_lineups_england.csv
```

## Tier C — tactical/tracking context

### SkillCorner Open Data
Useful for experimenting with broadcast-tracking concepts such as physical metrics and dynamic events.

Use it for:
- tracking-style tactical experiments,
- physical metrics,
- movement/spacing features,
- validating concepts before seeking a paid/current data feed.

Limitations:
- small number of open matches,
- not enough for robust betting models alone.

### Metrica Sports sample data
Useful for learning tracking workflows and tactical analytics.

Use it for:
- pitch control experiments,
- formation/spacing prototypes,
- event + tracking integration examples.

Limitations:
- very small sample,
- anonymized teams/players,
- not suitable as the main modelling dataset.

## Tier D — current fixtures, lineups, injuries and odds

Free data is weakest here. For a real betting product, this layer usually needs one of:

- Betfair API for current prices/markets,
- manual odds export for paper mode,
- official competition fixtures or OpenFootball when updated,
- a paid/current provider for lineups, injuries and player props.

## Recommended practical stack

### Club 1X2 / totals

1. Football-Data.co.uk for historical results + odds.
2. ClubElo or internal ELO for team strength.
3. OpenFootball/manual CSV for upcoming fixtures.
4. Betfair/manual odds for current prices.
5. Backtest and paper ledger before trusting EV.

### National-team tournament forecasting

1. international-results for historical results.
2. internal ELO + optional external ELO benchmark.
3. StatsBomb Open Data for tournament player-event modelling when available.
4. Official/manual fixture file for upcoming tournament matches.
5. Simulate tournament and track predictions.

### Player props

1. StatsBomb Open Data for high-quality schema and tactical/substitution events.
2. Wyscout public dataset for larger historical event coverage.
3. Separate player contexts: player-global baseline + team/scope/competition context.
4. Minutes model from lineups/substitutions.
5. Sustituto+ adjustment only for markets where the bookmaker rule supports it.
6. Paper tracking before any real-money use.

## Important limitation

No single free public source gives everything: current fixtures, current lineups, advanced individual events, substitutions, tactical changes, odds and Betfair player-prop rules at daily production quality. The right approach is multi-source ingestion with provenance columns, strict scope validation and quality gates.


## Elo / ClubElo policy

Elo is a first-class feature family for Mundialytics.

Implemented baseline:

```text
src/mundialytics/ratings/elo.py
```

Use cases:

- internal Elo for every walk-forward validation,
- national-team Elo/rating features from historical international results,
- optional external Elo/ClubElo as benchmark or prior when canonical rows include it.

Accepted canonical columns:

```text
home_external_elo
away_external_elo
home_clubelo
away_clubelo
home_elo
away_elo
```

These become model-side features:

```text
external_team_elo
external_opponent_elo
external_elo_diff
```

Internal Elo remains the default because it is reproducible, offline and available for both clubs and national teams.


## v0.49.6 External enrichment sources

Implemented enrichment levels:

### ClubElo

Use:

```text
scripts/download_clubelo.py
scripts/enrich_matches_with_clubelo.py
```

ClubElo data is cached in:

```text
data/external/clubelo/daily/
```

The enrichment output belongs in:

```text
data/processed/enriched/
```

### xG

xG is optional and provider-agnostic. Accepted sources:

```text
Understat research CSVs
TheStatsAPI or other paid API exports
manual provider CSV exports
StatsBomb-derived match xG when coverage exists
```

Use:

```text
scripts/download_understat_xg.py
scripts/enrich_matches_with_xg.py
```

xG enrichments must be accompanied by:

```text
xg_enrichment_report.json
xg_join_report.csv
external_xg_matches_canonical.csv
```

### Team registry

All provider joins should use:

```text
data/processed/entities/team_registry.csv
```

This registry is editable and should be reviewed before trusting external joins.

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



## v0.49.9 free xG/event source

StatsBomb Open Data is the preferred free xG/event source. It should be imported from the local open-data repository, not from the paid API. Coverage is partial and must be measured before use. Understat direct scraping is no longer considered a dependable primary source.

Provider priority for xG:

1. Paid/full-coverage API or provider CSV when available.
2. StatsBomb Open Data for free official event/xG enrichment with partial coverage.
3. Understat direct scraping only as research best-effort fallback.

## v0.50.0 advanced provider strategy

Accepted direction:

```text
Do not depend on one xG source.
Use a multi-provider advanced data layer with priority, cache, coverage audit and leakage guardrails.
```

Provider roles:

```text
FBref/soccerdata/worldfootballR: primary free path for advanced Big 5 match/team/player stats when accessible.
Kaggle Understat: historical xG/xA/xGChain/xGBuildup backfill when the user has downloaded the dataset and accepts its license.
RapidAPI Football xG Statistics: freemium/incremental/smoke provider.
StatsBomb Open Data: official high-quality events and shot xG for covered matches, especially tournaments/player-event work.
Manual/provider CSV: universal fallback for any future export.
```

The canonical advanced data contracts are documented in `docs/V0500_ADVANCED_FOOTBALL_DATA_LAYER_SPEC.md`.
