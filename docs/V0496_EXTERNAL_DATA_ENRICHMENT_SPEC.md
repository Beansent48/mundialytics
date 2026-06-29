# v0.49.6 — External Data Enrichment & Feature Expansion Spec

Status: Implemented  
Date: 2026-06-25  
Version: 0.49.6

## 1. Context

v0.49.5 created the hybrid Big 5 model-ready snapshot contract:

```text
canonical_matches.csv
→ model_ready_match_snapshots.csv
→ model_ready_feature_contract.csv
→ model_ready_snapshot_report.json
```

The next user requirement was explicit:

```text
obtener los datos que hemos añadido ahora
```

That means the repository must not only reserve columns for richer data. It must provide practical, reproducible scripts to build provider aliases, download/cache external ratings, ingest xG data, enrich canonical matches and then rebuild model-ready snapshots.

## 2. Core Decision

External data is handled as additive, cached enrichment:

```text
Level 1: Football-Data foundation
Level 2: ClubElo/external ratings
Level 3: xG/event enrichment
```

Rules:

- Level 1 remains sufficient for baseline operation.
- Level 2 and Level 3 improve the model only when coverage is available.
- Missing ClubElo/xG must not break the baseline.
- External provider data is cached under `data/external/`.
- Raw provider files under `data/raw/` are not mutated.
- Provider aliases live in an editable `team_registry.csv`.
- For modelling, post-match xG values may only be used through prior rolling features in snapshots.

## 3. Implemented Data Pipeline

### 3.1 Team registry

New script:

```bash
python scripts/build_team_registry.py \
  --matches data/processed/foundation_epl_multi_season/canonical_matches.csv \
            data/processed/foundation_laliga_multi_season/canonical_matches.csv \
            data/processed/foundation_seriea_multi_season/canonical_matches.csv \
            data/processed/foundation_bundesliga_multi_season/canonical_matches.csv \
            data/processed/foundation_ligue1_multi_season/canonical_matches.csv \
  --out-dir data/processed/entities \
  --dataset-name big5_team_registry
```

Outputs:

```text
data/processed/entities/team_registry.csv
data/processed/entities/team_registry_report.json
```

Purpose:

- stable canonical team identifiers,
- Football-Data aliases,
- ClubElo aliases,
- Understat aliases,
- StatsBomb alias placeholders,
- human-review fields.

The generated registry is intentionally editable. Provider joins should be reviewed if the report contains `generated_review_needed` rows.

### 3.2 ClubElo download

v0.49.7 update: the original v0.49.6 daily-snapshot download remains available, but it is no longer the recommended default for multi-season datasets. Use v0.49.7 team-history mode instead because it downloads one file per team alias rather than one file per match date.

New script:

```bash
python scripts/download_clubelo.py \
  --matches data/processed/foundation_epl_multi_season/canonical_matches.csv \
  --out-dir data/external/clubelo
```

Legacy v0.49.6 daily mode downloads one cached full daily ClubElo snapshot per match date. v0.49.7 default mode downloads one cached team history per club:

```text
data/external/clubelo/teams/clubelo_team_<alias>.csv
# legacy: data/external/clubelo/daily/clubelo_YYYY-MM-DD.csv
data/external/clubelo/clubelo_download_report.json
```

### 3.3 ClubElo enrichment

New script:

```bash
python scripts/enrich_matches_with_clubelo.py \
  --matches data/processed/foundation_epl_multi_season/canonical_matches.csv \
  --registry data/processed/entities/team_registry.csv \
  --clubelo-dir data/external/clubelo \
  --out-dir data/processed/enriched/epl_clubelo
```

Outputs:

```text
canonical_matches_with_clubelo.csv
clubelo_match_features.csv
clubelo_join_report.csv
clubelo_enrichment_report.json
```

New columns include:

```text
home_clubelo
away_clubelo
clubelo_diff
clubelo_available
home_external_elo
away_external_elo
```

### 3.4 Optional Understat xG research download

New optional script:

```bash
python scripts/download_understat_xg.py \
  --league-season EPL:2021 EPL:2022 EPL:2023 EPL:2024 EPL:2025 \
  --out-dir data/external/xg/understat
```

Outputs:

```text
understat_xg_matches.csv
understat_xg_download_report.json
```

This is an optional research connector. The user must confirm provider terms/licensing before relying on scraped data beyond local research.

### 3.5 xG enrichment from provider/manual CSV

New script:

```bash
python scripts/enrich_matches_with_xg.py \
  --matches data/processed/enriched/epl_clubelo/canonical_matches_with_clubelo.csv \
  --xg data/external/xg/understat/understat_xg_matches.csv \
  --registry data/processed/entities/team_registry.csv \
  --provider understat \
  --out-dir data/processed/enriched/epl_clubelo_xg
```

Outputs:

```text
canonical_matches_with_xg.csv
external_xg_matches_canonical.csv
xg_join_report.csv
xg_enrichment_report.json
```

New columns include:

```text
home_xg
away_xg
home_npxg
away_npxg
xg_provider
xg_available
```

## 4. Model-Ready Snapshot Expansion

`build_model_ready_dataset.py` now emits v0.49.6 enriched snapshots.

New feature groups:

```text
calendar:
- home_rest_days_pre
- away_rest_days_pre
- rest_days_diff_pre
- season_match_index_pre
- season_progress_pre

external strength:
- home_clubelo_pre
- away_clubelo_pre
- clubelo_diff_pre
- home_external_elo_pre
- away_external_elo_pre
- external_elo_diff_pre

derived rolling conversion:
- goal_conversion_last3/5/10
- sot_rate_last3/5/10
- sot_conversion_last3/5/10
- xg_per_shot_last3/5/10
- goals_minus_xg_last3/5/10
- defensive conversion allowed features
```

Targets now also include optional xG labels:

```text
target_home_xg
target_away_xg
target_home_npxg
target_away_npxg
```

Hard rule:

```text
Current-match xG is a target/observation. Only prior rolling xG features are model inputs.
```

## 5. Recommended Big 5 Data Workflow

For each league:

```text
foundation canonical_matches.csv
→ ClubElo enrichment
→ optional xG enrichment
→ model_ready_match_snapshots.csv
→ historical validation
```

For a global hybrid model, combine enriched canonical match files after team registry review, then build one model-ready snapshot file with league/context features visible.

## 6. What This Phase Does Not Do

This phase does not:

- guarantee external provider availability,
- embed paid API credentials,
- make xG mandatory,
- mutate `data/raw`,
- change model-selection logic,
- claim model metrics improve before local validation.

## 7. Validation

Focused validation completed:

```text
compileall: passed
pytest focused enrichment/snapshot/data-foundation tests: passed
```

The known artifact_tool spreadsheet warmup warning may appear in stderr, but the validation commands returned exit code 0.


## v0.49.8 Addendum — Understat direct scraping blocked

Direct Understat scraping is now treated as best-effort. If the page no longer exposes inline `datesData`, the downloader reports `blocked` but still writes an empty canonical `understat_xg_matches.csv` so downstream batch jobs can continue.

Preferred fallback:

```bash
python scripts/import_xg_csv.py   --input data/external/xg/provider_export.csv   --provider provider_csv   --out-dir data/external/xg/understat
```

Then run `scripts/enrich_matches_with_xg.py` normally.

For optional xG batch processing, use:

```bash
python scripts/enrich_matches_with_xg.py ... --allow-missing-xg
```
