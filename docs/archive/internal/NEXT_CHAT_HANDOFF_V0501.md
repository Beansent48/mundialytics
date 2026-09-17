# Next Chat Handoff — v0.50.1 Maximum Useful Data Ingestion

## What changed

Implemented ingestion-first improvements:

- Expanded canonical advanced match schema.
- Added neutral venue/country/city home-advantage logic.
- Added shots inside/outside box.
- Added header / left-foot / right-foot shot and xG metrics.
- Added set-piece, corner, free-kick, counterattack xG fields.
- Added possession/territory/progression/defensive/keeper fields.
- Added generic player-match CSV import.
- Added generic shot-event CSV import.
- Added StatsBomb lineup output.
- Added optional FBref player stats, lineups and events downloads.
- Fixed advanced enrichment so provider nulls do not wipe existing Football-Data shots/SOT/corners/fouls/cards.
- Changed advanced merge to column-level priority.

## Files changed

```text
src/mundialytics/enrichment/advanced.py
src/mundialytics/data/loaders.py
src/mundialytics/features/team_features.py
src/mundialytics/data_quality/model_ready_snapshots.py
scripts/import_statsbomb_open_advanced.py
scripts/import_advanced_csv.py
scripts/import_kaggle_understat.py
scripts/download_fbref_advanced.py
docs/SPEC_V0501_MAXIMUM_USEFUL_DATA_INGESTION.md
docs/ADVANCED_METRICS_CATALOG.md
docs/PLAYER_LINEUP_STRENGTH_RATING_PLAN.md
docs/NEXT_CHAT_HANDOFF_V0501.md
```

## Validation run

```bash
python -m compileall -q src scripts
python -m pytest -q tests/test_v0495_hybrid_model_ready_snapshots.py tests/test_v0496_external_data_enrichment.py tests/test_v0500_advanced_football_data_layer.py
```

Result:

```text
11 passed
```

## Known warnings

Pandas PerformanceWarning appears because rolling features are generated column-by-column. This is not a correctness failure, but later optimization should batch-created rolling columns.

## What to run next locally

Re-run ingestion from the project root, then audit:

```powershell
python scripts\audit_advanced_data_coverage.py `
  --matches data\processed\enriched\foundation_big5_multi_season_advanced\canonical_matches_with_advanced_stats.csv `
  --out-dir data\processed\enriched\foundation_big5_multi_season_advanced `
  --dataset-name foundation_big5_multi_season_advanced
```

Then rebuild snapshots:

```powershell
python scripts\build_model_ready_dataset.py `
  --matches data\processed\enriched\foundation_big5_multi_season_advanced\canonical_matches_with_advanced_stats.csv `
  --out-dir data\processed\model_ready\foundation_big5_multi_season_advanced `
  --dataset-name foundation_big5_multi_season_advanced
```

## Main thing to check

The previous audit showed:

```text
xG/shots/SOT rolling coverage = 0%
corners/fouls/cards rolling coverage ~99%
```

After this patch, shots/SOT should be preserved from Football-Data and rolling coverage should improve. xG depends on Understat/FBref/StatsBomb matching actual Big 5 fixtures.

## Do not do yet

- Do not train an advanced supervised model until coverage improves.
- Do not implement player Elo until lineups + player stats coverage are audited.
- Do not add market/odds features in this data-ingestion version.
