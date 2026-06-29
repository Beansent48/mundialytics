# Next Chat Handoff — v0.50.3 FBref Team-Match Normalization Repair

## Current state

- Football-Data foundation is good for Big 5 results, shots/SOT, corners, fouls and cards.
- StatsBomb Open Data joins partially into Big 5 through aliases, but coverage is small.
- Raw FBref files are useful but previously normalized incorrectly.
- v0.50.3 repairs FBref one-row-per-team normalization for shooting, misc, keeper and schedule.
- Current FBref raw shooting files do not contain xG/npxG in the observed flattened columns, so xG still requires Understat/Kaggle or a richer FBref export.

## What to run locally after applying v0.50.3

Re-import raw FBref files:

```powershell
python scripts\import_advanced_csv.py --input data\external\advanced\fbref\raw\fbref_team_match_stats_shooting.csv --provider fbref_shooting --out-dir data\external\advanced\fbref\normalized\shooting
python scripts\import_advanced_csv.py --input data\external\advanced\fbref\raw\fbref_team_match_stats_misc.csv --provider fbref_misc --out-dir data\external\advanced\fbref\normalized\misc
python scripts\import_advanced_csv.py --input data\external\advanced\fbref\raw\fbref_team_match_stats_keeper.csv --provider fbref_keeper --out-dir data\external\advanced\fbref\normalized\keeper
python scripts\import_advanced_csv.py --input data\external\advanced\fbref\raw\fbref_team_match_stats_schedule.csv --provider fbref_schedule --out-dir data\external\advanced\fbref\normalized\schedule
```

Merge advanced sources:

```powershell
python scripts\merge_advanced_sources.py `
  --source `
    fbref_shooting=data\external\advanced\fbref\normalized\shooting\fbref_shooting_advanced_match_stats.csv `
    fbref_misc=data\external\advanced\fbref\normalized\misc\fbref_misc_advanced_match_stats.csv `
    fbref_keeper=data\external\advanced\fbref\normalized\keeper\fbref_keeper_advanced_match_stats.csv `
    fbref_schedule=data\external\advanced\fbref\normalized\schedule\fbref_schedule_advanced_match_stats.csv `
    statsbomb_open_data=data\external\advanced\statsbomb\statsbomb_advanced_match_stats.csv `
  --provider-priority fbref_shooting fbref_misc fbref_keeper fbref_schedule statsbomb_open_data provider_csv `
  --out-dir data\external\advanced\canonical
```

Re-enrich and rebuild model-ready snapshots:

```powershell
python scripts\enrich_matches_with_advanced_stats.py `
  --matches data\processed\enriched\foundation_big5_multi_season_clubelo\canonical_matches_with_clubelo.csv `
  --advanced data\external\advanced\canonical\canonical_advanced_match_stats.csv `
  --registry data\processed\entities\team_registry.csv `
  --manual-aliases data\processed\entities\provider_team_aliases_manual.csv `
  --provider-alias-column football_data_name `
  --out-dir data\processed\enriched\foundation_big5_multi_season_advanced `
  --dataset-name foundation_big5_multi_season_advanced

python scripts\build_model_ready_dataset.py `
  --matches data\processed\enriched\foundation_big5_multi_season_advanced\canonical_matches_with_advanced_stats.csv `
  --out-dir data\processed\model_ready\foundation_big5_multi_season_advanced `
  --dataset-name foundation_big5_multi_season_advanced
```

## Expected result

Expected improvement:

- FBref shots/SOT coverage should appear in normalized/canonical files.
- Keeper saves and save percentage should appear.
- Misc discipline/fouls/interceptions/tackles should appear.
- Enriched provider coverage should increase for Premier League, LaLiga and Ligue 1 if aliases are sufficient.
- xG coverage will not materially improve from this FBref raw export because the observed FBref raw shooting columns do not contain xG/npxG.

## Next recommended spec

`v0.50.4 — Understat/Kaggle xG + player data ingestion`

Goals:

- Import Understat/Kaggle match xG for Big 5.
- Import Understat player match stats if available.
- Generate canonical player match stats.
- Raise enriched xG coverage to at least 40%, ideally 70%+.
