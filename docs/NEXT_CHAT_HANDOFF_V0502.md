# Next Chat Handoff — v0.50.2 Advanced Join Alias Repair

## Current project state

The Big 5 foundation dataset has strong Football-Data coverage:

- shots/SOT: ~99-100%
- corners/fouls/cards/reds: ~99-100%
- ClubElo enrichment exists
- neutral venue context exists

The advanced canonical file currently has provider rows from:

- `statsbomb_open_data`
- `fbref_shooting`

A manual join diagnostic found 84 matches when aliases were applied, confirming
that the join problem is primarily team naming.

## What changed in v0.50.2

- `enrich_matches_with_advanced_stats.py` now accepts `--manual-aliases`.
- `enrich_matches_with_advanced_stats()` now accepts `manual_aliases`.
- Alias maps are applied to both foundation matches and provider matches.
- Added `config/team_aliases/provider_team_aliases_manual.csv`.
- Added a regression test proving that `paris sg` joins `Paris Saint-Germain`
  while preserving base Football-Data shots/SOT.

## Validation run

```bash
python -m compileall -q src scripts
python -m pytest -q tests/test_v0500_advanced_football_data_layer.py
```

Result:

```text
4 passed
```

Warnings remain:

- pandas `PerformanceWarning` from column-by-column insertion in advanced import.
- a pandas `FutureWarning` from `combine_first` with empty values.

They do not invalidate this repair, but should be optimized later.

## Recommended next local commands

1. Copy or create aliases:

```powershell
New-Item -ItemType Directory -Force -Path data\processed\entities | Out-Null
Copy-Item config\team_aliases\provider_team_aliases_manual.csv `
  data\processed\entities\provider_team_aliases_manual.csv -Force
```

2. Rebuild advanced canonical with FBref shooting and StatsBomb:

```powershell
python scripts\merge_advanced_sources.py `
  --source `
    fbref_shooting=data\external\advanced\fbref\normalized\shooting\fbref_shooting_advanced_match_stats.csv `
    statsbomb_open_data=data\external\advanced\statsbomb\statsbomb_advanced_match_stats.csv `
  --provider-priority fbref_shooting statsbomb_open_data provider_csv `
  --out-dir data\external\advanced\canonical
```

3. Re-enrich with aliases:

```powershell
python scripts\enrich_matches_with_advanced_stats.py `
  --matches data\processed\enriched\foundation_big5_multi_season_clubelo\canonical_matches_with_clubelo.csv `
  --advanced data\external\advanced\canonical\canonical_advanced_match_stats.csv `
  --registry data\processed\entities\team_registry.csv `
  --manual-aliases data\processed\entities\provider_team_aliases_manual.csv `
  --provider-alias-column football_data_name `
  --out-dir data\processed\enriched\foundation_big5_multi_season_advanced `
  --dataset-name foundation_big5_multi_season_advanced
```

4. Audit and rebuild snapshots.

## Next recommended spec

`v0.50.3 — Understat/Kaggle + FBref Coverage Expansion`

Goals:

- Import Kaggle Understat match xG/player data.
- Diagnose why `fbref_shooting` provides rows but did not join in the manual test.
- Expand aliases based on unmatched examples.
- Add provider-level join coverage report by league/season/provider.
