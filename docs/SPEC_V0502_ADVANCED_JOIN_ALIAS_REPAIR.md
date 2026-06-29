# Spec v0.50.2 — Advanced Join Alias Repair

## Goal

Repair the advanced enrichment join so provider team names such as
`Paris Saint-Germain`, `Bayer Leverkusen`, `Olympique Lyonnais`, and
`Manchester United` can match Football-Data names such as `paris sg`,
`leverkusen`, `lyon`, and `man united`.

This version does not add model logic. It only improves ingestion and
coverage integrity.

## Problem observed

The v0.50.1 audit showed that Football-Data base stats were preserved after
regeneration, but provider-level advanced stats still did not attach to
foundation matches:

- `home_shots/home_sot` coverage recovered to ~99%.
- `provider` coverage stayed at 0%.
- `home_xg/home_npxg/progression/keeper` coverage stayed at 0%.
- A manual overlap test showed 84 joinable matches when aliases were applied.

## Scope

Included:

- Add manual alias support to `scripts/enrich_matches_with_advanced_stats.py`.
- Add core alias-aware join logic in `src/mundialytics/enrichment/advanced.py`.
- Apply aliases to both foundation team names and provider team names.
- Preserve existing Football-Data stats when provider values are null.
- Add a reusable alias seed file at `config/team_aliases/provider_team_aliases_manual.csv`.
- Add a regression test for `paris sg` vs `Paris Saint-Germain`.

Excluded:

- No Understat/Kaggle download.
- No model training.
- No market/odds features.
- No fuzzy matching in production yet. Fuzzy matching should stay diagnostic
  until we can audit false positives.

## New CLI option

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

## Acceptance criteria

- Shots/SOT coverage stays around 99%.
- `provider` coverage becomes greater than 0 when matching advanced rows exist.
- Known PSG/Ligue 1 and Leverkusen/Bundesliga examples join.
- Existing canonical match stats are not overwritten by null provider values.
- Model-ready rolling shots/SOT remain around 99%.

## Expected limitation

With currently available data, the first alias-aware join is expected to attach
only partial xG coverage. The observed test found 84 alias-matchable rows, mostly
from StatsBomb Open Data. Full Big 5 xG still requires a better FBref join and/or
Kaggle Understat import.
