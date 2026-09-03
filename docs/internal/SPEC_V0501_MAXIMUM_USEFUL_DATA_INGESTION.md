# Spec v0.50.1 — Maximum Useful Football Data Ingestion

## 1. Context

The v0.50.0 advanced data layer proved that the pipeline can ingest and merge advanced providers, but the first coverage audit showed that the model-ready snapshots had strong coverage for corners/fouls/cards and 0% coverage for xG/shots/SOT rolling features. That means the project should prioritize ingestion and data coverage before advanced modelling.

## 2. Objective

Maximize useful football data coverage for match, player, lineup and shot-event analysis, excluding market/odds work for this version.

## 3. Scope

Included:

- Preserve base Football-Data match stats when advanced provider values are missing.
- Merge advanced providers by column-level priority instead of selecting one whole provider row per match.
- Expand canonical match stats with:
  - neutral venue context
  - xG/npxG/xA and shot quality
  - shots inside/outside the box
  - headers / left foot / right foot shots and xG
  - open-play, penalty, set-piece, corner, free-kick and counterattack xG
  - possession, field tilt, PPDA, touches box, deep completions, final-third entries
  - progression, defensive, discipline and goalkeeper metrics
- Expand shot events with body part, distance and inside/outside-box flags.
- Add canonical lineups output for StatsBomb Open Data.
- Add generic player-match and shot-event CSV normalization.
- Add FBref optional downloads for player match stats, lineups and events.
- Add neutral venue home-advantage logic:
  - normal home fixture: home advantage applies
  - neutral fixture: no scheduled home advantage
  - neutral fixture in a listed team's country/city: advantage applies to that side
- Extend rolling snapshot creation to dynamically roll all numeric team metrics ending in `_for` / `_against`.
- Keep leakage policy explicit: current-match observations are targets/diagnostics, not model inputs.

Excluded:

- Market odds / implied probabilities / line movement (`K` from the previous metrics list).
- Live betting or production decisions.
- Full player Elo implementation.
- Paid provider integrations as mandatory dependencies.

## 4. Inputs

Preferred sources:

- Football-Data CSVs
- Kaggle/Understat match, player and shot CSVs
- FBref/soccerdata team and player match stats
- StatsBomb Open Data events and lineups
- Optional API provider outputs exported as CSV

## 5. Outputs

Primary outputs:

```text
data/external/advanced/*/*_advanced_match_stats.csv
data/external/advanced/*/*_player_match_stats.csv
data/external/advanced/*/*_shot_events.csv
data/external/advanced/*/*_lineups.csv
data/external/advanced/canonical/canonical_advanced_match_stats.csv
data/processed/enriched/*/canonical_matches_with_advanced_stats.csv
data/processed/model_ready/*/model_ready_match_snapshots.csv
```

Documentation outputs:

```text
docs/SPEC_V0501_MAXIMUM_USEFUL_DATA_INGESTION.md
docs/ADVANCED_METRICS_CATALOG.md
docs/PLAYER_LINEUP_STRENGTH_RATING_PLAN.md
docs/NEXT_CHAT_HANDOFF_V0501.md
```

## 6. Acceptance Criteria

- [x] Advanced enrichment preserves existing Football-Data shots/SOT/corners/fouls/cards when provider advanced values are null.
- [x] Advanced source merge is column-level priority, so Understat xG and Football-Data/FBref shots can coexist.
- [x] Canonical contract includes neutral venue/country/city context.
- [x] Canonical contract includes shots inside/outside box and shot body-part features.
- [x] StatsBomb importer writes match/player/shot/lineup outputs.
- [x] Model-ready snapshots dynamically include rolling features for expanded metrics.
- [x] Relevant tests pass for advanced layer and snapshot builders.
- [ ] User reruns ingestion locally and confirms coverage report improves for shots/SOT/xG.

## 7. Validation Performed

Commands run locally in this development environment:

```bash
python -m compileall -q src scripts
python -m pytest -q tests/test_v0495_hybrid_model_ready_snapshots.py tests/test_v0496_external_data_enrichment.py tests/test_v0500_advanced_football_data_layer.py
```

Result:

```text
11 passed
```

Known warning:

- Pandas PerformanceWarning from adding many rolling columns one-by-one. This does not change outputs, but can be optimized later.

## 8. Risks

- FBref scraping can fail due to provider blocking or schema changes.
- Kaggle/Understat datasets vary by uploader, so CSV aliases may need minor extension after seeing actual files.
- StatsBomb Open Data is high quality but partial coverage.
- Neutral venue advantage depends on venue/team country/city fields being present.
- Player ratings should not be implemented until lineups, minutes and player match stats have enough coverage.

## 9. Next Step

Run the ingestion again locally, then audit:

```powershell
python scripts\audit_advanced_data_coverage.py `
  --matches data\processed\enriched\foundation_big5_multi_season_advanced\canonical_matches_with_advanced_stats.csv `
  --out-dir data\processed\enriched\foundation_big5_multi_season_advanced `
  --dataset-name foundation_big5_multi_season_advanced
```
