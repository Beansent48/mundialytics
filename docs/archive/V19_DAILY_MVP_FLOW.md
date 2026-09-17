# v0.19 Daily MVP Flow

This version moves the project from isolated modelling scripts toward an operational daily MVP.

## Source decision

Free fixture discovery cannot rely on API-Football for current World Cup fixtures because the free plan can block current seasons. The free MVP source stack is now:

1. **Primary fixtures:** SofaScore scheduled-events public endpoint.
   - Pros: keyless, simple daily schedule, stable event IDs in practice, broad coverage.
   - Cons: unofficial, may change, must cache raw JSON.
2. **Fallback fixtures:** ESPN public site scoreboard endpoint.
   - Pros: keyless, World Cup endpoint uses `soccer/fifa.world/scoreboard`, good fallback for tournament fixtures.
   - Cons: undocumented, less complete for lineups/player props, may change.
3. **Historical training:** StatsBomb Open Data.
   - Used for event/player/team statistics and calibration, not as the daily fixture source.

Sportmonks was evaluated as reliable but paid for the useful World Cup API tier, so it is not the free MVP source.

## Daily fixtures

```powershell
python scripts/fetch_today_fixtures.py `
  --competition world_cup `
  --today `
  --timezone America/New_York `
  --provider auto `
  --out outputs/free_world_cup_today_et.csv `
  --raw-out outputs/free_world_cup_today_et_raw.json
```

`--provider auto` tries SofaScore first, then ESPN if SofaScore errors or returns no rows. The output is post-filtered by the requested local date to avoid the yesterday/tomorrow timezone bug.

## Lineups

If SofaScore lineups are available:

```powershell
python scripts/fetch_fixture_lineups_free.py `
  --fixture-id <sofascore_fixture_id> `
  --fixtures outputs/free_world_cup_today_et.csv `
  --out outputs/lineups_<fixture_id>.csv `
  --raw-out outputs/lineups_<fixture_id>_raw.json
```

Lineups may be unavailable until near kickoff. When no provider lineup exists, use `data/templates/current_lineups_manual_template.csv` and keep the low-confidence warnings.

## Team/match stats layer

Build team-level historical stats from player events:

```powershell
python scripts/build_team_match_stats.py `
  --player-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --out outputs/team_match_stats.csv `
  --report-out outputs/team_match_stats_report.json

python scripts/validate_team_props.py `
  --team-match-stats outputs/team_match_stats.csv `
  --out outputs/team_props_validation_report.json `
  --strict
```

Predict current team props:

```powershell
python scripts/predict_team_props.py `
  --team-match-stats outputs/team_match_stats.csv `
  --fixtures outputs/free_world_cup_today_et.csv `
  --out outputs/team_props_today.csv
```

Corner markets are only produced when the training source contains real corner counts. The code deliberately does not invent corners from shots or passes.

## HTML product report

```powershell
python scripts/build_daily_report.py `
  --fixtures outputs/free_world_cup_today_et.csv `
  --match-predictions outputs/matchday_analysis/match_predictions.csv `
  --team-props outputs/team_props_today.csv `
  --player-props outputs/matchday_analysis/safe_lineup_props.csv `
  --picks outputs/matchday_analysis/match_value_picks.csv `
  --out outputs/daily_report.html
```

## What is still experimental

- SofaScore and ESPN endpoints are public/unofficial; cache raw JSON and keep fallbacks.
- Team props are an auditable recent-rate baseline. Next model upgrade should be Negative Binomial + hierarchical calibration.
- Corners depend on source coverage; do not offer corner picks unless `corners_for` exists and validation passes.
- Odds ingestion is still separate/manual unless a free odds source is added.
