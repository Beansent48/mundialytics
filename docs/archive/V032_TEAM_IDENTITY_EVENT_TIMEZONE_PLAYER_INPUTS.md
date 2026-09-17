# v0.32.1 — Team identity, event timezone, player inputs and status fix

This update hardens the operational matchday input layer before real odds integration.

## Main changes

- Adds stronger team-name canonicalization for World Cup/national-team provider names.
- Fixes cases such as `Ivory Coast`, `Côte d'Ivoire`, `CIV` -> `cote d ivoire`, so provider fixtures match the historical event data.
- Adds event-local kickoff columns in addition to user-local columns:
  - `kickoff_user`, `kickoff_user_date`, `kickoff_user_time`
  - `kickoff_event`, `kickoff_event_date`, `kickoff_event_time`
  - `event_timezone`, `event_timezone_source`
- Adds `--date-mode` to today-input generation:
  - `user`: only selected timezone date
  - `event`: only event/venue local date
  - `event_or_user`: include either; default
  - `utc`: UTC date
  - `any`: user, event, or UTC
- Adds best-effort fetching/parsing of current player inputs:
  - SofaScore event lineups through `/event/{id}/lineups`
  - SofaScore team player fallback through `/team/{id}/players`
  - ESPN summary/boxscore fallback through `/summary?event={id}`
  - ESPN team roster/team fallback through `/teams/{team_id}/roster` and `/teams/{team_id}`

## Important limitations

Free provider endpoints are undocumented and may change. Lineups usually become available only near kickoff. Squad/roster rows are fallback candidates, not confirmed starting elevens. The audit distinguishes lineups from squad fallback via `player_input_report` and row-level `source` / `lineup_status` / `status` fields.

## Recommended command

```powershell
python scripts/build_today_matchday_inputs.py `
  --today `
  --timezone Europe/Madrid `
  --date-mode event_or_user `
  --competition world_cup `
  --provider auto `
  --out-dir data/input/generated
```

Then run:

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/generated/today_fixtures.csv `
  --lineups data/input/generated/today_current_lineups.csv `
  --squads data/input/generated/today_squads.csv `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --model-config outputs/rolling_model_lab_current/best_rolling_model_config.json `
  --event-model-config outputs/player_prop_champion_full/prediction_registry.json `
  --out-dir outputs/statistical_matchday_today `
  --clean-out-dir `
  --no-demo-picks
```

## v0.32.1 status/timezone hardening

- ESPN status strings such as `STATUS_FULL_TIME`, `STATUS_SECOND_HALF` and `STATUS_SCHEDULED` are now mapped correctly to `completed`, `live` and `scheduled`.
- Completed games are excluded when `--include-completed` is not set, even when `--exclude-unknown-status` is not used.
- Added regression coverage for event-local date filtering with live/scheduled/completed ESPN rows.
