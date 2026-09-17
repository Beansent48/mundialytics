# v0.18.1 — Today Fixtures Command

This update fixes an operational gap in v0.18: examples should not require a manually invented `fixture_id`.

## Goal

Fetch today's available API-Football fixtures in a US timezone, print the real `fixture_id` values, and save the fixtures to CSV/JSON for the next steps.

## Main command

```powershell
$env:API_FOOTBALL_KEY="YOUR_KEY"

python scripts/fetch_today_fixtures.py `
  --timezone America/New_York `
  --out outputs/api_football_today_fixtures_et.csv `
  --raw-out outputs/api_football_today_fixtures_et.json
```

The command defaults to `--today`, `--timezone America/New_York`, and `--print-table` if those are not passed.

## Generic command

```powershell
python scripts/fetch_api_football_fixtures.py `
  --today `
  --timezone America/New_York `
  --out outputs/api_football_today_fixtures_et.csv `
  --raw-out outputs/api_football_today_fixtures_et.json `
  --print-table
```

## Useful filters

For a specific known API-Football league/season:

```powershell
python scripts/fetch_today_fixtures.py `
  --league 1 `
  --season 2026 `
  --timezone America/New_York `
  --out outputs/api_football_today_worldcup_et.csv
```

For a specific date:

```powershell
python scripts/fetch_api_football_fixtures.py `
  --date 2026-06-26 `
  --timezone America/New_York `
  --out outputs/api_football_2026_06_26_et.csv `
  --print-table
```

## Output columns

The CSV includes at least:

- `fixture_id`
- `provider_match_id`
- `match_id`
- `date`
- `fixture_timezone`
- `competition`
- `home_team`
- `away_team`
- `status_short`
- `status_long`
- taxonomy columns such as `team_type`, `competition_context`, and `gender` when inferable.

## Next step

Use a real `fixture_id` from the table:

```powershell
python scripts/fetch_api_football_lineups.py `
  --fixture-id <REAL_FIXTURE_ID_FROM_TABLE> `
  --out outputs/api_football_current_lineups.csv `
  --raw-out outputs/api_football_current_lineups_raw.json
```

## Notes

- The timezone must be an IANA timezone string accepted by API-Football, e.g. `America/New_York`, `America/Los_Angeles`, `America/Chicago`.
- On Windows, local timezone calculation may require `tzdata`; if unavailable, pass `--date YYYY-MM-DD` explicitly.
- API-Football free-plan coverage can vary by league/competition. Empty results may mean no accessible fixtures for that date under the current subscription.
