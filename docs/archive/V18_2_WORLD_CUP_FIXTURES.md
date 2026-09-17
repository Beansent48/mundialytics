# v0.18.2 — World Cup fixtures command

This release adds a dedicated command to fetch **only FIFA World Cup 2026 fixtures** from API-Football.

## Why

`fetch_today_fixtures.py` returns all football fixtures for the requested date/timezone. For Mundialytics matchday work, the first operational step should usually be scoped to World Cup only. A placeholder fixture id is not acceptable: fixture ids must come from the provider response.

API-Football's 2026 World Cup guide uses:

- `league=1`
- `season=2026`

## Command: today's World Cup fixtures in US Eastern time

```powershell
$env:API_FOOTBALL_KEY="TU_KEY"

python scripts/fetch_world_cup_fixtures.py `
  --today `
  --timezone America/New_York `
  --out outputs/api_football_world_cup_today_et.csv `
  --raw-out outputs/api_football_world_cup_today_et.json
```

The CSV includes:

- `fixture_id`
- `provider_match_id`
- `match_id`
- `date`
- `kickoff_local_date`
- `kickoff_local_time`
- `competition`
- `league_id`
- `season`
- `round`
- `home_team`
- `away_team`
- `status_short`
- `status_long`

## Exact date

If timezone boundaries are confusing, pass the exact US date:

```powershell
python scripts/fetch_world_cup_fixtures.py `
  --date 2026-06-17 `
  --timezone America/New_York `
  --out outputs/api_football_world_cup_2026-06-17_et.csv `
  --raw-out outputs/api_football_world_cup_2026-06-17_et.json
```

## Generic equivalent

```powershell
python scripts/fetch_api_football_fixtures.py `
  --world-cup `
  --date 2026-06-17 `
  --timezone America/New_York `
  --print-table `
  --out outputs/api_football_world_cup_2026-06-17_et.csv
```

Internally `--world-cup` maps to `--league 1 --season 2026`.

## Timezone safety

The command now post-filters by kickoff calendar date in the requested timezone. This prevents accidental inclusion of fixtures from yesterday/tomorrow when the API payload is interpreted in UTC or the user's local machine timezone.

The report contains:

- `rows_before_post_filter`
- `rows`
- `local_date_filter_warning`

If rows are removed by the local-date filter, the warning will say how many rows were dropped.
