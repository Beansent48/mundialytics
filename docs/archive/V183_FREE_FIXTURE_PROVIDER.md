# v0.18.3 Free fixture provider

## Problem

API-Football's free plan can block current seasons, including the current World
Cup season, with a provider error. That makes it unsuitable as the only free
source for the daily match slate.

## Decision

Use SofaScore public scheduled-events for free fixture discovery:

```text
https://api.sofascore.com/api/v1/sport/football/scheduled-events/{YYYY-MM-DD}
```

The endpoint is public and keyless but unofficial. It must be used defensively:
cache raw responses, keep request volume small, and expect schema changes.

## Commands

```powershell
python scripts/fetch_world_cup_fixtures_free.py `
  --today `
  --timezone America/New_York `
  --out outputs/sofascore_world_cup_today_et.csv `
  --raw-out outputs/sofascore_world_cup_today_et.json
```

For exact date:

```powershell
python scripts/fetch_world_cup_fixtures_free.py `
  --date 2026-06-17 `
  --timezone America/New_York
```

## Output contract

The CSV includes provider-agnostic IDs:

- `provider = sofascore`
- `provider_match_id`
- `fixture_id`
- `match_id = sofascore:<event_id>`
- kickoff timestamps and local kickoff fields
- competition/team names
- provider team IDs where SofaScore supplies them

## Follow-up

Lineups still need a source. SofaScore lineups may be available close to kickoff
via `/event/{event_id}/lineups`; API-Football lineups remain an optional source
when plan access allows it. For player identity, this means fixtures can come
from SofaScore while current lineups may come from SofaScore, API-Football, or a
manual CSV with provider IDs when available.
