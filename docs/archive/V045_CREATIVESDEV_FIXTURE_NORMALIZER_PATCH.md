# v0.45 — Creativesdev Fixture Normalizer Patch

This patch fixes the first real Creativesdev probe result from RapidAPI.

## Fixes

- `football-get-matches-by-date` kickoff is now parsed from `status.utcTime`.
- Final score is parsed from `status.scoreStr`.
- `status.reason.long` and `status.reason.short` are extracted into `status` and `status_short`.
- Empty lineups/stats/events outputs are written with valid CSV headers, avoiding `pandas.errors.EmptyDataError`.
- Fixture/date payloads no longer create fake empty event rows.
- Adds `scripts/creativesdev_build_fixtures_input.py` to convert normalized provider fixtures into Mundialytics fixture input.

## Recommended flow

```powershell
python scripts/creativesdev_probe.py `
  --config config/mundialytics_api_config.local.yaml `
  --endpoint-key fixtures_by_date `
  --vars date=20241107 `
  --max-api-calls 1

python scripts/creativesdev_build_fixtures_input.py `
  --fixtures outputs/creativesdev_probe_current/fixtures_by_date_fixtures_normalized.csv `
  --out data/input/generated/today_fixtures_from_creativesdev.csv `
  --include-finished
```

Use `--include-finished` only when building historical/training inputs. For today's upcoming fixtures, omit it.
