# v0.46.4 — Bet365-only backfill with safe long windows

## Decision

Mundialytics now treats `bet365` as the only bookmaker for historical odds backfill by default.

Do not mix bookmakers in the first ROI/CLV training pass. Use one market source first, then add executable-market layers later if needed.

## Fixture discovery windows

OddsPapi fixture discovery with `sportId` + `from` + `to` must stay under 10 days according to OddsPapi documentation. This version therefore uses large but safe windows:

- `--chunk-hours 216` = 9 days raw chunks
- `--pad-hours 6` = 6h before and after each chunk
- total API span ~= 9.5 days
- `--api-max-window-hours 239` = hard cap below 10 days
- default range clamp = first/last API windows stay inside the prepared internal match date range

If a bigger chunk is requested, the planner caps it automatically and records `chunk_was_capped=true` in `fixture_request_windows.csv` and `fixture_plan_summary.json`. If you ever want padding outside the internal date range for boundary matching, pass `--no-clamp-to-internal-range`; the default is stricter sync.

## Date sync policy

The plan is built from the internal historical dataset after applying `--min-date`.

The script writes:

- `internal_matches_prepared.csv`
- `fixture_request_windows.csv`
- `fixture_plan_summary.json`
- `input_column_diagnostics.json`

Check `fixture_plan_summary.json` before downloading fixtures. It includes:

- `internal_min_kickoff_utc`
- `internal_max_kickoff_utc`
- `first_window_from`
- `last_window_to`
- `max_window_span_hours`
- `total_expected_match_window_hits`

## Recommended commands

```powershell
python scripts\oddspapi_build_historical_fixture_plan.py `
  --matches outputs\event_line_backtest_current_v0391\settled_event_line_signals.csv `
  --out-dir outputs\oddspapi_historical_fixture_plan_bet365 `
  --min-date 2026-01-01 `
  --chunk-hours 216 `
  --pad-hours 6 `
  --api-max-window-hours 239
```

Then dry-run fixture calls:

```powershell
python scripts\oddspapi_fetch_historical_fixtures.py `
  --windows outputs\oddspapi_historical_fixture_plan_bet365\fixture_request_windows.csv `
  --out-dir outputs\oddspapi_historical_fixtures_bet365_preview `
  --provider-config config\mundialytics_api_config.local.yaml `
  --mode rapidapi `
  --bookmakers bet365 `
  --max-api-calls 999 `
  --monthly-budget 250 `
  --dry-run
```

If the plan is correct, fetch fixtures:

```powershell
python scripts\oddspapi_fetch_historical_fixtures.py `
  --windows outputs\oddspapi_historical_fixture_plan_bet365\fixture_request_windows.csv `
  --out-dir outputs\oddspapi_historical_fixtures_bet365 `
  --provider-config config\mundialytics_api_config.local.yaml `
  --mode rapidapi `
  --bookmakers bet365 `
  --max-api-calls 999 `
  --monthly-budget 250
```

