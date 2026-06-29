# v0.46.3 — Internal Match Date Parsing Patch

This patch fixes historical OddsPapi fixture-plan creation when the internal historical match file stores dates in a different format than expected.

## Fixed

- `normalize_internal_matches` no longer calls `.dt` on non-datetimelike columns.
- Accepts date-only strings, ISO datetime strings, `yyyymmdd` integers/strings, Unix seconds and Unix milliseconds.
- Falls back from empty `kickoff_utc` to `date`.
- Writes `input_column_diagnostics.json` from `oddspapi_build_historical_fixture_plan.py` to make wrong input files easier to debug.
- Deduplicates market-line files to one row per `match_id` before building request windows.

## Why this matters

Large backtest files such as `settled_event_line_signals.csv` can contain many rows per match and dates stored as strings or integers. The historical odds backfill must normalize those safely before querying OddsPapi fixtures.

## Command

```powershell
python scripts\oddspapi_build_historical_fixture_plan.py `
  --matches outputs\event_line_backtest_current_v0391\settled_event_line_signals.csv `
  --out-dir outputs\oddspapi_historical_fixture_plan_current `
  --min-date 2026-01-01 `
  --chunk-hours 24 `
  --pad-hours 4 `
  --max-windows 7
```

If it fails again, inspect:

```text
outputs/oddspapi_historical_fixture_plan_current/input_column_diagnostics.json
```
