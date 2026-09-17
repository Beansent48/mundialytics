# v0.37 Extra market stats + event line backtest

This release moves Mundialytics closer to bookmaker-style market modelling before paper betting.

## Added data ingestion

New scripts:

- `scripts/import_football_data_stats.py`
  - Imports Football-Data.co.uk CSVs.
  - Supports goals, shots, shots on target, corners, fouls, yellow cards and red cards.
  - Useful for historical club-league corners.

- `scripts/import_provider_fixture_stats.py`
  - Imports saved provider fixture-stat JSON, especially API-Football-style payloads.
  - Supports corners and goalkeeper saves when the provider returns them.

- `scripts/build_statsbomb_raw_extra_stats.py`
  - Parses StatsBomb raw event JSON.
  - Corners are counted from Pass type Corner.
  - Goalkeeper saves are counted from goalkeeper save events, not from SOT-goals proxies.

All scripts output a standard table:

`data/processed/team_match_market_stats.csv`

with fields such as:

- `corners_for`, `corners_against`
- `saves_for`, `saves_against`
- `shots_for`, `shots_against`
- `shots_on_target_for`, `shots_on_target_against`
- `yellow_cards_for`, `yellow_cards_against`
- `fouls_for`, `fouls_against`

## Added event line backtest

New script:

- `scripts/build_event_line_backtest.py`

It builds settled over/under signals for:

- match corners / team corners
- match shots / team shots
- match shots on target / team shots on target
- match yellow cards / team yellow cards
- match fouls / team fouls
- goalkeeper saves

Output:

`outputs/event_line_backtest_current/settled_event_line_signals.csv`

This file can be passed into:

```powershell
python scripts/backtest_pick_policy.py `
  --match-backtest outputs/evaluation_current/match_backtest_predictions.csv `
  --line-signals outputs/event_line_backtest_current/settled_event_line_signals.csv `
  --out-dir outputs/pick_policy_backtest_current `
  --min-picks 30 `
  --write-odds-template
```

## Conservative rules

- Corners are only available when a real corners target exists.
- Goalkeeper saves are only available when real saves or save events exist.
- No silent saves proxy from `shots_on_target_against - goals_against`.
- Every over/under market is evaluated by side: over and under separately.
- Pick-policy conclusions should be side-specific, not just market-level.

## Recommended flow

```powershell
python scripts/import_football_data_stats.py `
  --csv-dir data/raw/football_data `
  --out data/processed/team_match_market_stats.csv

python scripts/build_event_line_backtest.py `
  --team-match-stats data/processed/team_match_market_stats.csv `
  --out-dir outputs/event_line_backtest_current

python scripts/backtest_pick_policy.py `
  --match-backtest outputs/evaluation_current/match_backtest_predictions.csv `
  --line-signals outputs/event_line_backtest_current/settled_event_line_signals.csv `
  --out-dir outputs/pick_policy_backtest_current `
  --min-picks 30 `
  --write-odds-template
```
