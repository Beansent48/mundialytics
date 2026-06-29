# v0.38 — Real saves, corners and relational event-line models

This version is focused on the user's current objective: improve model logic and evaluation before paper betting.

## What this version is supposed to solve

1. **Corners must be first-class markets**, not forgotten placeholders.
   - Team corners and match total corners are supported with over/under sides.
   - Football-Data.co.uk CSVs provide real historical corner targets for many club leagues.

2. **Goalkeeper saves must not rely only on a weak proxy.**
   - Team-level derived saves from Football-Data are still allowed only when explicitly requested and flagged.
   - Player-level goalkeeper saves are now supported from:
     - StatsBomb raw goalkeeper events.
     - API-Football fixture player stats JSONs when available.
   - StatsBomb Starting XI goalkeeper rows can create zero-save rows, reducing survivorship bias.

3. **Models should use football logic, not isolated averages.**
   - Corners expectation blends corner history with shot-volume pressure.
   - Saves expectation blends goalkeeper/player saves history, team save history and opponent shots-on-target pressure.

4. **Evaluation must match bookmaker markets.**
   - Every over/under market has both over and under sides.
   - BTTS has yes and no.
   - Result market has home/draw/away.
   - Event-line backtest outputs can be passed into `backtest_pick_policy.py` for side-level evaluation.

## Main new scripts

```powershell
python scripts/download_statsbomb_open_data_events.py --list-competitions
python scripts/download_statsbomb_open_data_events.py --competition-id 43 --season-id 106 --max-matches 50
python scripts/build_statsbomb_raw_goalkeeper_stats.py --event-json-dir data/raw/statsbomb/events --out data/processed/goalkeeper_match_stats.csv
python scripts/download_api_football_fixture_player_stats.py --league 39 --season 2024 --max-fixtures 50
python scripts/import_provider_goalkeeper_stats.py --json-dir data/raw/provider_goalkeeper_stats/api_football --out data/processed/goalkeeper_match_stats.csv
```

## Recommended pipeline

```powershell
# 1) Football-Data: real corners + shots/cards/fouls, optional derived team saves
python scripts/download_football_data_stats.py `
  --seasons 2526 2425 2324 2223 2122 `
  --mode zip `
  --out-dir data/raw/football_data

python scripts/import_football_data_stats.py `
  --csv-dir data/raw/football_data `
  --derive-saves-from-sot `
  --out data/processed/team_match_market_stats.csv

# 2) StatsBomb Open Data: real raw event goalkeeper saves when coverage exists
python scripts/download_statsbomb_open_data_events.py --list-competitions
python scripts/download_statsbomb_open_data_events.py `
  --competition-id <ID> `
  --season-id <ID> `
  --max-matches 50 `
  --out-dir data/raw/statsbomb/events

python scripts/build_statsbomb_raw_extra_stats.py `
  --event-json-dir data/raw/statsbomb/events `
  --out data/processed/statsbomb_raw_extra_match_stats.csv

python scripts/build_statsbomb_raw_goalkeeper_stats.py `
  --event-json-dir data/raw/statsbomb/events `
  --out data/processed/goalkeeper_match_stats.csv

# 3) Build line signals
python scripts/build_event_line_backtest.py `
  --team-match-stats data/processed/team_match_market_stats.csv `
  --goalkeeper-match-stats data/processed/goalkeeper_match_stats.csv `
  --out-dir outputs/event_line_backtest_current

# 4) Evaluate all markets/sides with the pick policy backtest
python scripts/backtest_pick_policy.py `
  --match-backtest outputs/evaluation_current/match_backtest_predictions.csv `
  --line-signals outputs/event_line_backtest_current/settled_event_line_signals.csv `
  --out-dir outputs/pick_policy_backtest_current `
  --min-picks 30 `
  --write-odds-template
```

## Hard rules

- No unflagged saves proxy.
- No corners proxy when real corners are missing.
- Derived saves must be analysed separately from real goalkeeper saves.
- Do not treat team-level goalkeeper saves as player goalkeeper saves.
- Do not select a market only because it wins globally; inspect side-level calibration and sample size.
