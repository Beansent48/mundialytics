# Mundialytics data download commands

## Corners: Football-Data.co.uk

Download season ZIPs:

```powershell
python scripts/download_football_data_stats.py `
  --seasons 2526 2425 2324 2223 2122 `
  --mode zip `
  --out-dir data/raw/football_data
```

Then import them:

```powershell
python scripts/import_football_data_stats.py `
  --csv-dir data/raw/football_data `
  --out data/processed/team_match_market_stats.csv
```

Then build settled over/under line signals:

```powershell
python scripts/build_event_line_backtest.py `
  --team-match-stats data/processed/team_match_market_stats.csv `
  --out-dir outputs/event_line_backtest_current
```

Then include those lines in pick-policy evaluation:

```powershell
python scripts/backtest_pick_policy.py `
  --match-backtest outputs/evaluation_current/match_backtest_predictions.csv `
  --line-signals outputs/event_line_backtest_current/settled_event_line_signals.csv `
  --out-dir outputs/pick_policy_backtest_current `
  --min-picks 30 `
  --write-odds-template
```

## Goalkeeper saves: API-Football/API-Sports

Requires an API key:

```powershell
$env:API_FOOTBALL_KEY="YOUR_KEY"

python scripts/download_api_football_fixture_stats.py `
  --league 39 `
  --season 2024 `
  --max-fixtures 50 `
  --out-dir data/raw/provider_fixture_stats/api_football
```

Import provider stats:

```powershell
python scripts/import_provider_fixture_stats.py `
  --json-dir data/raw/provider_fixture_stats/api_football `
  --out data/processed/provider_team_match_market_stats.csv
```

Combine with Football-Data stats if needed:

```powershell
python scripts/combine_team_match_market_stats.py `
  --csv data/processed/team_match_market_stats.csv `
  --csv data/processed/provider_team_match_market_stats.csv `
  --out data/processed/team_match_market_stats_combined.csv
```
