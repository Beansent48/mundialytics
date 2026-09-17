# Live operation checklist

This engine is ready to run in **paper mode** against upcoming fixtures once you provide current fixtures and odds.

## 1. Choose one scope

Do not mix:

- `national` for selections;
- `club` for clubs.

Train separate models.

## 2. Build data

National teams:

```bash
python scripts/download_data_sources.py international-results --out data/raw/international_results/results.csv
python scripts/build_dataset.py --source international-results --input data/raw/international_results/results.csv --out data/processed/national_matches.csv
```

Clubs:

```bash
python scripts/download_data_sources.py football-data-uk --url https://www.football-data.co.uk/mmz4281/2526/E0.csv --out data/raw/football_data_uk/2526_E0.csv
python scripts/build_dataset.py --source football-data-uk --input data/raw/football_data_uk/2526_E0.csv --out data/processed/epl_2526.csv
```

If Python cannot download because of DNS/internet restrictions, download manually in the browser and place the CSV in the same path.

## 3. Diagnose and train

```bash
python scripts/diagnose_dataset.py --matches data/processed/national_matches.csv --out outputs/national_diagnostic.json
python scripts/train_from_csv.py --matches data/processed/national_matches.csv --model-out models/national_goal_model.pkl --model-type poisson --data-source international-results
```

## 4. Prepare upcoming fixtures

Create a CSV with:

```text
fixture_id,date,home_team,away_team,neutral,competition,season,stage,team_scope,source
```

For World Cup/neutral matches, set `neutral=1` and `team_scope=national`.

## 5. Predict fixtures

```bash
python scripts/predict_fixtures.py --bundle models/national_goal_model.pkl --fixtures data/processed/upcoming_national_fixtures.csv --out outputs/national_predictions.csv
```

## 6. Add odds

Create an odds CSV:

```text
fixture_id,bookmaker,market_type,selection,odds
```

For 1X2 use `market_type=match_winner` and `selection=home`, `draw`, `away` or exact team names.

```bash
python scripts/value_from_predictions.py --predictions outputs/national_predictions.csv --odds data/processed/current_match_odds.csv --out outputs/today_value_picks.csv
```

## 7. Track only in paper mode

```bash
python scripts/paper_track.py append --picks outputs/today_value_picks.csv --ledger outputs/paper_ledger.csv --created-at 2026-06-15T20:00:00Z --stake 1
```

## 8. Run safety check

Use the dataset diagnostic, backtest summary and quality gate before trusting any value flags.

If the quality gate says `NOT_READY_KEEP_IN_DEVELOPMENT_OR_PAPER_ONLY`, do not use real-money staking.
