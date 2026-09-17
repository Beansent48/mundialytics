# Operational Runtime Fixes

The first historical validation command was too broad for operational use: it tried to validate national-team football using essentially the whole international history from 1872 onward. That is not useful for a modern betting model and can run for hours.

## New default behaviour

For `international-results`, `run_historical_validation.py` now defaults to:

- modern football only: `start_date >= 2010-01-01`
- most recent completed matches only: `--max-completed-matches 3000`
- capped backtest predictions: `--max-backtest-predictions 1200`
- chunked walk-forward retraining every `--retrain-every` matches

You can override this with:

```bash
--full-history
```

but this is not recommended unless you are benchmarking offline.

## Recommended national-team validation

```powershell
python scripts/run_historical_validation.py `
  --source international-results `
  --input data/raw/international_results/results.csv `
  --out-dir outputs/validation_national_modern `
  --min-train-matches 1000 `
  --retrain-every 100 `
  --max-completed-matches 3000 `
  --max-backtest-predictions 1000 `
  --min-matches-ready 1500 `
  --min-backtest-predictions-ready 500
```

This validates both Poisson and Random Forest by default.

## Event/player-prop validation

After building event data from StatsBomb or Wyscout:

```powershell
python scripts/validate_player_props.py `
  --player-events data/processed/wyscout_player_events.csv `
  --lineups data/processed/wyscout_lineups.csv `
  --out-dir outputs/validation_player_props `
  --min-train-matches 500 `
  --test-matches 300
```

The script checks markets such as:

- player 1+ shots
- player 1+ shots on target
- player 1+ fouls committed
- player 1+ yellow card

This is the correct layer for validating props. `international-results` is only valid for the match-result engine: ELO, goals, 1X2, over/under, and tournament simulation.
