# Mundialytics v0.41 Odds Readiness and Coverage Audit

v0.41 adds a safety layer before paying for or wiring an odds API. It does **not** fetch odds, change model predictions, or place bets.

## Why this exists

v0.40 created the odds-ready contract. The next practical risk is not modelling; it is provider coverage and join quality:

- Does the provider actually offer the markets we need?
- Does it expose line + side + team/player subjects cleanly?
- How many model lines can be priced after mapping provider odds?
- Which markets are impossible or too sparse before we waste time integrating an API?

## New files

- `src/mundialytics/betting/odds_readiness.py`
- `scripts/check_odds_readiness.py`
- `scripts/audit_historical_odds_coverage.py`
- `tests/test_v041_odds_readiness.py`

## Readiness before provider/API selection

```powershell
python scripts/check_odds_readiness.py `
  --odds-template outputs/odds_ready_current/odds_needed_template.csv `
  --line-signals outputs/event_line_backtest_current_v0391/settled_event_line_signals.csv `
  --decision-matrix outputs/market_distribution_lab_current_v0391_clean/market_side_decision_matrix.csv `
  --out-dir outputs/odds_readiness_current
```

Outputs:

- `provider_market_requirements.csv`
- `odds_market_coverage_shopping_list.csv`
- `odds_readiness_report.json`

Use `provider_market_requirements.csv` when comparing The Odds API, OpticOdds, SportsGameOdds, Betfair Sportsbook/Exchange, Sportradar, or any paid historical file.

## After receiving mapped historical odds

```powershell
python scripts/audit_historical_odds_coverage.py `
  --model-lines outputs/odds_ready_current/model_market_lines.csv `
  --historical-odds data/processed/historical_odds_input.csv `
  --out-dir outputs/historical_odds_coverage_current
```

Only if coverage is acceptable should we run the EV/ROI calculation:

```powershell
python scripts/calculate_value_edges_from_odds.py `
  --model-lines outputs/odds_ready_current/model_market_lines.csv `
  --historical-odds data/processed/historical_odds_input.csv `
  --out-dir outputs/value_edges_current `
  --min-ev 0.03 `
  --min-edge 0.02
```

## Practical recommendation

Prioritize historical odds coverage for:

1. `team_yellow_cards`
2. `team_fouls`
3. `team_goalkeeper_saves` / `goalkeeper_saves`
4. `team_shots_on_target`
5. `yellow_cards`
6. `fouls`

Do not prioritize generic shots first: model distribution quality is weaker there, especially for team shots.
