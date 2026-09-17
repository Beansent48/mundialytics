# Mundialytics v0.40 Odds-Ready Contract

This layer prepares the betting engine for a future odds API without changing the statistical model, market distribution lab, or pick-policy logic.

## What v0.40 does

It creates a clean contract between Mundialytics model output and any future odds provider:

1. `model_market_lines.csv` — what the model believes.
2. `odds_needed_template.csv` — what odds we need from a bookmaker/odds API.
3. `historical_odds_input_schema.csv` — the universal CSV shape expected from a provider adapter.
4. `value_edges.csv` — model probabilities compared with bookmaker odds.

No odds API is called in v0.40. No bets are placed. No existing model performance logic is changed.

## Core formulas

```text
fair_odds = 1 / model_probability
edge = model_probability - (1 / bookmaker_odds)
ev = model_probability * bookmaker_odds - 1
```

`min_acceptable_odds` is stricter than fair odds because it includes a minimum EV and edge buffer:

```text
odds >= (1 + min_ev) / model_probability
odds >= 1 / (model_probability - min_edge)
```

The required odds is the stricter of both.

## Main command

```powershell
python scripts/build_odds_ready_shortlist.py `
  --line-signals outputs/event_line_backtest_current_v0391/settled_event_line_signals.csv `
  --decision-matrix outputs/market_distribution_lab_current_v0391_clean/market_side_decision_matrix.csv `
  --out-dir outputs/odds_ready_current `
  --decisions candidate `
  --min-model-probability 0.52 `
  --min-fair-odds 1.25 `
  --max-fair-odds 3.50 `
  --max-rows-per-signal-group 5000
```

The most important output is:

```text
outputs/odds_ready_current/odds_needed_template.csv
```

This is the file that a future API adapter must fill with bookmaker odds.

## Historical odds value check

When historical odds exist:

```powershell
python scripts/calculate_value_edges_from_odds.py `
  --model-lines outputs/odds_ready_current/model_market_lines.csv `
  --historical-odds data/processed/historical_odds_input.csv `
  --out-dir outputs/value_edges_current `
  --min-ev 0.03 `
  --min-edge 0.02
```

## Universal odds input columns

```text
snapshot_time_utc
bookmaker
provider
provider_event_id
internal_match_id
match_id
date
home_team
away_team
market_key
market
scope
subject_team
subject_player
line
side
bookmaker_odds
is_live
source_url
notes
```

Provider-specific adapters must map raw provider fields into this schema. The model should not care whether odds come from The Odds API, OpticOdds, SportsGameOdds, Betfair Exchange, a sportsbook export, or a paid historical file.

## What should not be done yet

Do not automate betting yet. First we need historical odds or paper tracking to measure:

- EV accuracy
- ROI
- yield
- closing line value
- market-specific profitability

