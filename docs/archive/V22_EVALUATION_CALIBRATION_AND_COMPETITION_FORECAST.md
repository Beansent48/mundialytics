# v0.22 Evaluation, Calibration & Competition Forecast

## Goal

v0.22 turns the v0.21 statistical core from a working prediction pipeline into a more auditable modelling product. The focus is not more markets; it is measuring whether model probabilities are useful.

## Main additions

### 1. Temporal evaluation and calibration

New module:

```text
src/mundialytics/statistical_core/evaluation.py
```

New script:

```text
scripts/evaluate_statistical_core.py
```

It builds historical match results from the processed event file, performs a chronological train/test split, predicts holdout matches, and reports:

- 1X2 log loss
- 1X2 Brier score
- max-probability pick accuracy
- Over 2.5 log loss/Brier
- BTTS log loss/Brier
- calibration bins
- a simple shrinkage calibration model

The calibration model can be applied to a live matchday run with:

```powershell
--calibration-model outputs/evaluation/match_calibration_model.json
```

This is deliberately conservative: it shrinks overconfident probabilities toward empirical base rates when the holdout suggests that helps.

### 2. Clean output directory

`run_statistical_matchday.py` now supports:

```powershell
--clean-out-dir
```

This deletes the output directory before writing new outputs, avoiding accumulated stale files.

### 3. Demo odds pick blocker

`run_statistical_matchday.py` now supports:

```powershell
--no-demo-picks
```

If odds use `bookmaker=demo_book`, the engine still writes `betting_edges.csv`, but blocks recommended picks and sets virtual stake to zero. This prevents demo odds from looking like real value.

### 4. Football.meets.data-style competition outputs

New module:

```text
src/mundialytics/statistical_core/scorer_model.py
```

New script:

```text
scripts/build_competition_forecast.py
```

New outputs from the main matchday command:

```text
top_scorer_predictions.csv
award_predictions.csv
competition_summary.csv
```

These include:

- team power rankings from tournament simulation
- match goal-environment summaries
- approximate top scorer probabilities
- approximate Golden Boot ranking
- approximate best-player attacking-impact ranking
- explicit `not_available` rows for goalkeeper/clean-sheet and breakout awards when required data is missing

## Honest limitations

- The match model is still independent Poisson, not Dixon-Coles/Bivariate Poisson.
- Backtesting currently uses a single chronological holdout, not full rolling-origin validation.
- Some historical event files may not contain true home/away venue information, so evaluation uses deterministic neutral team A/team B labels.
- Top scorer is estimated from player shot volume and position-level shot conversion assumptions, not a dedicated player xG model.
- Best-player ranking is an attacking-impact proxy, not an official award model.
- Goalkeeper and breakout predictions are explicitly unavailable unless goalkeeper clean-sheet data and age/minutes priors are added.

## Recommended commands

Evaluate and create calibration:

```powershell
python scripts/evaluate_statistical_core.py `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --out-dir outputs/evaluation_current `
  --clean-out-dir `
  --min-train-matches 50 `
  --test-fraction 0.25
```

Run a clean calibrated matchday:

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/fixtures.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --odds data/input/odds.csv `
  --tournament-config data/input/tournament_config.csv `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --calibration-model outputs/evaluation_current/match_calibration_model.json `
  --out-dir outputs/statistical_matchday_current `
  --clean-out-dir `
  --no-demo-picks
```

Build competition forecasts from existing outputs:

```powershell
python scripts/build_competition_forecast.py `
  --player-events outputs/statistical_matchday_current/player_event_predictions.csv `
  --tournament-simulation outputs/statistical_matchday_current/tournament_simulation.csv `
  --match-predictions outputs/statistical_matchday_current/match_predictions.csv `
  --out-dir outputs/competition_forecast_current
```
