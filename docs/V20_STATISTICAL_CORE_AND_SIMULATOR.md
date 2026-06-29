# v0.20 Statistical Core & Tournament Simulator

This version adds a new manual-input-first package:

```text
src/mundialytics/statistical_core/
  match_model.py
  team_stats_model.py
  player_event_model.py
  tournament_simulator.py
  betting_value.py
  distributions.py
  calibration.py
  schemas.py
  reporting.py
```

## Purpose

The goal is to stop blocking progress on free fixture providers and build a modular statistical engine that can run from clean manual inputs:

- `fixtures.csv`
- `current_lineups.csv`
- `squads.csv`
- `odds.csv`
- optional `tournament_config.csv`
- processed historical event/player rows

## Main command

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/fixtures.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --odds data/input/odds.csv `
  --tournament-config data/input/tournament_config.csv `
  --historical-events data/sample/sample_player_events.csv `
  --out-dir outputs/statistical_matchday
```

Outputs:

```text
match_predictions.csv
scoreline_distribution.csv
team_stats_predictions.csv
player_event_predictions.csv
betting_edges.csv
recommended_picks.csv
tournament_simulation.csv
tournament_details.csv
daily_report.html
audit_report.json
```

## What works now

- Independent Poisson scoreline distributions with normalized finite support.
- 1X2, over/under goals, BTTS and most likely score.
- Team count expectations for shots, shots on target, fouls and yellow cards when historical columns exist.
- Corners are explicitly marked `not_available` when no real corner column exists. They are not invented.
- Player props are generated only from current lineups/squads, never by scanning historical players as inference candidates.
- Player props are tied to team expected events with player share, expected minutes, position prior and historical rate blend.
- Betting value output is paper mode only and includes model probability, no-vig implied probability where possible, edge, EV, recommendation flag, virtual stake, confidence, risk and reason.
- Tournament simulation is reproducible with a seed and produces group/qualification/champion probabilities plus approximate top scorer probabilities.
- HTML report shows match predictions, team stats, player props, edges, picks, tournament probabilities and warnings.

## Still experimental

- Match model is not yet a trained Dixon-Coles or bivariate Poisson model. It is a transparent profile-based independent Poisson model.
- Tournament knockouts are approximate unless explicit knockout fixtures are supplied.
- Top scorer probabilities are approximated from player shot expectations, not from a dedicated goals/player finishing model.
- Best player, goalkeeper and revelation awards are not yet implemented because v0.20 does not have a defensible award model.
- Odds mapping covers common markets and core props; exotic bookmaker naming still needs a market dictionary.

## Audit rules enforced

- No real betting execution; all staking is virtual paper mode.
- No future real minutes are required; `expected_minutes` is the operational column.
- Historical players do not become inference candidates unless they are present in `current_lineups.csv` or `squads.csv`.
- Explicit `status=retired/inactive/not_current` squad rows are dropped.
- Probability columns are auditable and outcome probabilities sum to one.
- Corners remain unavailable unless real historical corner columns exist.

## Tests

New tests live in:

```text
tests/test_v020_statistical_core.py
```

They cover:

- scoreline probability sums
- match model output probabilities
- team stats schema and no corner invention
- current-candidate-only player props
- dropped retired players
- betting EV calculation
- reproducible tournament simulation
- report generation
- full command output contract
