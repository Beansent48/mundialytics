# v0.47 — Prediction & Simulation Consolidation

## Status

Accepted.

## Date

2026-06-24

## Context

Mundialytics Betting Engine should continue as a football prediction, statistical simulation, dynamic-market-lines and paper-value engine.

The project must not be converted into a Betfair Exchange trading bot. Exchange-specific execution, back/lay logic and liquidity workflows are not part of this version.

The v0.46.x work explored OddsPapi/RapidAPI for Bet365 odds. Current and historical odds by known fixture ID can be useful, but bulk historical fixture discovery was not reliable enough to make it a core dependency.

## Decision

v0.47 consolidates the existing prediction and simulation engine without changing model strategy or project architecture.

The statistical core remains the first-class workflow:

```text
fixtures/current players/historical events
-> match predictions
-> scoreline distribution
-> team/player event projections
-> dynamic market lines
-> tournament simulation
-> paper-mode value layer if odds are provided
-> HTML/CSV/audit outputs
```

Odds providers remain optional adapters. OddsPapi historical bulk backfill is treated as experimental and should not block the prediction/simulation flow.

## Goals

- Keep Mundialytics focused on prediction, simulation, advanced football statistics and paper value analysis.
- Preserve the current project structure and existing behavior.
- Make the main statistical matchday runner the recommended smoke path.
- Keep odds comparison optional.
- Make missing or unreliable markets explicit with `not_available` or audit warnings.
- Keep player props restricted to current lineups/squads, never historical players alone.
- Provide a clean baseline ZIP before adding new v0.47+ features.

## Non-Goals

- No Betfair Exchange conversion.
- No live betting.
- No automated staking.
- No new paid odds provider.
- No model training/evaluation redesign.
- No folder restructure.
- No historical OddsPapi bulk backfill retry campaign.
- No invented player, corner, goalkeeper-save or prop data.

## Main Command

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/fixtures.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --odds data/input/odds.csv `
  --tournament-config data/input/tournament_config.csv `
  --historical-events data/sample/sample_player_events.csv `
  --out-dir outputs/statistical_matchday_v047 `
  --n-simulations 1000 `
  --seed 42 `
  --clean-out-dir
```

The same command can run without `--odds`; value/edge outputs should then remain empty or marked unavailable while predictions and simulation still run.

## Expected Outputs

```text
match_predictions.csv
scoreline_distribution.csv
team_stats_predictions.csv
player_event_predictions.csv
betting_edges.csv
recommended_picks.csv
tournament_simulation.csv
tournament_details.csv
top_scorer_predictions.csv
award_predictions.csv
competition_summary.csv
dynamic_market_lines.csv
audit_report.json
daily_report.html
```

## Acceptance Criteria

- [x] Package version is aligned to `0.47.0`.
- [x] The main statistical matchday command still runs with sample inputs.
- [x] The audit report identifies paper mode and missing/unreliable markets.
- [x] The flow does not require live API calls.
- [x] OddsPapi historical bulk is documented as experimental, not required.
- [x] The README points users to the prediction/simulation-first flow.
- [x] A focused smoke test exists for the v0.47 matchday runner.

## Validation

Focused validation command:

```bash
pytest tests/test_v047_statistical_matchday_smoke.py
```

Manual smoke command:

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/fixtures.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --odds data/input/odds.csv `
  --tournament-config data/input/tournament_config.csv `
  --historical-events data/sample/sample_player_events.csv `
  --out-dir outputs/statistical_matchday_v047 `
  --n-simulations 100 `
  --seed 42 `
  --clean-out-dir
```

## Known Limitations

- OddsPapi historical fixture discovery by date is not considered reliable enough for bulk ROI/CLV backfill.
- Current odds and historical odds by known fixture ID remain useful but optional.
- Tournament knockout simulation remains approximate unless explicit knockout fixtures are supplied.
- Top scorer/awards are approximate and depend on available player/event inputs.
- ROI, yield and CLV require real odds snapshots or a long paper ledger; statistical signal alone is not proof of profitability.
