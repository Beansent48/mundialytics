# v0.48 — Statistical Simulator Upgrade

## Status

Accepted.

## Date

2026-06-24

## Context

Mundialytics Betting Engine should now prioritise the statistical simulator before expanding the betting-picks layer.

The core product direction is:

```text
football data
-> probabilistic match engine
-> scoreline and event distributions
-> Monte Carlo match/tournament simulation
-> statistical reports
-> optional fair odds / paper-value layer later
```

Betting picks remain useful, but they should be derived from calibrated probabilities and fair odds. They must not drive the simulator design.

## Decision

v0.48.0 establishes a professional simulator contract without rewriting model strategy or project structure.

This version:

- makes the statistical simulator the explicit first-class workflow;
- documents the expected simulator outputs and schemas;
- adds a machine-readable simulator contract report;
- keeps odds optional and experimental;
- keeps player inference gated by current lineups/squads;
- optimises tournament score sampling so larger Monte Carlo runs are practical;
- documents 50,000 simulations as the recommended serious tournament-report run size;
- keeps small simulation counts for tests/smoke checks.

## Goals

- Consolidate a reliable simulator-first foundation before adding new betting features.
- Define expected match, scoreline, team-stat, player-stat, dynamic-line and tournament outputs.
- Make output schema changes visible through `simulation_contract_report.json`.
- Support large Monte Carlo tournament simulations such as `--n-simulations 50000`.
- Improve auditability of simulation settings: seed, simulation count and retained detail rows.
- Preserve existing v0.47 behaviour where possible.

## Non-Goals

- No live betting.
- No Betfair Exchange conversion.
- No automated staking.
- No new odds provider.
- No paid API dependency.
- No model-training redesign.
- No claim that 50,000 simulations improve model accuracy by themselves.
- No invented corners, goalkeeper saves or player props where reliable inputs do not exist.
- No production ROI/yield/CLV claims.

## Simulator Contract

The simulator-first runner is:

```text
scripts/run_statistical_matchday.py
```

Expected outputs:

```text
match_predictions.csv
scoreline_distribution.csv
team_stats_predictions.csv
player_event_predictions.csv
dynamic_market_lines.csv
betting_edges.csv
recommended_picks.csv
tournament_simulation.csv
tournament_details.csv
top_scorer_predictions.csv
award_predictions.csv
competition_summary.csv
audit_report.json
simulation_contract_report.json
daily_report.html
```

The new contract report is:

```text
simulation_contract_report.json
```

It records:

- expected files;
- required columns for key CSVs;
- missing files;
- schema failures;
- simulation count;
- large-run policy;
- warnings carried over from `audit_report.json`.

## Serious Tournament Simulation Command

For a real tournament-style report, use a large Monte Carlo run:

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/fixtures.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --odds data/input/odds.csv `
  --tournament-config data/input/tournament_config.csv `
  --historical-events data/sample/sample_player_events.csv `
  --out-dir outputs/statistical_simulator_v048_50k `
  --n-simulations 50000 `
  --seed 42 `
  --clean-out-dir
```

For development and CI, use a small smoke run:

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/fixtures.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --odds data/input/odds.csv `
  --tournament-config data/input/tournament_config.csv `
  --historical-events data/sample/sample_player_events.csv `
  --out-dir outputs/statistical_simulator_v048_smoke `
  --n-simulations 25 `
  --seed 42 `
  --clean-out-dir
```

## Interpretation of 50,000 Simulations

A larger simulation count reduces Monte Carlo noise in tournament probabilities. It does not fix:

- weak input data;
- uncalibrated model probabilities;
- missing lineups;
- missing player-event history;
- approximate knockout bracket assumptions.

For serious reports, use the same seed/config when comparing versions.

## Report Direction

v0.48.0 only prepares the contract. Later v0.48.x work should improve reports in this order:

1. advanced single-match report;
2. matchday summary;
3. tournament probability report;
4. evaluation report against baselines;
5. optional paper-value layer on top of simulator outputs.

## Acceptance Criteria

- [x] Package version is aligned to `0.48.0`.
- [x] The main runner audit identifies `v0.48_statistical_simulator_upgrade`.
- [x] The runner writes `simulation_contract_report.json`.
- [x] The contract report validates expected simulator output files and key schemas.
- [x] Tournament simulation records `n_simulations`, `seed` and detail sample policy.
- [x] Odds remain optional and are not required for simulation.
- [x] Player props remain gated by current lineups/squads.
- [x] Missing/unreliable markets remain explicit through warnings or availability flags.
- [x] A focused v0.48 smoke test validates the simulator contract.

## Validation

Focused tests:

```bash
pytest tests/test_v047_statistical_matchday_smoke.py tests/test_v048_statistical_simulator_contract.py
```

Static check:

```bash
python -m compileall -q src scripts/run_statistical_matchday.py tests/test_v047_statistical_matchday_smoke.py tests/test_v048_statistical_simulator_contract.py
```

Manual serious-run check:

```bash
python scripts/run_statistical_matchday.py \
  --fixtures data/input/fixtures.csv \
  --lineups data/input/current_lineups.csv \
  --squads data/input/squads.csv \
  --odds data/input/odds.csv \
  --tournament-config data/input/tournament_config.csv \
  --historical-events data/sample/sample_player_events.csv \
  --out-dir outputs/statistical_simulator_v048_50k \
  --n-simulations 50000 \
  --seed 42 \
  --clean-out-dir
```

## Known Limitations

- Current match-score generation still uses an auditable independent Poisson profile, not Dixon-Coles or bivariate Poisson.
- Knockout simulation remains approximate unless explicit knockout fixtures/bracket rules are supplied.
- Top scorer and award outputs are approximate and should be labelled experimental.
- Report design is still basic; v0.48.1 should improve visual and interpretive reporting.
- Full ROI/yield/CLV requires real odds snapshots or paper trading over time.
