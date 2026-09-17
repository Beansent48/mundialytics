# Feature Spec: v0.48.1 — Advanced Match Report

## 1. Context

Mundialytics v0.48.0 established a simulator-first foundation with auditable CSV/JSON outputs and support for large Monte Carlo tournament runs. The next step is to make those outputs easier to inspect and communicate through a stronger daily HTML report.

The project remains a football statistical prediction and simulation engine. Betting picks and odds comparison are optional paper-mode layers, not the core product.

## 2. Problem

The current `daily_report.html` is useful as a generated artifact, but it is too flat for serious analysis. It does not clearly separate match probabilities, top scorelines, dynamic lines, team/player statistics, missing-market policy, data quality and simulation metadata.

## 3. Goals

- Improve `daily_report.html` as the primary human-readable simulator report.
- Surface existing model and simulator outputs without retraining models.
- Add per-match report cards.
- Show top scorelines and dynamic lines per match.
- Show data-quality flags and `not_available` policy explicitly.
- Keep optional paper-value information clearly separated from the statistical report.
- Extend the simulator contract so required HTML report sections are validated.

## 4. Non-Goals

- No model retraining.
- No new model features.
- No new odds provider.
- No live betting.
- No Betfair Exchange work.
- No paper staking/ROI/CLV tracker.
- No new heavy dependencies.
- No folder-structure rewrite.
- No separate per-match HTML files yet.

## 5. System Flow

```text
fixtures + lineups/squads + historical events
→ match predictions
→ scoreline distribution
→ team/player stat predictions
→ dynamic market lines
→ tournament simulation
→ daily_report.html
→ simulation_contract_report.json
```

The report consumes already-generated simulator outputs. It must not change the statistical meaning of the underlying predictions.

## 6. Inputs

- `data/input/fixtures.csv`
- `data/input/current_lineups.csv`
- `data/input/squads.csv`
- `data/input/odds.csv` when provided, paper-mode only
- `data/input/tournament_config.csv`
- `data/sample/sample_player_events.csv`
- Generated in-memory dataframes from `scripts/run_statistical_matchday.py`

## 7. Outputs

Primary:

- `daily_report.html`

Related contract/audit outputs:

- `simulation_contract_report.json`
- `audit_report.json`

Existing CSV outputs preserved:

- `match_predictions.csv`
- `scoreline_distribution.csv`
- `team_stats_predictions.csv`
- `player_event_predictions.csv`
- `dynamic_market_lines.csv`
- `tournament_simulation.csv`
- `tournament_details.csv`
- `competition_summary.csv`
- `top_scorer_predictions.csv`
- `award_predictions.csv`
- `betting_edges.csv`
- `recommended_picks.csv`

## 8. Report Contract

`daily_report.html` should include these required sections:

- `Mundialytics Statistical Simulator v0.48.1`
- `Executive Summary`
- `Match Probabilities`
- `Advanced Match Cards`
- `Top Scorelines`
- `Dynamic Goal Lines`
- `Not Available Markets`
- `Team Statistics`
- `Player Statistics`
- `Data Quality`
- `Simulation Metadata`

## 9. Constraints

- Must remain runnable without APIs.
- Must remain runnable without real odds.
- Must not invent unavailable markets.
- Must keep player predictions gated by current lineups/squads.
- Must preserve the existing main CLI flow.
- Must avoid new dependencies.
- Must keep output schemas backward-compatible where practical.

## 10. Architecture Impact

Affected files:

- `src/mundialytics/statistical_core/reporting.py`
- `src/mundialytics/statistical_core/simulation_contract.py`
- `scripts/run_statistical_matchday.py`
- `tests/test_v048_statistical_simulator_contract.py`
- `tests/test_v0481_advanced_match_report.py`
- `README.md`
- `CHANGELOG.md`
- `pyproject.toml`
- `src/mundialytics/__init__.py`

No new package layer is introduced.

## 11. Edge Cases

- Empty match predictions.
- Empty scoreline distribution.
- Dynamic lines disabled.
- No odds supplied.
- Demo odds supplied.
- Missing or unavailable markets.
- Player rows with low sample size or unresolved identity flags.
- Small smoke-test simulation counts vs large 50,000 simulation runs.

## 12. Acceptance Criteria

- [ ] Package version is `0.48.1`.
- [ ] Main runner audit version is `v0.48.1_advanced_match_report`.
- [ ] `daily_report.html` contains all required sections.
- [ ] Report shows per-match cards with probabilities, scorelines, dynamic lines, team stats, player stats and data quality.
- [ ] Optional paper-value content is separated from simulator content.
- [ ] `simulation_contract_report.json` validates required HTML sections.
- [ ] Existing simulator CSV outputs are preserved.
- [ ] Focused tests pass.
- [ ] No real API calls are required.

## 13. Validation Plan

Automated validation:

```bash
pytest tests/test_v047_statistical_matchday_smoke.py tests/test_v048_statistical_simulator_contract.py tests/test_v0481_advanced_match_report.py -q
```

Static validation:

```bash
python -m compileall -q src scripts/run_statistical_matchday.py tests/test_v047_statistical_matchday_smoke.py tests/test_v048_statistical_simulator_contract.py tests/test_v0481_advanced_match_report.py
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
  --out-dir outputs/statistical_simulator_v0481 `
  --n-simulations 1000 `
  --seed 42 `
  --clean-out-dir
```

For serious tournament reporting, use:

```powershell
--n-simulations 50000
```

## 14. Rollback Plan

Revert the v0.48.1 changes in:

- `reporting.py`
- `simulation_contract.py`
- `run_statistical_matchday.py`
- version metadata
- tests/docs

The simulator CSV outputs remain compatible with v0.48.0.
