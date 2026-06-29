# Feature Spec: v0.48.2 Matchday Summary Rankings

## 1. Context

Mundialytics is now being developed simulator-first. v0.48.1 improved the advanced daily HTML report, but the report still mostly presented raw tables. v0.48.2 adds a daily statistical reading layer that ranks the matchday by the most relevant simulator signals.

## 2. Problem

The simulator generates many outputs:

- match probabilities
- scoreline distributions
- dynamic lines
- team stats
- player stats
- tournament probabilities
- data-quality warnings

Without a matchday ranking layer, the user must manually inspect many tables to understand what matters most today.

## 3. Objective

Create a simulator-first matchday summary that orders the day by statistical characteristics, not betting recommendations.

The report should answer questions such as:

- Which matches project the highest goal environment?
- Which matches project the lowest goal environment?
- Which matches are most balanced?
- Which teams are the strongest favorites?
- Which matches have the most uncertainty?
- Which matches have the strongest BTTS lean?
- Which dynamic statistical lines stand out?
- Which matches need data-quality review?

## 4. Current Behavior

`run_statistical_matchday.py` generates `daily_report.html` plus CSV/JSON outputs, but it does not create a dedicated `matchday_summary.csv` or grouped matchday summary JSON.

## 5. Desired Behavior

The pipeline should generate:

```text
matchday_summary.csv
matchday_summary.json
```

The HTML report should include a `Matchday Summary Rankings` section.

The summary should remain explicitly statistical:

- no stake sizing
- no ROI/yield claims
- no best-bet language
- no live betting behavior
- no dependency on odds

## 6. Scope

Included:

- Add a small matchday summary module.
- Build rankings from existing simulator outputs.
- Add summary CSV/JSON outputs.
- Add summary section to `daily_report.html`.
- Add contract validation for the new outputs and HTML section.
- Add focused smoke tests.

## 7. Non-Goals

Not included:

- No model retraining.
- No calibration changes.
- No feature engineering changes.
- No OddsPapi changes.
- No betting recommendation engine.
- No stake sizing.
- No ROI/yield/CLV evaluation.
- No architecture rewrite.

## 8. Inputs

- `match_predictions`
- `scoreline_distribution`
- `dynamic_market_lines`
- `team_stats_predictions`
- `player_event_predictions`
- `audit`

## 9. Outputs

New:

```text
matchday_summary.csv
matchday_summary.json
```

Updated:

```text
daily_report.html
simulation_contract_report.json
audit_report.json
```

## 10. Ranking Categories

Initial categories:

```text
high_goal_expectation
low_goal_environment
most_balanced_matches
strongest_favorites
highest_uncertainty
btts_lean
top_dynamic_statistical_signals
data_quality_watchlist
```

## 11. Output Schema

`matchday_summary.csv` should include at least:

```text
ranking_category
rank
match_id
match
home_team
away_team
metric_name
metric_value
secondary_metric_name
secondary_metric_value
market
scope
side
line
over_under
statistical_label
data_quality_flag
evidence_tags
short_structured_reason
```

## 12. Constraints

- Must not require odds.
- Must not call external APIs.
- Must not change model outputs.
- Must not invent unavailable markets.
- Must keep player inference gated by current lineups/squads.
- Must preserve CLI compatibility.

## 13. Acceptance Criteria

- [ ] `run_statistical_matchday.py` generates `matchday_summary.csv`.
- [ ] `run_statistical_matchday.py` generates `matchday_summary.json`.
- [ ] `daily_report.html` contains `Matchday Summary Rankings`.
- [ ] `simulation_contract_report.json` validates the new summary output.
- [ ] Focused tests pass.
- [ ] Existing v0.47/v0.48/v0.48.1 smoke tests still pass.
- [ ] No new dependency is added.
- [ ] No model/odds/live betting behavior is changed.

## 14. Validation Plan

Automated validation:

```bash
pytest tests/test_v047_statistical_matchday_smoke.py
pytest tests/test_v048_statistical_simulator_contract.py
pytest tests/test_v0481_advanced_match_report.py
pytest tests/test_v0482_matchday_summary.py
```

Smoke test:

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/fixtures.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --odds data/input/odds.csv `
  --tournament-config data/input/tournament_config.csv `
  --historical-events data/sample/sample_player_events.csv `
  --out-dir outputs/statistical_simulator_v0482 `
  --n-simulations 1000 `
  --seed 42 `
  --clean-out-dir
```

Large simulation validation:

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/fixtures.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --odds data/input/odds.csv `
  --tournament-config data/input/tournament_config.csv `
  --historical-events data/sample/sample_player_events.csv `
  --out-dir outputs/statistical_simulator_v0482_50k `
  --n-simulations 50000 `
  --seed 42 `
  --clean-out-dir
```

## 15. Risks

- Ranking labels can be misread as betting advice.
  - Mitigation: wording remains statistical and paper-only.
- Low sample data can create noisy rankings.
  - Mitigation: include data-quality flags and watchlist rows.
- Too many ranking categories can clutter the report.
  - Mitigation: keep top-N sections and expose CSV/JSON for deeper inspection.

## 16. Rollback Plan

Remove:

```text
src/mundialytics/statistical_core/matchday_summary.py
tests/test_v0482_matchday_summary.py
docs/V0482_MATCHDAY_SUMMARY_RANKINGS_SPEC.md
```

Then remove the small integration points from:

```text
scripts/run_statistical_matchday.py
src/mundialytics/statistical_core/reporting.py
src/mundialytics/statistical_core/simulation_contract.py
README.md
CHANGELOG.md
```
