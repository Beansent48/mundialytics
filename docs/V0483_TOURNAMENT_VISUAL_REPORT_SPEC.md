# v0.48.3 — Tournament Visual Report Spec

## 1. Context

Mundialytics is moving simulator-first: the statistical engine should produce useful football tournament analysis before expanding betting-pick workflows.

v0.48.0 established the simulator contract. v0.48.1 improved the match report. v0.48.2 added matchday summary rankings. v0.48.3 adds a tournament-focused visual report layer over existing Monte Carlo outputs.

## 2. Problem

`tournament_simulation.csv` contains useful probabilities, but the daily HTML report previously showed it mostly as a raw table. Users need a clearer tournament view:

- champion race
- qualification race
- group winner race
- expected group tables
- paths by round
- attacking projection
- uncertainty watchlist
- competition/player award context when available

## 3. Objective

Create an auditable tournament report layer that summarizes existing simulation outputs without changing the probability model or adding betting behavior.

## 4. Current Behavior

The runner generates `tournament_simulation.csv` and `tournament_details.csv`, and the HTML report includes a generic tournament simulation table.

## 5. Desired Behavior

The runner should additionally generate:

```text
tournament_report.csv
tournament_report.json
```

The HTML report should include a `Tournament Visual Report` section with compact probability bars and categorized tables.

## 6. Scope

Included:

- Add a tournament report summary module.
- Generate tournament report CSV/JSON from existing outputs.
- Add tournament visual sections to `daily_report.html`.
- Add contract validation for the new outputs and HTML section.
- Add focused v0.48.3 smoke test.
- Update README and CHANGELOG.

## 7. Non-Goals

Not included:

- No model retraining.
- No change to Monte Carlo score sampling or probability logic.
- No new odds provider.
- No betting picks, stake sizing, ROI or live execution.
- No new dependencies.
- No full UI framework.
- No charting dependency.

## 8. Inputs

- `tournament_simulation.csv` data frame.
- `tournament_details.csv` data frame.
- Match predictions.
- Fixtures/tournament config for group context.
- Competition summary and top scorer outputs when available.
- Audit metadata.

## 9. Outputs

- `tournament_report.csv`
- `tournament_report.json`
- Updated `daily_report.html`
- Updated `simulation_contract_report.json`

## 10. Constraints

- Odds are optional.
- Betting recommendations must not be introduced.
- Missing/unreliable data must remain explicit.
- 50,000 simulations are recommended for serious tournament probability reports.
- Small simulation counts are acceptable for smoke tests but must be labelled as such.

## 11. Architecture Impact

Affected modules/files:

- `src/mundialytics/statistical_core/tournament_report.py`
- `src/mundialytics/statistical_core/reporting.py`
- `src/mundialytics/statistical_core/simulation_contract.py`
- `scripts/run_statistical_matchday.py`

No folder restructuring.

## 12. Edge Cases

- No tournament context: report may be empty but files should still exist.
- Small simulation count: data quality flag should indicate smoke-run probability noise.
- Missing group metadata: group should be `unknown`.
- No top scorer data: player award section should be omitted or empty.
- No explicit knockout bracket: path probabilities remain approximate according to current simulator limitations.

## 13. Acceptance Criteria

- [x] `tournament_report.csv` is generated.
- [x] `tournament_report.json` is generated.
- [x] CSV contains categorized rows for champion/qualification/group/uncertainty views when tournament simulation exists.
- [x] HTML contains `Tournament Visual Report`.
- [x] Contract validates the new files.
- [x] Focused v0.48.3 test passes.
- [x] Existing v0.47-v0.48.2 focused smoke tests remain compatible.

## 14. Validation Plan

Automated validation:

```bash
python -m compileall -q src scripts/run_statistical_matchday.py tests/test_v0483_tournament_report.py
pytest tests/test_v0483_tournament_report.py -q
```

Focused regression checks:

```bash
pytest tests/test_v047_statistical_matchday_smoke.py -q
pytest tests/test_v048_statistical_simulator_contract.py -q
pytest tests/test_v0481_advanced_match_report.py -q
pytest tests/test_v0482_matchday_summary.py -q
```

Manual/smoke validation:

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/fixtures.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --odds data/input/odds.csv `
  --tournament-config data/input/tournament_config.csv `
  --historical-events data/sample/sample_player_events.csv `
  --out-dir outputs/statistical_simulator_v0483 `
  --n-simulations 50000 `
  --seed 42 `
  --clean-out-dir
```

## 15. Documentation Updates

- [x] README
- [x] CHANGELOG
- [x] v0.48.3 spec

## 16. Risks

- The report can make approximate tournament probabilities look more definitive than they are.
- Small simulation counts are noisy.
- Existing knockout logic is approximate unless explicit knockout fixtures are supplied.
- Player award projections remain experimental.

## 17. Rollback Plan

Revert:

- `src/mundialytics/statistical_core/tournament_report.py`
- v0.48.3 additions in `run_statistical_matchday.py`
- tournament report additions in reporting/contract/tests/docs
- version metadata back to v0.48.2
