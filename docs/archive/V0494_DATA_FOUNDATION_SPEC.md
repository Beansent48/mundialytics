# v0.49.4 — Data Foundation & Match Dataset Treatment Spec

Status: Implemented  
Date: 2026-06-25  
Version: 0.49.4

## 1. Context

The project direction is to improve Mundialytics as a statistical football engine before trying to improve profitability, staking or value-pick logic.

The user explicitly wants to "max out" data quality and data treatment first, then observe how the model reacts. This is the correct sequence for this project:

```text
better raw/canonical data foundation
→ better model inputs
→ more honest evaluation
→ then model/calibration improvements
```

## 2. Problem

Previous validation runs proved that the current Poisson baseline has signal, but model changes such as calibration, time decay and shrinkage do not automatically improve every run.

The next risk is not model complexity. The next risk is training and validating on narrow or insufficiently profiled datasets:

- one season at a time,
- inconsistent coverage across leagues/seasons,
- hidden missing stats for corners/cards/shots,
- duplicated or unsafe match identities,
- mixed club/national contexts,
- event features that appear available but are sparse,
- unverified anomalies such as shots-on-target greater than shots.

## 3. Goals

v0.49.4 adds a data-foundation layer that can:

- Build cleaned canonical match datasets from one or many input files.
- Support multi-season Football-Data.co.uk datasets.
- Preserve raw files and write processed outputs separately.
- Report feature coverage before training.
- Report quality by competition and season.
- Flag anomalies that could damage goals, corners, cards or shot models.
- Integrate the data-foundation summary into historical validation.
- Keep model logic unchanged.

## 4. Non-Goals

This phase does not:

- Add a new predictive model.
- Claim improved metrics.
- Tune calibration gates.
- Implement corners/cards Negative Binomial models.
- Optimize profit, ROI, staking or value picks.
- Download external datasets automatically.
- Mutate raw files in `data/raw`.

## 5. Implemented Files

```text
src/mundialytics/data_quality/match_dataset_foundation.py
scripts/build_match_dataset.py
tests/test_v0494_match_dataset_foundation.py
```

Updated:

```text
scripts/run_historical_validation.py
src/mundialytics/data_quality/__init__.py
docs/
README.md
CHANGELOG.md
pyproject.toml
src/mundialytics/__init__.py
```

## 6. Data Treatment

The data-foundation step performs conservative treatment only:

- Coerces dates and numeric match columns.
- Drops structurally unsafe rows:
  - invalid dates,
  - missing teams,
  - optionally missing goals for training datasets,
  - duplicate `match_id` rows after keeping the first.
- Sorts by date and match ID.
- Preserves `source_file`.
- Does not impute football stats.
- Does not create model-specific features.
- Does not alter raw files.

## 7. Feature Coverage

The foundation report profiles:

```text
goals
shots
shots_on_target
corners
fouls
yellow_cards
red_cards
xg
external_elo
clubelo
```

Coverage is marked as:

```text
available
partial
sparse
unavailable
```

This tells us which future statistical models are data-ready.

For example, a corners/cards model should not be opened until coverage is sufficient in the chosen dataset.

## 8. Anomaly Detection

The anomaly report flags:

- negative goals,
- extreme goal counts,
- shots-on-target greater than shots,
- extreme corner counts,
- extreme yellow-card counts,
- duplicate fixture keys,
- mixed team scopes.

Critical anomalies block the dataset. Warning anomalies require review but do not automatically drop rows.

## 9. CLI Usage

Build a multi-season Football-Data dataset:

```bash
python scripts/build_match_dataset.py \
  --source football-data-uk \
  --inputs "data/raw/football_data/2223/2223_E0.csv" "data/raw/football_data/2324/2324_E0.csv" "data/raw/football_data/2425/2425_E0.csv" \
  --out-dir data/processed/foundation_epl_2223_2425 \
  --dataset-name foundation_epl_2223_2425 \
  --drop-incomplete-goals
```

PowerShell glob example:

```powershell
python scripts/build_match_dataset.py `
  --source football-data-uk `
  --inputs "data/raw/football_data/2*/2*_E0.csv" `
  --out-dir data/processed/foundation_epl_multi_season `
  --dataset-name foundation_epl_multi_season `
  --drop-incomplete-goals
```

Then validate the generated canonical file:

```bash
python scripts/run_historical_validation.py \
  --matches data/processed/foundation_epl_multi_season/canonical_matches.csv \
  --out-dir outputs/validation_epl_foundation_v0494 \
  --min-train-matches 300 \
  --retrain-every 20 \
  --max-backtest-predictions 600 \
  --min-matches-ready 500 \
  --min-backtest-predictions-ready 200 \
  --model-types poisson random_forest_lambda
```

## 10. Outputs

`build_match_dataset.py` writes:

```text
canonical_matches_raw_combined.csv
canonical_matches.csv
match_dataset_foundation_report.json
match_dataset_feature_coverage.csv
match_dataset_quality_by_competition_season.csv
match_dataset_anomalies.csv
match_dataset_dropped_rows.csv
```

`run_historical_validation.py` now also writes the same foundation profile for its validation input:

```text
match_dataset_foundation_report.json
match_dataset_feature_coverage.csv
match_dataset_quality_by_competition_season.csv
match_dataset_anomalies.csv
match_dataset_dropped_rows.csv
```

and includes a `data_foundation` section in `operational_validation_report.json`.

## 11. Acceptance Criteria

Implemented:

- A multi-file data foundation builder exists.
- Feature coverage is reported by feature group.
- Competition/season quality is reported.
- Structural drops are explicit and auditable.
- Anomalies are reported instead of hidden.
- Historical validation consumes the cleaned foundation output.
- Raw files are not modified.
- Model logic is unchanged.
- Focused tests validate the new foundation behavior.

## 12. Validation Plan

Focused validation:

```bash
python -m compileall -q src scripts/build_match_dataset.py scripts/run_historical_validation.py tests/test_v0494_match_dataset_foundation.py
python -m pytest tests/test_v0494_match_dataset_foundation.py -q
```

Recommended local user validation:

1. Build a multi-season Big 5 dataset.
2. Inspect feature coverage.
3. Validate the generated canonical file.
4. Compare v0.49.3 one-season metrics against v0.49.4 multi-season metrics.
