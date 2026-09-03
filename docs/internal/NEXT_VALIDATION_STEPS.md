# Next validation steps

## Product interpretation

The primary objective is to validate the **statistical football engine**, not to prove betting profitability.

Separate layers:

```text
Statistical Engine
→ football probabilities and distributions: 1X2, goals, scorelines, BTTS, corners,
  cards, shots, player events, tournament simulations and awards.

Value Pick Engine
→ later selective layer for a small number of market opportunities when the
  statistical model, data coverage and market calibration justify it.
```

Rules:

- Use statistical metrics to improve the simulator.
- Do not use ROI/profit to select or tune the core statistical model at this stage.
- Do not interpret the optional 1X2 value backtest as the project goal.
- Value picks should later be selective, market-specific and quality-gated, not broad betting on every 1X2 edge.


This is the practical route to check whether the engine has signal before using it for any real decision.


## 0) Build the data foundation first

Before running serious model validation, build a cleaned and profiled canonical dataset.

For one league across multiple seasons:

```powershell
python scripts/build_match_dataset.py `
  --source football-data-uk `
  --inputs "data/raw/football_data/2*/2*_SP1.csv" `
  --out-dir data/processed/foundation_laliga_multi_season `
  --dataset-name foundation_laliga_multi_season `
  --drop-incomplete-goals
```

Inspect:

```text
data/processed/foundation_laliga_multi_season/match_dataset_foundation_report.json
data/processed/foundation_laliga_multi_season/match_dataset_feature_coverage.csv
data/processed/foundation_laliga_multi_season/match_dataset_quality_by_competition_season.csv
data/processed/foundation_laliga_multi_season/match_dataset_anomalies.csv
```

Use the generated `canonical_matches.csv` as the input to validation:

```powershell
python scripts/run_historical_validation.py `
  --matches data/processed/foundation_laliga_multi_season/canonical_matches.csv `
  --out-dir outputs/validation_laliga_foundation_v0494 `
  --min-train-matches 300 `
  --retrain-every 20 `
  --max-backtest-predictions 600 `
  --min-matches-ready 500 `
  --min-backtest-predictions-ready 200 `
  --model-types poisson random_forest_lambda
```

Rules:

- Do not interpret model metrics until the foundation report is acceptable.
- Do not open corners/cards models unless the feature coverage report confirms enough rows.
- Do not mix club and national datasets.
- Keep raw files immutable.




## 0.5) Build model-ready snapshots after the foundation

After `canonical_matches.csv` exists, build leakage-safe model-ready snapshots.

Example for LaLiga:

```powershell
python scripts/build_model_ready_dataset.py `
  --matches data/processed/foundation_laliga_multi_season/canonical_matches.csv `
  --out-dir data/processed/model_ready_laliga_multi_season_v0495 `
  --dataset-name model_ready_laliga_multi_season_v0495
```

Do the same for:

```text
foundation_epl_multi_season
foundation_seriea_multi_season
foundation_bundesliga_multi_season
foundation_ligue1_multi_season
```

Inspect:

```text
model_ready_snapshot_report.json
model_ready_feature_contract.csv
model_ready_match_snapshots.csv
```

Rules:

- Use only columns marked as `feature` in the feature contract for model inputs.
- Keep target columns as labels only.
- Report metrics globally and by league for any future Big 5 model.
- Do not assume xG/ClubElo exists just because the model contract supports it.
- If xG or ClubElo is enriched later, rerun the snapshot builder and compare data-level reports.

Recommended next validation after v0.49.5:

```text
Build all Big 5 snapshots
→ concatenate into a global Big 5 snapshot dataset
→ validate global model metrics
→ report by-league residuals and calibration
→ decide whether league-level lambda calibration is needed
```

## 1) Run a full historical validation

Use `scripts/run_historical_validation.py`. It runs, in one command:

- canonical dataset conversion if needed
- data-quality diagnostics
- walk-forward backtests for Poisson and Random-Forest-lambda models
- readiness gates
- final model training with the best backtest model
- optional historical value backtest if odds are available

### National teams

Download the `results.csv` file from `martj42/international_results` manually if automatic download fails, then run:

```bash
python scripts/run_historical_validation.py \
  --source international-results \
  --input data/raw/international_results/results.csv \
  --out-dir outputs/validation_national \
  --min-train-matches 500 \
  --retrain-every 50 \
  --min-matches-ready 1000 \
  --min-backtest-predictions-ready 300
```

### Clubs with Football-Data.co.uk

Download one league CSV, for example Premier League `E0.csv`, then run:

```bash
python scripts/run_historical_validation.py \
  --source football-data-uk \
  --input data/raw/football_data_uk/2526_E0.csv \
  --season 2025-2026 \
  --auto-football-data-odds \
  --out-dir outputs/validation_epl_2526 \
  --min-train-matches 100 \
  --retrain-every 10 \
  --min-matches-ready 200 \
  --min-backtest-predictions-ready 100
```

For a serious club model, concatenate several seasons first. One season is useful for a smoke test, not for trust.

## 2) Read the output report

Open:

```text
outputs/<validation_folder>/operational_validation_report.json
```

The most important fields are:

- `status`
- `best_model_type`
- `backtests.*.summary.log_loss`
- `backtests.*.summary.rps`
- `quality_gates.*.result.passed`
- `historical_value_backtest.summary`, if odds were provided

## 3) Predict upcoming fixtures

After validation, the script saves a final model bundle in the validation folder.

Then run:

```bash
python scripts/predict_fixtures.py \
  --bundle outputs/validation_national/final_national_poisson_model.pkl \
  --fixtures data/processed/upcoming_national_fixtures.csv \
  --out outputs/today_predictions.csv
```

## 4) Add odds and generate value picks

Prepare a CSV:

```csv
fixture_id,bookmaker,market_type,selection,odds
fx_001,betfair,match_winner,home,1.80
fx_001,betfair,match_winner,draw,3.60
fx_001,betfair,match_winner,away,4.90
```

Then run:

```bash
python scripts/value_from_predictions.py \
  --predictions outputs/today_predictions.csv \
  --odds data/processed/current_match_odds.csv \
  --out outputs/today_value_picks.csv
```

## 5) Track in paper mode

```bash
python scripts/paper_track.py append \
  --picks outputs/today_value_picks.csv \
  --ledger outputs/paper_ledger.csv \
  --stake 1.0
```

After results are known, settle the ledger with an outcomes CSV.

## Interpretation rule

- `READY_FOR_EXTENDED_PAPER_MODE` means the pipeline is clean enough to track.
- It does **not** mean the model is profitable.
- Real staking should only be considered after a long paper ledger, stable calibration, and positive value backtests.


## 6) Statistical-engine evaluation outputs — v0.49.2

`run_historical_validation.py` now writes statistical-engine outputs for each model type:

```text
statistical_engine_<model_type>_summary.json
statistical_engine_goal_errors_<model_type>.csv
statistical_engine_goal_lines_<model_type>.csv
statistical_engine_line_calibration_<model_type>.csv
statistical_engine_scorelines_<model_type>.csv
statistical_engine_calibration_layer_<model_type>.csv
statistical_engine_dixon_coles_scorelines_<model_type>.csv
```

These outputs evaluate:

- expected goals vs actual goals,
- total-goals over/under lines,
- BTTS yes/no,
- exact-score probability and top-k coverage.

The operational report stores these paths under:

```text
backtests.<model_type>.statistical_engine_evaluation
```

Use these files when deciding how to improve the simulator. Treat optional `historical_value_backtest` as diagnostic only.

## 7) Latest local validation interpretation from v0.49.2 planning

The user-run validation shared during v0.49.2 planning showed:

```text
National teams:
- 3000 completed matches
- 1200 backtest predictions
- Poisson outperformed random_forest_lambda on log loss, RPS and accuracy
- Status: READY_FOR_EXTENDED_PAPER_MODE

Club examples:
- EPL 2025/26
- EPL 2024/25
- LaLiga 2024/25
- Serie A 2024/25
```

Across these club examples, Poisson also outperformed `random_forest_lambda` on the main 1X2 metrics. This supports keeping Poisson as the current baseline statistical engine.

Important interpretation:

```text
Poisson baseline: useful statistical baseline
RandomForest lambda: secondary experiment
Value backtest: not validated as a decision engine
Profit/ROI: not evidence for simulator quality at this stage
```


## 8) v0.49.3 calibration/model-improvement validation

The current improvement path is statistical quality, not profit.

After running historical validation, inspect:

```text
statistical_engine_<model_type>_summary.json
statistical_engine_line_calibration_<model_type>.csv
statistical_engine_calibration_layer_<model_type>.csv
statistical_engine_dixon_coles_scorelines_<model_type>.csv
```

What to compare:

```text
Poisson vs random_forest_lambda
baseline scoreline metrics vs Dixon-Coles scoreline metrics
pre-calibration vs post-calibration log loss / Brier / RPS
line calibration gaps by over/under and BTTS
```

New CLI parameters:

```bash
python scripts/run_historical_validation.py \
  ... \
  --poisson-alpha 1.0 \
  --time-decay-half-life-days 365 \
  --rolling-shrinkage-prior-matches 10
```

Interpretation:

- Elo is part of the model contract.
- Time decay and shrinkage are stability tools, not a claim of improved performance by themselves.
- Calibration layer and Dixon-Coles outputs are offline diagnostics until they are validated across multiple leagues/seasons.
- Corners/cards need separate count models and should not be treated as simple derivatives of goals.


## Enriched Big 5 validation path after v0.49.6

The next serious validation should compare three tiers:

```text
Tier A: Level 1 Football-Data foundation only
Tier B: Tier A + ClubElo
Tier C: Tier B + xG
```

For each tier, report:

```text
global metrics
by-league metrics
1X2 log_loss/RPS/Brier
goal MAE/RMSE
scoreline log loss
top1/top3/top5 scoreline coverage
over/under and BTTS calibration
```

Do not claim xG or ClubElo improves the model until metrics are compared on temporal holdout.

## 2026-06-25 — v0.49.7 ClubElo team-history download fix

Status: Implemented

The v0.49.6 ClubElo downloader used daily full-table snapshots by match date, which is too slow for multi-season Big 5 datasets. v0.49.7 makes team-history download the default.

New default:

```text
one ClubElo API request per team alias
```

instead of:

```text
one ClubElo API request per unique match date
```

The legacy daily snapshot mode remains available through `--mode daily-snapshot`, but normal enrichment should use `--mode team-history` and `--source-mode auto`.




## xG provider fallback after v0.49.8

If Understat direct download reports `Could not find Understat datesData JSON in page`, do not block validation.

Use either:

```powershell
python scripts/import_xg_csv.py `
  --input data/external/xg/provider_export.csv `
  --provider provider_csv `
  --out-dir data/external/xg/understat
```

or continue with zero xG coverage:

```powershell
python scripts/enrich_matches_with_xg.py `
  --matches data/processed/enriched/epl_clubelo/canonical_matches_with_clubelo.csv `
  --xg data/external/xg/understat/understat_xg_matches.csv `
  --out-dir data/processed/enriched/epl_clubelo_xg `
  --allow-missing-xg
```

xG remains a Level 3 enrichment, not a hard dependency.


## v0.49.9 validation step

After importing StatsBomb Open Data xG, run xG enrichment against each ClubElo-enriched league dataset and inspect `xg_enrichment_report.json`. If coverage is low, continue with `xg_available=false` and do not interpret Big 5 validation as xG-powered.

## v0.50.0 advanced data validation route

After building provider outputs, validate in this order:

```text
1. Import/merge advanced providers.
2. Enrich canonical matches.
3. Audit advanced data coverage.
4. Build model-ready snapshots.
5. Compare baseline Football-Data+ClubElo vs advanced-enriched snapshots.
6. Report metrics globally and by league.
```

Do not declare xG useful until coverage and temporal-safety checks pass.
