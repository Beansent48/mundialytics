# scripts/

~200 CLI entry points. Almost all of them are one-off pipeline stages, backtests
or research runs kept for reproducibility — **you only need the handful below.**

## Start here

| Command | What it does |
|---|---|
| `python scripts/update_season.py` | The one command that matters. Refreshes results, re-fits, re-predicts and logs the upcoming round. Everything else is a stage of this. |
| `python scripts/benchmark_vs_bet365.py` | Reproduces the headline number: engine vs Bet365 closing odds, RPS and log loss, split by season and league. |
| `python scripts/plot_calibration.py` | Regenerates the reliability diagram in the README. |
| `python scripts/run_statistical_matchday.py` | Predictions for a single matchday, with the full market set. |
| `python scripts/predict_match.py` | One fixture, one report. |
| `streamlit run app/streamlit_app.py` | The UI over all of the above. |

## Everything else, by prefix

The naming is systematic, so the prefix tells you what a script is:

| Prefix | Count | What it is |
|---|---|---|
| `download_` `fetch_` `import_` | ~30 | Ingestion from public sources (football-data.co.uk, Understat, StatsBomb, ClubElo, FBref). |
| `build_` | ~30 | Dataset construction: canonical schema, features, model-ready snapshots, registries. |
| `train_` `fit_` `calibrate_` | ~12 | Model fitting and Platt calibration. |
| `evaluate_` `validate_` `backtest_` `measure_` | ~25 | Offline evaluation. Always temporal out-of-sample. |
| `experiment_` | ~15 | Research runs. **Several of these are negative results** kept on purpose — see the "What didn't work" section of the main README. |
| `diagnose_` | ~10 | Investigations into a specific failure or gap, e.g. `diagnose_market_gap.py`. |
| `audit_` `quality_gate` `run_data_audit` | ~6 | Data-quality gates. |
| `oddspapi_` | ~16 | Odds-provider adapter: fixtures, historical odds, market mapping. |
| `predict_` `run_` `simulate_` | ~20 | Prediction and simulation entry points. |

## Two things that will trip you up

**`scripts/test_*.py` are not tests.** `test_lineup_oracle.py`, `test_xi_strength.py`
and three others are research experiments that happen to start with `test_`. The
real suite is `tests/`, and CI runs `pytest tests/` for exactly this reason —
plain `pytest` from the repo root would try to collect them.

**Most scripts need data that isn't in the repo.** `data/raw/`, `data/external/`
and most of `data/processed/` are gitignored (they're large and re-downloadable).
Run the `download_` scripts first, or see
[`docs/DATA_DOWNLOADS.md`](../docs/DATA_DOWNLOADS.md). The exception is
`data/processed/logs/predictions_log.csv`, which is versioned deliberately: it is
the forward-test track record and losing it would mean losing the only
un-backfittable evidence in the project.
