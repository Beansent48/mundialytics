# scripts/

~200 CLI entry points. Almost all of them are one-off pipeline stages, backtests
or research runs kept for reproducibility — **you only need the handful below.**

## Start here

| Command | What it does |
|---|---|
| `python scripts/update_season.py` | The one command that matters. Downloads new results, rebuilds the modelling foundation (integrity check + rollback), settles the logged markets and logs the upcoming round before kick-off. `--full` also regenerates the walk-forward cache below. `scripts/run_update.ps1` runs it daily as a Windows scheduled task. |
| `python scripts/audit_prediction_coverage.py` | Which played matches had no prediction logged before kick-off. `--season` for the whole season; without it, the last 7 days, failing on any miss not listed in `data/curated/coverage_known_gaps.csv`. Runs inside the daily refresh. |
| `python scripts/generate_deployed_walkforward.py` | Walk-forward predictions of the deployed engine over 2020/21–2025/26 — the file the benchmark and the calibration figure score. |
| `python scripts/benchmark_vs_bet365.py` | Reproduces the headline number: engine vs Bet365 closing odds, RPS and log loss, split by season and league. |
| `python scripts/plot_calibration.py` | Regenerates the reliability diagram in the README. |
| `uvicorn api.main:app --port 8000` + `npm --prefix web run dev` | The web UI (Next.js in `web/`), including the single-fixture analysis the API serves at `/match/{competition}/{fixture}`. |

## Everything else, by prefix

The naming is systematic, so the prefix tells you what a script is:

| Prefix | Count | What it is |
|---|---|---|
| `download_` `fetch_` `import_` | 25 | Ingestion from public sources (football-data.co.uk, ESPN, Sofascore, StatsBomb, ClubElo, FBref, Understat). |
| `build_` | 29 | Dataset construction: foundation, features, xG, squads, ratings, SquadLab cards. |
| `train_` `fit_` `calibrate_` | 13 | Model fitting and calibration. |
| `evaluate_` `validate_` `backtest_` `measure_` | 27 | Offline evaluation. Always temporal out-of-sample. |
| `experiment_` | 20 | Research runs. **Several of these are negative results** kept on purpose — see the "What didn't work" section of the main README. |
| `diagnose_` | 6 | Investigations into a specific failure or gap, e.g. `diagnose_market_gap.py`. |
| `audit_` `quality_gate` `run_data_audit` | 7 | Data-quality gates. |
| `oddspapi_` | 15 | Odds-provider adapter from the betting phase (dormant): fixtures, historical odds, market mapping. |
| `predict_` `run_` `simulate_` | 15 | Prediction and simulation entry points, mostly from the earlier manual-input workflow. |

Some of the older entry points predate the deployed engine and are kept for the
record rather than for use: `run_statistical_matchday.py` is the World Cup-era
runner over hand-written fixture and lineup CSVs (`data/input/`), and
`predict_match.py` loads a v0.4 model bundle that no longer unpickles with a
current scikit-learn. For a fixture today, use the API.

## Two things that will trip you up

**`scripts/test_*.py` are not tests.** `test_lineup_oracle.py`, `test_xi_strength.py`
and three others are research experiments that happen to start with `test_`. The
real suite is `tests/`, and CI runs `pytest tests/` for exactly this reason —
plain `pytest` from the repo root would try to collect them.

**Most scripts need data that isn't in the repo.** `data/raw/`, `data/external/`
and most of `data/processed/` are gitignored (they're large and re-downloadable).
`python scripts/update_season.py` fetches what the live pipeline needs; the older
per-source download commands are in
[`docs/archive/DATA_DOWNLOADS.md`](../docs/archive/DATA_DOWNLOADS.md). The exception is
`data/processed/logs/predictions_log.csv`, which is versioned deliberately: it is
the forward-test track record and losing it would mean losing the only
un-backfittable evidence in the project.
