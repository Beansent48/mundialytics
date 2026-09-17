# Audit and regression report v0.3.1

This report records the cautious audit performed after the v0.3 patches.

## Checks performed

- Full package import/compile check with `python -m compileall`.
- Unit/regression test suite with `pytest`.
- National model training and national fixture prediction.
- Club model training and club fixture prediction.
- Negative control: club model against national fixtures must fail.
- Walk-forward backtest.
- Demo daily picks generation.
- Paper ledger append.
- Source adapters smoke test for Football-Data.co.uk, international-results and OpenFootball sample structures.

## Bugs found and fixed

### 1. Raw lineups did not match canonical odds/events

Odds and player events were canonicalized, but projected lineups were still raw. This meant `Sustituto+` often failed to find a replacement even when the sample lineup file contained one.

Fix: added `load_lineups()` and normalized lineups inside `MinutesModel` and `SubstitutePlusModel`.

### 2. Team identity helper could silently truncate rows

When a dataframe had `team` but no `team_scope`, `DataFrame.get("team_scope", "unknown")` returned a string and `zip()` iterated characters of that string.

Fix: added a row-aligned scope helper and regression test.

### 3. Paper tracking CLI/documentation mismatch

The CLI did not accept `--created-at`, although reproducible paper tracking needs a stable timestamp.

Fix: added `--created-at` to `paper_track.py append`.

### 4. Paper ledger summary hid open stake

The summary showed zero stake when all picks were still open.

Fix: ledger summary now reports `total_stake`, `open_stake`, `settled_stake` and `roi_on_settled`.

### 5. Football-Data.co.uk date ambiguity

Football-Data CSV dates are commonly day-first. Generic pandas parsing can be ambiguous.

Fix: adapter now parses dates with `dayfirst=True` before schema normalization.

### 6. Walk-forward backtest could use stale ELO/features

With `retrain_every > 1`, the model was cached, but ELO and rolling features were also cached. That meant predictions between retrains did not use all matches available before the test match.

Fix: ELO and rolling features are rebuilt for every test match; only the fitted model is cached between retrains.

## Latest verification

Executed successfully with thread limits to avoid BLAS oversubscription in the container:

```bash
python -m compileall -q src scripts app tests
python scripts/diagnose_dataset.py --matches data/sample/sample_matches.csv --out /tmp/diag.json
python scripts/train_from_csv.py --matches data/sample/sample_matches.csv --model-out /tmp/national.pkl --model-type poisson
python scripts/train_from_csv.py --matches data/sample/sample_club_matches.csv --model-out /tmp/club.pkl --model-type poisson
python scripts/predict_fixtures.py --bundle /tmp/national.pkl --fixtures data/sample/sample_national_fixtures.csv --out /tmp/national_preds.csv
python scripts/predict_fixtures.py --bundle /tmp/club.pkl --fixtures data/sample/sample_club_fixtures.csv --out /tmp/club_preds.csv
python scripts/backtest_from_csv.py --matches data/sample/sample_matches.csv --out /tmp/backtest.csv --summary-out /tmp/backtest.json --min-train-matches 10 --model-type poisson --retrain-every 5
python scripts/run_demo.py
python scripts/paper_track.py append --picks outputs/demo_daily_picks.csv --ledger /tmp/ledger.csv --created-at 2026-06-15T16:45:00Z --stake 1.0
python -m pytest -q
```

Result: `14 passed`.

## Still not production betting

The project is more robust, but still needs large real data, probability calibration, real odds/liquidity, lineups/injuries feeds and long paper-trading evaluation before any real-money usage.
