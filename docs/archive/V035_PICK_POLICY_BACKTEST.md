# v0.35 Pick Policy Backtest

This version adds a transparent pick-policy backtesting layer.

## What it does

`scripts/backtest_pick_policy.py` creates historical betting-signal rows from a historical match prediction backtest, then tunes simple pick policies over a chronological train/validation/test split.

It supports two modes:

1. **Signal-only mode**: no historical odds supplied. This evaluates calibration, hit rate and signal quality. It is useful for checking whether the model probabilities are sensible, but it does **not** prove profitability.
2. **Priced value mode**: historical bookmaker odds supplied. This evaluates ROI, profit, edge and EV. This is the real betting-strategy validation mode.

## Inputs

Required:

```powershell
python scripts/backtest_pick_policy.py `
  --match-backtest outputs/evaluation_current/match_backtest_predictions.csv `
  --out-dir outputs/pick_policy_backtest_current
```

Optional odds:

```powershell
python scripts/backtest_pick_policy.py `
  --match-backtest outputs/evaluation_current/match_backtest_predictions.csv `
  --historical-odds data/odds/historical_odds.csv `
  --out-dir outputs/pick_policy_backtest_current
```

## Historical odds schema

Recommended columns:

- `match_id`
- `market`: `1x2`, `goals`, `btts`
- `selection`: for 1X2 use `home`, `draw`, `away`; for BTTS use `yes`, `no`; for goals use `over`, `under`
- `line`: only needed for goals totals, e.g. `2.5`
- `book_odds`: decimal odds

Aliases such as `odds`, `price`, or `decimal_odds` are accepted and normalised to `book_odds`.

## Outputs

- `pick_policy_signals.csv`
- `pick_policy_leaderboard.csv`
- `pick_policy_selected_picks.csv`
- `pick_policy_best.json`
- `pick_policy_summary.json`

## Important limitation

Without historical odds, this is not a profitable-picks test. It only tells whether the model signals would have been directionally correct often enough at different probability/fair-odds thresholds. Real value validation requires bookmaker odds.
