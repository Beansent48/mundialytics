# v0.25 — Champion model selection and recovered player-prop architecture

## Purpose

v0.25 stops treating the prediction engine as one single model. The engine now
supports the practical rule required for betting/paper-mode work:

> keep the best validated model per market and per useful segment; block or
> downgrade markets that do not beat their baseline.

This version also recovers the stronger v0.16 player-prop philosophy inside the
new statistical core workflow:

- pre-match expected minutes, never observed test minutes;
- player historical rate per 90;
- team event environment;
- player share of team events;
- sample-size shrinkage toward position/global priors;
- hierarchical calibration by competition/domain/team type/gender/market;
- market policies based on performance versus baseline.

## New command

```powershell
python scripts/run_player_prop_champion_lab.py `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --out-dir outputs/player_prop_champion_current `
  --clean-out-dir `
  --max-test-matches 400 `
  --max-calibration-matches 800 `
  --min-calibration-rows 300 `
  --min-group-rows 400 `
  --min-segment-rows 120
```

For a fast smoke run:

```powershell
python scripts/run_player_prop_champion_lab.py `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --out-dir outputs/player_prop_champion_current `
  --clean-out-dir `
  --max-test-matches 20 `
  --max-calibration-matches 40 `
  --min-calibration-rows 50 `
  --min-group-rows 60 `
  --min-segment-rows 30 `
  --n-trials 2
```

## Outputs

- `player_prop_champion_leaderboard.csv`
- `player_prop_champion_summary.csv`
- `player_prop_segment_metrics.csv`
- `player_prop_champion_predictions.csv`
- `prediction_registry.json`
- `player_prop_champion_audit.json`
- `player_prop_champion_report.html`

## Champion selection

The lab runs several candidate prop architectures:

- `v024_team_share_blend`
- `v16_rate_recovery`
- `team_context_share`
- `conservative_cards_sot`
- `player_rate_heavy`

It then selects a champion separately for:

- `player_shots`
- `player_shots_on_target`
- `player_fouls_committed`
- `player_yellow_card`

The registry is intentionally market-specific. Do not force one universal prop
model across all markets if another candidate wins a specific market.

## Validation policy

A market can only be considered a paper-value candidate if it beats baseline on
both Brier and log loss with enough rows. Otherwise it is downgraded to paper
tracking or curiosity-only.

The audit records:

- `uses_observed_test_minutes = false`
- `lineup_known_backtest = true`
- train/calibration/test match counts
- calibration hierarchy used
- per-market champion/policy

## Relationship to v0.16

v0.25 does not simply copy the old v0.16 report. It reintroduces the key ideas
inside the new v0.20+ statistical core structure and compares them against the
newer team-share model. This makes the engine safer: the old model only wins if
it beats the alternatives in the current champion lab.
