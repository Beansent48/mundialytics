# Mundialytics v0.27 — Rolling & Segment Hardening

v0.27 adds two statistical upgrades before any betting integration:

1. **Rolling-origin match validation**: each fold trains on the past, calibrates on the recent past, and scores future matches. This is stricter than a single temporal holdout and exposes unstable 1X2 models.
2. **Segment deployment policies for player props**: the champion model is still selected per market, but the registry now includes segment-level guardrails. Segments that fail baseline, have thin improvement, or show large calibration bias are downgraded or blocked for value usage.

## Rolling match lab

```powershell
python scripts/run_rolling_model_lab.py `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --out-dir outputs/rolling_model_lab_current `
  --clean-out-dir `
  --n-trials 14 `
  --min-train-matches 900 `
  --calibration-matches 500 `
  --test-matches 250 `
  --step-matches 250 `
  --max-folds 6
```

Outputs:

- `rolling_model_leaderboard.csv`
- `best_rolling_model_config.json`
- `rolling_model_report.html`
- per-trial fold metrics and predictions under `trials/`

## Player prop champion lab

The existing command now writes segment policies into `prediction_registry.json`:

```powershell
python scripts/run_player_prop_champion_lab.py `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --out-dir outputs/player_prop_champion_full `
  --clean-out-dir `
  --max-test-matches 1200 `
  --max-calibration-matches 2500 `
  --min-calibration-rows 500 `
  --min-group-rows 700 `
  --min-segment-rows 180
```

The registry now contains:

- `player_props`: market champions.
- `segment_policies`: guardrails by dimension/segment.

## Why this matters

The goal is not to force one model everywhere. The system should keep the best model per market, validate stability over time, and only deploy predictions in segments where the model beats baseline with acceptable calibration.
