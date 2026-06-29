# v0.26 Statistical Model Upgrade

This release focuses on improving prediction performance before moving deeper into betting execution.

## Match model improvements

- Optional Dixon-Coles low-score correction via `dixon_coles_rho`.
- Model Lab includes new Dixon-Coles candidate configs:
  - `dc_draw_light`
  - `dc_draw_medium`
  - `dc_draw_strong`
  - `dc_light_hardening`
- The default independent Poisson path remains available with `dixon_coles_rho = 0.0`.
- All scoreline matrices are renormalized after correction so probabilities remain auditable.

Why this matters: the previous Poisson profile model was overconfident and often weak on draws. Dixon-Coles only touches low-score cells (`0-0`, `1-0`, `0-1`, `1-1`), which is exactly where football draw behaviour usually needs correction.

## Player prop model improvements

The Champion Prop Lab now includes v0.26 challenger architectures that try to improve on the recovered v0.16-style model:

- `v26_v16_starter_minutes`
- `v26_sot_conditional`
- `v26_nb_moderate`
- `v26_nb_shots_only`
- `v26_card_conservative_role`

New statistical options:

- Starter-role expected minutes: uses historical starter/substitute medians instead of one player median for every lineup role.
- Optional negative-binomial count-to-probability link for overdispersed events.
- Optional shots-on-target conditional layer from expected shots × historical SOT conversion.
- Market-specific count distribution settings.

The lab still selects the champion separately by market, so one architecture can win for shots while another wins for cards.

## Recommended workflow

### 1. Match model lab

```powershell
python scripts/run_model_lab.py `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --out-dir outputs/model_lab_current `
  --clean-out-dir `
  --n-trials 14
```

For faster smoke runs:

```powershell
python scripts/run_model_lab.py `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --out-dir outputs/model_lab_current `
  --clean-out-dir `
  --n-trials 14 `
  --max-test-matches 350
```

### 2. Player prop champion lab

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

### 3. Final matchday with selected configs

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/fixtures.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --odds data/input/odds.csv `
  --tournament-config data/input/tournament_config.csv `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --model-config outputs/model_lab_current/best_model_config.json `
  --calibration-model outputs/model_lab_current/best_calibration_model.json `
  --event-model-config outputs/player_prop_champion_full/prediction_registry.json `
  --out-dir outputs/statistical_matchday_current `
  --clean-out-dir `
  --no-demo-picks
```

## Honest limitations

- Dixon-Coles rho is selected by model lab, not maximum-likelihood fitted yet.
- Player props still depend heavily on expected minutes quality.
- Yellow cards remain fragile without referee and role/contact data.
- Real betting validation still requires real odds and closing-line tracking.
