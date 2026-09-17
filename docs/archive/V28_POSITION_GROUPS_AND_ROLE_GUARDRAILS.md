# Mundialytics v0.28 — Position Groups & Role Guardrails

This version improves the statistical player-prop layer before betting integration.

## Why

The v0.27 segment report showed that the global player-prop model was strong, but several fine-grained position strings had large calibration bias. Provider positions such as `left center forward`, `right wing`, `center attacking midfield` and `goalkeeper` were too fragmented for stable calibration.

## Changes

- Verbose/free-text positions are normalized into compact tactical keys: `st`, `lw`, `rw`, `am`, `cm`, `dm`, `fb`, `wb`, `cb`, `gk`.
- Tactical keys are grouped into stable role groups: `forward`, `winger`, `attacking_midfield`, `central_midfield`, `defensive_midfield`, `fullback_wingback`, `center_back`, `goalkeeper`, `unknown_outfield`.
- Player-prop champion lab now uses position-group priors and expected-minute fallbacks.
- Hierarchical calibration now includes role-aware levels before broader fallbacks:
  - competition + position_group
  - domain/team_type/gender/context + position_group
  - position_group
  - competition
  - domain_context
  - team_type/gender
  - market global
- Goalkeepers are hard-blocked for attacking props (`player_shots`, `player_shots_on_target`). Outfield defenders are **not** blocked: centre-backs, fullbacks and defensive midfielders can still receive shot/SOT probabilities, just with role-appropriate priors and calibration.

## Recommended command

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

## What to inspect

- `player_prop_champion_summary.csv`: whether the v0.28 challengers beat v0.27 by market.
- `player_prop_segment_metrics.csv`: whether position-group bias improves.
- `prediction_registry.json`: role guardrails and segment policies for deployment.

## Honest limitation

This still evaluates statistical probability quality. Real betting value requires real odds, closing-line tracking and paper PnL simulation later.
