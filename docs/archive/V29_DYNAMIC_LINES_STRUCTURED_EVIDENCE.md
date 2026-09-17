# v0.29 Dynamic Lines & Structured Evidence

v0.29 adds a market-board layer on top of the existing statistical models. It does **not** replace the match model, team-stat model or player-prop champion models. Instead, it turns their outputs into auditable dynamic over/under lines.

## Markets and scopes

The generated `dynamic_market_lines.csv` includes:

- goals: match total and team goals
- shots: match total and team shots
- shots on target: match total and team SOT
- fouls/cards: match total and team scope
- corners: generated as `not_available` unless reliable corner data exists
- player props: player shots, SOT, fouls, yellow card lines

Each row contains a single line and side, such as `Over 1.5 goals`, `Under 3.5 goals`, `Spain Over 12.5 shots`, or `player_shots Over 1.5`.

## Evidence columns

The design intentionally avoids free-form generated justification text. Every row includes structured evidence:

- `model_probability`
- `fair_odds`
- `recent_hit_rate_n/d`
- `similar_elo_hit_rate_n/d`
- `h2h_recent_hit_rate_n/d`
- `time_window`
- `expected_stat`
- `availability`
- `data_quality_flag`
- `value_label`
- `evidence_tags`
- `reason_code`

H2H evidence is limited by a configurable recency cutoff (`--h2h-years`) and max recent meetings. Similar-Elo evidence is limited by a configurable recency window and rating range.

## Command

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/fixtures.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --odds data/input/odds.csv `
  --tournament-config data/input/tournament_config.csv `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --model-config outputs/rolling_model_lab_current/best_rolling_model_config.json `
  --event-model-config outputs/player_prop_champion_full/prediction_registry.json `
  --out-dir outputs/statistical_matchday_current `
  --clean-out-dir `
  --no-demo-picks `
  --recent-n 10 `
  --h2h-years 5 `
  --similar-elo-years 4 `
  --similar-elo-range 100
```

## Notes

- Corners remain `not_available` unless the input data contains reliable corner counts.
- This layer is for structured prediction/explanation, not final betting staking.
- Odds enrichment is intentionally conservative and only attaches odds on close exact matches.
