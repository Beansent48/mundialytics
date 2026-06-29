# v0.30 — Dynamic lines, line-specific evidence and price/value separation

This update hardens the v0.29 dynamic market board.

## Main fixes

- Player prop evidence is now **line-specific**:
  - `Over 0.5 player_shots` uses matches with 1+ shots.
  - `Over 1.5 player_shots` uses matches with 2+ shots.
  - `Over 2.5 player_shots` uses matches with 3+ shots.
- Player evidence now uses player-level historical rows instead of team-level totals.
- H2H and similar-Elo evidence are still restricted by recency windows.
- Similar-Elo and H2H small samples are tagged as thin/not enough data rather than treated as fully strong evidence.
- `signal_label` and `value_label` are separated:
  - `signal_label`: statistical/model signal without considering bookmaker odds.
  - `value_label`: price value only after odds are attached.
- Odds matching now supports common aliases like `total_goals -> goals`, but avoids attaching match-total odds to team/player rows.
- The HTML report shows `expected_stat`, odds fields, `signal_label`, `value_label`, evidence tags and reason codes.

## New useful columns

- `signal_label`
- `value_label`
- `value_reason_code`
- `expected_stat`
- `book_odds`
- `implied_probability`
- `edge`
- `ev`
- `recent_hit_rate_n/d`
- `similar_elo_hit_rate_n/d`
- `h2h_recent_hit_rate_n/d`

## CLI additions

```powershell
--h2h-max-matches 8
--similar-elo-max-matches 12
--min-context-sample 3
--min-strong-context-sample 5
--max-player-rows-per-market 60
--no-dynamic-lines
```

Recommended command:

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
  --h2h-max-matches 8 `
  --similar-elo-years 4 `
  --similar-elo-range 100 `
  --similar-elo-max-matches 12 `
  --min-context-sample 3 `
  --min-strong-context-sample 5
```
