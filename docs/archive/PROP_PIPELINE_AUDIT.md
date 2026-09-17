# Player Props Clean Rebuild and Audit

Do not calibrate or run temporal checks on a predictions CSV unless it has valid match dates and the expected competition domain.

The safe route is:

```powershell
python scripts/run_clean_props_rebuild.py `
  --player-events outputs/player_props_statsbomb_national/statsbomb_player_events.csv `
  --lineups outputs/player_props_statsbomb_national/statsbomb_lineups.csv `
  --out-dir outputs/player_props_statsbomb_clean_audited `
  --exclude-competitions "StatsBomb Open Data" `
  --expected-domain mixed `
  --min-train-matches 100 `
  --test-matches 300 `
  --min-calibration-market-rows 500
```

For men's national-only experiments:

```powershell
python scripts/run_clean_props_rebuild.py `
  --player-events outputs/player_props_statsbomb_national/statsbomb_player_events.csv `
  --lineups outputs/player_props_statsbomb_national/statsbomb_lineups.csv `
  --out-dir outputs/player_props_mens_national_clean_audited `
  --include-competitions "FIFA World Cup" "UEFA Euro" "African Cup of Nations" "Copa America" "Copa América" `
  --expected-domain national `
  --min-train-matches 50 `
  --test-matches 100 `
  --min-calibration-market-rows 200
```

This route fails early if:

- prediction dates are missing,
- placeholder competitions remain,
- club competitions are present in a national-only run,
- duplicate `match_id + player + market_type` rows appear,
- probabilities are outside `[0,1]`,
- minutes/counts are negative or impossible.

Operational inference must still use `run_safe_props_for_lineups.py`: historical players are valid for training only; candidates must come from the supplied current lineup CSV.
