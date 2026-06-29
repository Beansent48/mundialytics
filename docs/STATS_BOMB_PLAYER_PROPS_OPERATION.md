# StatsBomb player-props operation

The project no longer treats empty prop columns as acceptable. Match-result files such as `international_results` are only for the match engine: ELO, goals, 1X2, O/U and tournament simulation. Player props require real event data.

## Recommended free event source

Use **StatsBomb Open Data** as the first serious event source. It includes event JSON and lineups per match. The event files contain real football actions such as shots, passes, fouls, cards, substitutions, Starting XI and Tactical Shift events. This is the closest source to the scouting dashboard workflow.

Wyscout public data is also supported, but StatsBomb is preferred here because it exposes richer event fields and tactical events in a very accessible JSON structure.

## What the bot extracts from StatsBomb

From `events/*.json`:

- player shots
- bookmaker-like shots on target: Goal, Saved, Saved to Post
- goals
- passes, complete passes, key passes and assists
- fouls committed
- fouls drawn
- yellow/red cards from Foul Committed and Bad Behaviour events
- pressures, duels, dribbles, interceptions, recoveries

From Starting XI / Substitution / Tactical Shift events:

- starters
- substitute entries
- minutes played
- replacement player
- replacement minute
- formation changes
- tactical shifts

## Shot-on-target rule

For betting-style markets, the internal `shots_on_target` definition is:

```text
Goal OR Saved OR Saved to Post
```

Ordinary blocked shots are not counted. Some bookmakers/Opta count a last-defender block that prevents a goal as a SOT, but StatsBomb Open Data does not expose that distinction reliably enough to include it safely. The field is therefore labelled as source-compatible, not official Betfair/Opta settlement data.

## One-time setup

```powershell
python scripts/download_data_sources.py statsbomb-open-data --out data/raw/statsbomb/open-data/data
```

If automatic download fails, download this manually:

```text
https://github.com/statsbomb/open-data/archive/refs/heads/master.zip
```

Then run:

```powershell
python scripts/setup_statsbomb_open_data.py --zip C:\path\to\open-data-master.zip --out data/raw/statsbomb/open-data/data
```

## Build real player-event datasets

```powershell
python scripts/build_event_datasets.py statsbomb `
  --input data/raw/statsbomb/open-data/data `
  --team-scope club `
  --player-events-out data/processed/statsbomb_player_events.csv `
  --team-events-out data/processed/statsbomb_team_events.csv `
  --lineups-out data/processed/statsbomb_lineups.csv `
  --tactical-out data/processed/statsbomb_tactical_shifts.csv `
  --diagnostic-out outputs/statsbomb_event_diagnostic.json
```

## Strict event coverage check

```powershell
python scripts/diagnose_event_data.py `
  --player-events data/processed/statsbomb_player_events.csv `
  --lineups data/processed/statsbomb_lineups.csv `
  --out outputs/statsbomb_event_diagnostic.json `
  --strict
```

If this fails, do not validate player props yet. It means the event dataset is too small, too empty or missing required event markets.

## Validate player props

```powershell
python scripts/validate_player_props.py `
  --player-events data/processed/statsbomb_player_events.csv `
  --lineups data/processed/statsbomb_lineups.csv `
  --out-dir outputs/validation_player_props_statsbomb `
  --min-train-matches 50 `
  --test-matches 300
```

## One-command player props pipeline

```powershell
python scripts/run_player_props_pipeline.py `
  --statsbomb-data data/raw/statsbomb/open-data/data `
  --team-scope club `
  --out-dir outputs/player_props_statsbomb `
  --min-matches 50 `
  --min-player-rows 500 `
  --min-train-matches 50 `
  --test-matches 300
```

## Important limitation

StatsBomb Open Data gives historical event data, not live daily World Cup/club data. It is excellent for building and validating the player-props engine. To use the same logic on upcoming matches, you still need expected lineups/minutes and current odds. The bot can ingest those via CSV/manual export until a paid/live provider or Betfair feed is connected.
