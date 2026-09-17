# Event data pipeline

This project now supports event-data ingestion from StatsBomb Open Data and the Wyscout public event dataset.

## Generated tables

`build_event_datasets.py` can generate:

- `player_events`: one row per player-match with event counts and per90 metrics.
- `team_events`: one row per team-match with aggregated event counts.
- `lineups`: starters, substitutes, estimated minutes and replacement links.
- `tactical_shifts`: StatsBomb starting formations and tactical-shift events.

## Why this matters for props

Player props cannot be modelled only from final scores. The engine needs:

- player baseline rates, e.g. shots per90 and fouls committed per90;
- minutes expectation;
- starter/substitute context;
- team dominance context from ELO/Poisson;
- opponent and role context;
- substitution links for Sustituto+.

## StatsBomb example

```bash
python scripts/build_event_datasets.py statsbomb \
  --input data/raw/statsbomb/open-data/data/events \
  --competition "StatsBomb Open Data" \
  --team-scope club
```

Outputs default to:

```text
data/processed/statsbomb_player_events.csv
data/processed/statsbomb_team_events.csv
data/processed/statsbomb_lineups.csv
data/processed/statsbomb_tactical_shifts.csv
```

## Wyscout example

```bash
python scripts/build_event_datasets.py wyscout \
  --events data/raw/wyscout/events_England.json \
  --matches data/raw/wyscout/matches_England.json \
  --players data/raw/wyscout/players.json \
  --teams data/raw/wyscout/teams.json \
  --competition "Premier League" \
  --season 2017-2018 \
  --team-scope club
```

## Derived metrics

The player-event builder adds useful columns when source fields exist:

- `shots_per90`
- `shots_on_target_per90`
- `fouls_committed_per90`
- `yellow_cards_per90`
- `passes_per90`
- `key_passes_per90`
- `pressures_per90`
- `duels_per90`
- `interceptions_per90`
- `sot_rate`
- `pass_completion`
- `dribble_success_rate`

## Caveats

- Wyscout public events are broad and useful, but not current.
- StatsBomb Open Data is high quality but selective.
- Lineup-derived minutes are approximate when stoppage time is not available.
- Tactical shifts are explicit in StatsBomb, but tactical interpretation still requires modelling.
- Fouls drawn can be source-dependent; not all providers encode the fouled player cleanly.
