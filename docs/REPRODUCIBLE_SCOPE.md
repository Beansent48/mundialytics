# Reproducible scope: World Cup, national teams and clubs

This project is **not World-Cup-only**. `Mundialytics` is the project name, but the engine is source-agnostic.

The same core pipeline works for:

1. National teams: World Cup, Euros, Copa América, qualifiers, friendlies.
2. Clubs: Premier League, LaLiga, Serie A, Bundesliga, Ligue 1, Champions League, etc.
3. Mixed demos: only for testing; in production train national and club models separately unless you have a good reason to pool them.

## Canonical historical match schema

Minimum required columns:

```text
match_id,date,home_team,away_team,home_goals,away_goals,neutral
```

Recommended columns:

```text
competition,season,stage,team_scope,source
home_shots,away_shots,home_sot,away_sot
home_corners,away_corners,home_fouls,away_fouls
home_yellow_cards,away_yellow_cards
home_xg,away_xg
```

`team_scope` should be:

```text
national | club | mixed | unknown
```

## Canonical future-fixture schema

Minimum required columns:

```text
fixture_id,date,home_team,away_team,neutral
```

Recommended columns:

```text
competition,season,stage,team_scope,source
```

Use `scripts/predict_fixtures.py` after training a bundle:

```bash
python scripts/train_from_csv.py --matches data/sample/sample_matches.csv --model-out models/national_goal_model.pkl
python scripts/predict_fixtures.py --bundle models/national_goal_model.pkl --fixtures data/sample/sample_national_fixtures.csv --out outputs/national_fixture_predictions.csv
```

For clubs:

```bash
python scripts/train_from_csv.py --matches data/sample/sample_club_matches.csv --model-out models/club_goal_model.pkl
python scripts/predict_fixtures.py --bundle models/club_goal_model.pkl --fixtures data/sample/sample_club_fixtures.csv --out outputs/club_fixture_predictions.csv
```

For real use, do not train a club model with national-team data unless you explicitly add features that make that pooling sensible. Club football and national-team football have different scoring environments, scheduling, chemistry, home advantage and player availability dynamics.

## Reproducible data plan

### National teams

- Historical results: international-results style CSV.
- Event/player data: StatsBomb Open Data when competitions are available.
- Future fixtures: OpenFootball fixtures, official schedule manually exported to CSV, or API export.

### Clubs

- Historical results and odds: Football-Data.co.uk.
- Club ratings: ClubElo or your own ELO from historical results.
- Future fixtures: OpenFootball JSON/TXT, Football-Data fixture pages, or manual CSV.
- Player/event data: StatsBomb Open Data where available; otherwise add a licensed/API source later.

## Practical rule

The bot should always produce a **slate**:

```text
future fixtures -> match probabilities -> team events -> player props -> odds comparison -> paper picks
```

If no fixture source is available for a competition, create a manual CSV using the canonical fixture schema. That keeps the rest of the pipeline reproducible.
