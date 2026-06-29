# Player Lineup Strength Rating Plan

## Position

Player Elo is possible and useful, but it should not be implemented before the ingestion layer has enough lineups, minutes and player match stats.

The first implementation should be called `player_lineup_strength_rating`, not "definitive player Elo".

## Goal

Estimate how much a team's expected strength changes when the starting XI changes.

## Inputs Required

- `canonical_lineups.csv`
- `canonical_player_match_stats.csv`
- minutes played
- started/bench flag
- position/role
- team and opponent
- team Elo / ClubElo context
- match result and xG result when available
- player xG/xA/xGChain/xGBuildup when available

## v1 Features

For each fixture:

```text
home_lineup_rating
away_lineup_rating
home_attack_lineup_rating
away_attack_lineup_rating
home_defense_lineup_rating
away_defense_lineup_rating
home_keeper_rating
away_keeper_rating
home_missing_core_player_value
away_missing_core_player_value
lineup_confidence
```

## Rating Design

Use separate components:

```text
overall_rating
attack_rating
defense_rating
possession_rating
keeper_rating
uncertainty
```

Update using:

```text
rating_delta =
    K
    * minutes_weight
    * uncertainty_weight
    * context_adjusted_surprise
```

Where `context_adjusted_surprise` should depend on expected vs actual result and expected vs actual xG differential, adjusted by opponent strength and home/neutral context.

## Why not naive Elo?

A naive rule such as "all starters gain points when the team wins" is too noisy for football. It ignores low scoring variance, teammates, opponents, substitutions, red cards and player roles.

## Future v2

Once lineups and substitutions are strong enough, move from Elo-like ratings to regularized adjusted plus-minus using stints:

```text
segment_start_minute
segment_end_minute
players_on_pitch
score_state
red_cards
xG_for_segment
xG_against_segment
```

## Runtime Modes

- `pre_lineup_mode`: probable XI, injuries/suspensions, recent minutes.
- `confirmed_lineup_mode`: confirmed lineups, no substitutions/events from the match.
- `post_match_analysis_mode`: actual lineups and events for explanation/training only.

## Acceptance Criteria Before Implementation

- lineups coverage by competition/season is audited.
- player match stats coverage by competition/season is audited.
- player identity registry exists.
- minutes and position coverage are usable.
- neutral venue context is working.
