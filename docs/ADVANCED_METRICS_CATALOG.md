# Advanced Metrics Catalog — v0.50.1

This catalog defines useful non-market football data for the simulator, statistical consultant and future predictive models. Odds/market features are intentionally excluded from this version.

## Venue and Context

| Metric | Use | Leakage policy |
|---|---|---|
| `neutral` | Boolean neutral venue flag. Removes scheduled home advantage. | Pre-match safe if known before kickoff. |
| `venue_country`, `venue_city` | Determines whether neutral venue still favors one team. | Pre-match safe. |
| `home_team_country`, `away_team_country` | National/team-country mapping for neutral tournaments. | Pre-match safe. |
| `home_team_city`, `away_team_city` | Supports cases like finals hosted in a club's city. | Pre-match safe. |
| `home_advantage_factor_pre` | `1 = home`, `0 = none`, `-1 = away-listed side has venue edge`. | Derived pre-match. |

## Chance Quality

| Metric | Why it matters |
|---|---|
| `xG`, `npxG` | Better signal than goals for chance quality. |
| `xA` | Chance creation quality independent of finishing. |
| `npxG + xA` | Compact attacking involvement/quality proxy. |
| `xG per shot` | Shot selection / chance quality. |
| `goals - xG` | Finishing over/underperformance; needs shrinkage. |

## Shot Volume and Location

| Metric | Why it matters |
|---|---|
| `shots`, `shots_on_target` | Attacking volume and pressure. |
| `shots_inside_box` | Higher-danger shot volume. |
| `shots_outside_box` | Lower-quality volume; useful for style and simulator narrative. |
| `avg_shot_distance` | Shot profile and chance quality. |
| `big_chances` | Strong explanatory metric when available. |

## Shot Body Part

| Metric | Why it matters |
|---|---|
| `header_shots`, `header_xg` | Aerial/cross/set-piece profile. |
| `left_foot_shots`, `left_foot_xg` | Footedness profile, simulator and player tendencies. |
| `right_foot_shots`, `right_foot_xg` | Footedness profile, simulator and player tendencies. |

## Shot Situation

| Metric | Why it matters |
|---|---|
| `open_play_xg` | Sustainable attacking creation. |
| `penalty_xg` | Should be separated because penalties distort team attack signal. |
| `set_piece_xg` | Valuable for corners/free-kicks and team style. |
| `corner_xg` | Set-piece-specific threat. |
| `free_kick_xg` | Direct/free-kick threat. |
| `counterattack_xg` | Transition strength. |

## Possession and Territory

Field tilt appears once in the contract. It should not be duplicated as both possession and territory input; reports can reference the same column in different narratives without copying the data.

| Metric | Why it matters |
|---|---|
| `possession` | Control, but weak alone. |
| `field_tilt` | Share of territorial possession in attacking areas. |
| `PPDA` | Pressing intensity proxy. |
| `touches_attacking_third` | Territory pressure. |
| `touches_box` | Box presence, strong attacking signal. |
| `final_third_entries` | Territory progression. |
| `deep_completions` | Final-third penetration. |

## Progression

| Metric | Why it matters |
|---|---|
| `progressive_passes` | Passing progression. |
| `progressive_carries` | Carry progression. |
| `passes_into_final_third` | Territory gain. |
| `passes_into_penalty_area` | Dangerous progression. |
| `crosses_into_penalty_area` | Wide delivery profile. |
| `through_balls` | Chance creation style. |

## Defensive and Goalkeeping

| Metric | Why it matters |
|---|---|
| `xG_against`, `npxG_against` | Defensive chance suppression. |
| `shots_against`, `sot_against` | Defensive volume allowed. |
| `xG per shot allowed` | Defensive shot quality allowed. |
| `tackles`, `interceptions`, `blocks`, `clearances` | Defensive action profile. |
| `pressures`, `ball_recoveries` | Pressing/recovery style. |
| `keeper_saves`, `keeper_psxG`, `goals prevented` | Goalkeeper shot-stopping. |

## Discipline and Set Pieces

| Metric | Why it matters |
|---|---|
| `fouls`, `yellow_cards`, `red_cards` | Cards, stoppages and game-state volatility. |
| `corners` | Set-piece pressure and corner markets. |
| `set_piece_xG` | Better than corners alone when available. |

## Player / Lineup Inputs

| Metric | Why it matters |
|---|---|
| `minutes`, `started`, `position` | Required for player lineup strength. |
| `player_xG`, `player_xA`, `xGChain`, `xGBuildup` | Player contribution beyond goals/assists. |
| `shot body part`, `shot location` | Simulator narratives and player tendencies. |
| `lineups` | Required for confirmed/probable XI adjustments. |

## Leakage Rule

For a match at date `D`, model features can only use information known before kickoff. Current-match `home_xg`, `home_shots`, `home_lineups_actual`, shot events and substitutions are observations/targets unless the runtime mode explicitly represents confirmed-lineup prediction shortly before kickoff.
