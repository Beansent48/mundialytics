# v0.15 Competition taxonomy and domain labels

This release fixes the previous `team_scope` ambiguity. The project no longer treats “national” as “domestic league”. `team_scope` remains only for backward compatibility:

- `team_scope = club` means the teams are clubs.
- `team_scope = national` means the teams are national teams/selections.

The new objective labels are:

- `team_type`: `club`, `national_team`, `unknown`
- `competition_context`: `domestic_league`, `domestic_cup`, `continental_club`, `continental_club_qualifier`, `international_national_tournament`, `qualifier`, `friendly`, `unknown`
- `gender`: `men`, `women`, `unknown`

No subjective `match_importance` feature is used.

## Examples

| Competition | team_type | team_scope | competition_context | gender |
|---|---|---|---|---|
| La Liga | club | club | domestic_league | men |
| Liga F | club | club | domestic_league | women |
| Champions League | club | club | continental_club | men |
| UEFA Europa League | club | club | continental_club | men |
| Copa del Rey | club | club | domestic_cup | men |
| FIFA World Cup | national_team | national | international_national_tournament | men |
| UEFA Euro | national_team | national | international_national_tournament | men |
| Copa America | national_team | national | international_national_tournament | men |
| African Cup of Nations | national_team | national | international_national_tournament | men |
| UEFA Euro qualification | national_team | national | qualifier | men |
| Friendly | national_team | national | friendly | men |

## Operational policy

The labels are used to:

1. Audit that competitions are not mislabeled.
2. Preserve metadata in validation/calibration outputs.
3. Compare metrics by domain.
4. Provide a light fallback context for player-prop priors when a player has too little direct history.

They are not intended to dominate the model. Team strength, player history, expected minutes, position, lineup eligibility, and ELO/Poisson context remain the main drivers.

## Hard rules

- Historical players can be used for training.
- Current predictions can only include players present in the supplied current lineup/squad file.
- Expected minutes are allowed and required, but must be pre-match estimates.
- Observed test-match minutes must never be used as expected minutes in a pre-match backtest.
- `La Liga`, `Premier League`, `Liga F`, `Champions League`, etc. must never be labeled as `national_team`.
- `FIFA World Cup`, `UEFA Euro`, `Copa America`, etc. must never be labeled as `club`.
