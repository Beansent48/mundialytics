# v0.34 Squad Candidate Filtering

This release tightens player-prop inference when the current input comes from broad
team rosters instead of confirmed lineups.

## Policy

- Confirmed/manual lineups are treated as high-confidence candidates.
- Squad/roster fallback candidates are ranked per match and team using identity
  status, historical minutes, expected minutes and confidence.
- Unresolved identity, zero-sample and very-low-confidence squad candidates are
  marked `not_available` in dynamic player markets.
- Low-confidence squad candidates are allowed only for basic low lines and are
  blocked from SOT and higher dynamic lines.
- Medium-low squad candidates remain available but tagged as squad fallback.

## New columns

- `candidate_rank_team`
- `candidate_policy`
- `candidate_reason`
- `candidate_score`

The goal is to keep the bot useful before official lineups are published without
presenting every roster player as equally actionable.
