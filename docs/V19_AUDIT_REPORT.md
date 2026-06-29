# v0.19 Audit Report

## Round 1 — Fixture source reality check

Reviewed API-Football free-plan failure for current World Cup fixtures. Risk confirmed: it is not a valid free source for this MVP. Added free source stack: SofaScore primary, ESPN fallback. Both are marked unofficial and raw JSON is cached.

## Round 2 — Timezone/date leakage

Reviewed the “yesterday fixtures” failure mode. Added local-date post-filtering after UTC timestamps are converted to the requested timezone. The daily command fetches a small date window for SofaScore to catch boundary games, then filters to the local calendar date.

## Round 3 — Identity and lineups

Reviewed provider fixture/lineup IDs. Added free lineup fetch for SofaScore that preserves `provider_player_id`, `provider_team_id`, provider names, lineup status and formation. It feeds the existing provider identity map instead of silently relying on names.

## Round 4 — Team/match stats logic

Reviewed team props gap. Added `team_match_stats` build/validation/prediction. The implementation refuses to invent corners when the source does not contain corner data. Team props are flagged as a baseline model, not final edge engine.

## Round 5 — Product output

Reviewed end-user usability. Added an HTML report builder with fixtures, match predictions, team props, player props, picks and warning/confidence summaries. Warnings stay visible rather than being hidden.

## Remaining risks

- SofaScore/ESPN are not contracted APIs and can change.
- Player lineup availability can be late or incomplete.
- Team prop model is baseline recent-rate, not the final calibrated model.
- Odds and EV require a reliable odds feed or manual/exported odds file.
- Real-data execution must be tested on the user's PC because network access and provider responses vary.
