# Quarantined prediction rows

`invalid_retroactive_insample_2026-08-26.csv` — the 430 rows that were the entire
contents of predictions_log.csv until 2026-08-26. They are NOT a track record and
must never be shown as one:

- **Retroactive**: every row carries `logged_at = 2026-07-23T12:23:58` for matches
  played on 2026-05-23/24 — logged two months AFTER the results were known.
- **In-sample**: they were produced by `load_club_engine()`, which fits on the full
  foundation with no date cutoff. Those 29 matches were already in its training
  data, so the model was "predicting" games it had been fitted on.

Either flaw alone invalidates them; together they make the accuracy figures
optimistically biased. Kept only for provenance.

Genuine pre-match logging started 2026-09-02 via `scripts/log_upcoming_round.py`,
which fits on played matches only and logs fixtures whose kickoff is still ahead.
