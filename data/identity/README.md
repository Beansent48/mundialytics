# data/identity/

Hand-curated name fixes for players, versioned because they are reviewed by hand
rather than generated.

| File | Read by | What it does |
|---|---|---|
| `player_display_names.csv` | `mundialytics.identity.display_names`, `scripts/reconcile_fbref_duplicate_identities.py` | The name a player is shown under, when the feeds only carry a full legal name or disagree on spelling. |
| `player_aliases.csv` | `mundialytics.identity.player_resolver` | Short or accented lineup names mapped to the full historical (StatsBomb-style) name the rating data uses. |

Team-name aliases live elsewhere: `data/curated/fixture_team_aliases.csv` for
ESPN fixtures and squads, and the Understat / Sofascore / ClubElo maps in
`src/mundialytics/enrichment/` and `statistical_core/competition/club_aliases.py`.

Add a row only after checking both spellings refer to the same player. A wrong
alias silently merges two careers, which is worse than leaving a name unmatched.
