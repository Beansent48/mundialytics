# v0.32.2 Player input guardrails

This patch hardens v0.32.1 after real today-matchday output showed that ESPN roster position dictionaries were leaking into CSVs. That broke role parsing and allowed unconfirmed goalkeepers/unresolved squad players to receive attacking prop rows.

Changes:

- ESPN roster/summary parsers now write compact position strings such as `G`, `F`, `M`, `D` instead of nested dict payloads.
- Player position normalization now unwraps dict-like strings and maps ESPN abbreviations to tactical groups.
- Dynamic player market rows are marked `not_available` when identity is unresolved/ambiguous or historical sample size is zero.
- Squad fallback rows remain useful for candidate discovery, but unresolved zero-sample player props are no longer treated as available markets.

This keeps match/team markets available while protecting player props until identity and sample quality are sufficient.
