# docs/

The current write-up of the project is the top-level [README](../README.md):
results, method, what didn't work and what is in progress. Release notes are in
[CHANGELOG.md](../CHANGELOG.md), and the reasoning behind a specific component
lives next to its code, in module docstrings.

| File | What it is |
|---|---|
| [EUROPEAN_ELO_COVERAGE.md](EUROPEAN_ELO_COVERAGE.md) | Why 16% of UEFA fixtures can't be priced yet, and why fuzzy name matching is not the fix |
| [img/](img/) | Figures used by the README |
| [archive/](archive/) | Historical material, kept for the record — see below |

## archive/

Everything written while the project was being built out, from the first
versions up to v0.50 (June–early September 2026): per-version specs
(`V0xx_*.md`, `V1x_*.md`, `V2x_*.md`), the original decision log and model design
notes, data-source plans, audit reports, the long chronological README
(`README_FULL.md`) and the hand-off notes in `internal/`.

Treat it as history, not documentation. Parts are in Spanish, several describe
pipelines that were later replaced (the World Cup matchday flow, the value-betting
pick policy, Understat as the live xG source), and the numbers in them predate the
walk-forward benchmark the README reports.
