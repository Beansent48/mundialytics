# Squad value (Transfermarkt) experiment — work in progress

Experiment scripts from 2026-09-29. **The engine is untouched**: nothing here is wired into
production. Run from the repo root. Intermediate results live in
`data/external/transfermarkt/work/` (git-ignored, together with the Kaggle dump).

## What was measured

Harness: weekly-refit walk-forward (`wf_refit.py SEASON OUT.csv`, one process per season,
~10 min in parallel). Production refits daily, so this is closer to production than
`scripts/generate_deployed_walkforward.py`, which fits once a season. The harness stores the
lambda components, so any lambda-level change can be scored without refitting
(`lofo.py` rebuilds the deployed 1X2 exactly: max error 2e-16).

| Result | RPS |
|---|---|
| Deployed chain, weekly refit | 0.2006 (published frozen harness: 0.2013) |
| Re-tuning w / gamma on this harness (`retune.py`) | no gain, 0/6 |
| Promoted-team lambda handicap (`lofo.py`, `lofo2.py`) | −0.00006 (4/6) or worse; D2 stats add nothing |
| **+ squad value, full history (`tm_test.py`)** | **0.1995 (−0.0011, 5/6)** |
| + squad value, values frozen at June (`tm_test_frozen.py`) | −0.0008 (5/6) |

The squad-value shift is s = β·Δlog(top-18 value) + κ·log(λh/λa), applied as λh·e^{s/2},
λa·e^{−s/2} (totals unchanged). β ≈ 0.12–0.15 and κ ≈ −0.21 to −0.29, stable across folds.
Gap to Bet365 0.0055 → 0.0044. Largest gain: matchdays 1–5 and promoted sides. **Bundesliga
gets worse (+0.0011)**, and a per-league β has not been tried yet.

## Data

Kaggle `davidcariboo/player-scores` (CC0), **frozen: valuations end 2026-06-12, and it only
scrapes top divisions**. `tm_map.py` maps TM clubs to canonical names (all teams since 2014
map, except the 2026/27 debutants). `tm_value*.py` builds weekly squad-value snapshots.

## Where it stopped

`roster_value.py` crosses the current ESPN rosters (complete: 96 teams, ~29 players each)
with each player's latest TM value. The user's point is that a player's value doesn't change
when he moves club in the summer. Matching reaches 82% of players, with 41–59% on some
promoted sides (Paderborn, Elversberg, Le Mans, Troyes, Santander, Málaga): their D2 players
are simply not in the dump.

## Next steps

1. `tm_value_prod.py DEFAULT_EUR` (written, **not run yet**): the backtest simulates the
   production limitation. A player only counts if he had been valued at a covered top-flight
   club before the date; everyone else gets a default value. Then re-run `tm_test.py` on that
   file. If the gain survives, the roster × June-value path is production-viable.
2. Try a per-league β (the Bundesliga regression).
3. If it passes: opt-in engine flag, weekly squad-value build in `update_season`, README
   numbers, and debutant handling (the 4 unmapped 2026/27 clubs take their value from rosters
   anyway).
