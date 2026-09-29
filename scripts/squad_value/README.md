# Squad value (Transfermarkt) experiment

> **Update (v0.55.0).** Production now reads every Big Five squad from
> transfermarkt.com once a week (`src/mundialytics/providers/transfermarkt.py`).
> The history is rebuilt with each snapshot's own valuations instead of June's
> (`build_history(freeze_june=False)`). Fresh values measured −0.00109 against
> −0.00079 June-frozen (5/6 both), and the constants were refitted on the fresh
> history: β 0.1337, κ −0.2617. The first live read mapped 96/96 teams with
> 98.8% of players valued. The June-frozen results below remain the record of
> the first deployment.

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
| **+ squad value, full history (`tm_eval.py`)** | **0.1995 (−0.0011, 5/6)** |
| + squad value, values frozen at June (`tm_eval_frozen.py`) | −0.0008 (5/6) |

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

## Production-faithful test (2026-09-29, second session)

`tm_value_prod.py DEFAULT_EUR MATCH_LOSS` simulates what production can see. A player counts
only if, before the June cutoff, he was at a club **playing a covered top-flight league that
season**. That is point-in-time from `games.csv`, because `player_club_domestic_competition_id`
on the valuations is the club's CURRENT league (Leif Davis shows "GB1" at Ipswich in
League One). Everyone else gets DEFAULT_EUR. Visible share is 81.7% (production matching:
82%). Promoted sides average 71%, with lows of 13–36% (Clermont, Cádiz, Spezia), as harsh as
2026/27.

`tm_eval_prod.py` (value+stretch, LOSO):

| Value file | pooled dRPS | seasons better | promoted | rest |
|---|---|---|---|---|
| frozen June, full visibility | −0.00082 | 5/6 | −0.00124 | −0.00066 |
| invisible = 0.5 M€ | −0.00077 | 5/6 | −0.00097 | −0.00069 |
| invisible = 1 M€ | −0.00079 | 5/6 | −0.00105 | −0.00069 |
| invisible = 2 M€ | −0.00083 | 5/6 | −0.00125 | −0.00068 |
| 1 M€ + 10% of visible lost at random | −0.00075 | 5/6 | −0.00112 | −0.00061 |

O/U 2.5 is unchanged (the shift keeps totals). **The roster × June-value path is viable.**

Bundesliga (`tm_eval_league.py`): in every fold it fits β≈0 and κ≈0, and the global shift
makes it worse (+0.0014 to +0.0017). The data is fine: corr(Δvalue, goal residual) is +0.10
there vs +0.11 to +0.15 elsewhere. Per-league params, shrunk params, and skipping the
Bundesliga all reach about −0.0010 pooled but only **4/6** seasons, so they fail the bar.
Keep the global version.

## Next steps

1. ~~Production-faithful backtest~~ passed (see above).
2. ~~Per-league β~~ fails the bar; keep the global version.
3. Implement it opt-in. A squad-value builder turns ESPN rosters × TM values
   (`roster_value.py`, 82% matched, unmatched players at 1–2 M€) into a file refreshed in
   `update_season`. The engine gets a `squad_value_shift` parameter (default None =
   byte-identical) with β/κ fitted on all six seasons. Validate end-to-end on the
   weekly-refit harness before switching it on, then update the README numbers.
