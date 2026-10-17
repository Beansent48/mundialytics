# In-season goals level (2026-10-17)

Experiment scripts. **The engine is untouched**: nothing here is wired into
production. Run from the repo root.

## Why

The live 2026/27 log under-prices totals, and the drift report has enough
matches for it not to be noise:

| drift alert | n | priced | happened | z |
|---|---|---|---|---|
| Goles 2.5 over | 187 | 50.2% | 62.6% | 3.44 |
| shots 22.5 over | 182 | 71.6% | 85.2% | 4.17 |
| sot 9.5 over | 182 | 38.9% | 54.4% | 4.38 |
| jug_tiros >1.5 | 2606 | 19.6% | 23.8% | 5.90 |

The season itself moved. Big-five goals per match: 2.811 / 2.766 / 2.885 /
2.827 / 2.765 in 2021/22-2025/26, and **3.094** in 2026/27 (297 matches);
shots 24.8-26.0 against **28.27**. The deployed chain serves λ_total ≈ 2.77 —
the historical level. Bet365 is also behind but less so: on the 9-12 Oct round
it priced 58.2% over 2.5 where we priced 51.7% and 72.3% landed.

`scripts/experiment_league_level.py` (2026-09-29) tested a league-season level
term for the **event** markets and recorded that the in-season level blends
50/50 with the previous season "for events, not for goals". This measures the
goals side properly, on the λ as served.

## Is there anything to learn? (theory side, no engine)

55 league-seasons since 2014/15, first 70 matches vs the rest of that season:

| predictor of the rest of the season | corr | RMSE |
|---|---|---|
| previous season (full) | +0.636 | 0.1726 |
| first 70 of this season | +0.613 | 0.2414 |
| previous + first 70 (OLS) | — | **0.1649** |

The in-season level does add information, with coefficients 0.424 on the
previous season and 0.241 on the first 70. The best simple blend puts
**w = 0.24** on the first 70 matches — against the 0.50 that events wanted.
In the `m = (n·r + K)/(n + K)` parametrisation below, w = 0.24 at n = 70 means
**K ≈ 220**: the correction has to be shrunk about three times harder for goals
than for events.

## Harness

`wf_lambdas.py SEASON OUT.csv [--src foundation]` — `scripts/squad_value/wf_refit.py`
with two changes:

* `squad_value_asof=cutoff`, so the squad-value shift (deployed at v0.55.0,
  after that harness was written) cannot read the future on a historical fold;
* it dumps the λ **as served** — through `predict_match_or_proxy`, after the
  squad-value tilt and the debutant stand-in — rather than the raw components.

Weekly refit (production refits daily), xG form updated between refits, seven
folds: 2020/21-2025/26 from the xG-available subset the squad-value harness
used, plus 2026/27 from the foundation (its xG comes from the ESPN commentary
connector and never landed in the understat-named file).

`score_level.py` grids the correction over the dump without refitting: a common
multiplier m on both λ commutes with the squad-value and lineup tilts, whose s
depends only on log(λh/λa).

**The vectorised chain is verified twice.** `verify()` asserts it against
`deployed_markets_from_lambdas` (max error < 1e-9 on 300 sampled matches), and
it reproduces the probabilities **actually served** in the live log — the 47
big-five rows that carry both λ and p since PR #25 — to 4.97e-05, which is
exactly the log's 4-decimal rounding. (The log's European rows differ: those
fixtures are priced by the ClubElo-based European chain, not this one.)

## Forms tested

| form | m |
|---|---|
| `res_league` | running actual/predicted total-goal ratio of this league-season before this **date**, shrunk to 1 with K matches |
| `res_pooled` | the same ratio pooled over the five leagues (2026/27's drift shows in all five, so pooling cuts the noise ~5x) |
| `res_prev` | as `res_league` but shrunk toward this league's **previous-season** ratio — a persistent per-league bias rather than an in-season shift |

m is clipped to [0.90, 1.10]. K grid 20/40/70/120/200/400. Scored on 1X2 RPS
(must not get worse) and over/under 2.5 log loss (the target).

2026/27 is the fold that generated the hypothesis, so the decision is taken on
the six historical folds and 2026/27 is reported as the confirmation it was
designed to fix.

## Results — NEGATIVE. Not deployed.

Baseline, 10,700 matches: **1X2 RPS 0.19941**, O/U 2.5 log loss **0.67819**.
(The 1X2 reproduces the squad-value chain's recorded 0.1995, which is the
harness checking out against the record.)

The bias the correction is supposed to learn barely exists in history, and
2026/27 is six times any of it:

| fold | n | goals | λ_total | bias |
|---|---|---|---|---|
| 2020/21 | 1823 | 2.800 | 2.736 | 1.023 |
| 2021/22 | 1823 | 2.811 | 2.827 | 0.995 |
| 2022/23 | 1825 | 2.766 | 2.817 | 0.982 |
| 2023/24 | 1747 | 2.883 | 2.842 | 1.015 |
| 2024/25 | 1444 | 2.760 | 2.756 | 1.002 |
| 2025/26 | 1741 | 2.768 | 2.716 | 1.019 |
| **2026/27** | 297 | **3.094** | 2.740 | **1.129** |

Historically the ratio wobbles ±2% in both directions and does not persist
inside a season, so a correction that reacts to it is buying noise. Every form
at every K is worse or flat:

| form | K | Δ 1X2 RPS | folds | Δ O/U | folds |
|---|---|---|---|---|---|
| res_league | 70 | +0.00004 | 2/7 | +0.00057 | 2/7 |
| res_league | 400 | +0.00001 | 3/7 | −0.00003 | 2/7 |
| res_pooled | 200 | +0.00003 | 2/7 | −0.00005 | 2/7 |
| res_pooled | 400 | +0.00002 | 3/7 | **−0.00008** | 2/7 |
| res_prev | 400 | +0.00006 | 0/7 | +0.00027 | 2/7 |

The best cell (res_pooled, K=400) is entirely one fold: **−0.00743 on 2026/27**
and +0.00013 worse across the six historical ones, with 1/6 improving. The bar
in this repo is 5/6.

A post-hoc gate (`--gated`: fire only when the pooled drift exceeds g after
n_min matches) does not rescue it. At n_min=100, g=0.08, K=200 it fires in 4.2%
of matches, never fires at all in 5 of the 6 historical folds, costs +0.00013
in the one where it does, and gains −0.01007 on 2026/27. **No historical fold
is ever helped** — the gate buys harmlessness, not evidence.

## What is worth keeping from this

* **The 1X2 is not at risk either way.** A common level multiplier moves RPS by
  ≤0.0001 in every cell. This is a totals-only question.
* **2026/27's shift is real, broad and not regressing.** +0.354 goals per match
  over λ (±0.102, t=+3.46); all five leagues between +7.6% and +19.1%; +0.326
  (t=+3.00) with Bundesliga removed; and by thirds of the season 1.090 → 1.129
  → 1.210, so it is not a hot August decaying away.
* **A correction would have been worth −0.0100 of O/U log loss this season**,
  about a quarter of the 0.037 gap to Bet365's closing price. The information is
  there; what is missing is any way to tell, in real time, a real level shift
  from the ±2% noise of a normal season — which is exactly why every K fast
  enough to catch this season loses money in the other six.

So the honest position is that this is a one-season judgment call, not a
validated model change, and it stays out of the deployed chain.

