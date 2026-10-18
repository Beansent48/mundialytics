# The draw deficit in the published 1X2: what it costs, and whether it is fixable

Run: `.venv/Scripts/python.exe scripts/experiment_sharpening_draws.py`
(~3 min, writes `folds.csv`). Measured 2026-10-18 on the deployed walk-forward
cache, 10,403 matches across six seasons (2020/21–2025/26).

## Why this was asked

`scripts/bivariate_goals` went looking for goal correlation, found none worth
modelling, and turned up something ten times larger on the way: the published,
sharpened 1X2 under-prices draws by 2.2 points. Since `coherent_markets=True`
(2026-09-23) pins each outcome region of the scoreline matrix to that published
trio, the deficit is pushed into every derived number on the page. This
measures the cost and tests four ways out.

## The mechanism

The deployed sharpening is `p_i ** 1.3` renormalized — which is exactly
`softmax(gamma * log p)`, temperature scaling on the 3-vector. A component
shrinks under it iff `p_i ** (g - 1) < sum_j p_j ** g`; on the 10,403 served
trios that threshold coincides with "is not the largest of the three" in
**100.00%** of matches. The draw is the largest in **2 of 10,403 (0.02%)** —
both low-scoring even games, mean lambda total 1.86 against 2.82 elsewhere — so
`gamma > 1` shrinks the draw in **10,401 matches (99.98%)**, carrying the mean
from 0.2560 to 0.2318. The defect is structural to the functional form, not a
bad parameter value: the draw sits near 0.26 while a favourite runs 0.40–0.60,
and temperature scaling always moves mass toward the leader.

| trio | draw predicted | observed | O/E | z |
|---|---|---|---|---|
| raw (`gamma` = 1) | 0.2560 | 0.2540 | 1.008 | **+0.48** |
| published (deployed `gamma` = 1.3) | 0.2318 | 0.2540 | 0.913 | **−5.20** |

The raw model is calibrated on draws. The sharpening is what breaks them.
(At `gamma` refitted per fold the deployed form lands at 1.22–1.26 and the
deviation is −0.0170 rather than −0.0222; `bivariate_goals` reports z = +5.4
using a paired standard error of 0.0041, this table uses the binomial 0.00427.)

## Finding 1 — structural: most derived markets *cannot* respond

Coherent reweighting multiplies every cell of an outcome region by
`target_r / raw_r`. For any market read off a single region, the log loss
therefore splits exactly:

```
exact score  2.908785  ==  1X2  0.988165  +  within-region  1.920620   (diff 0.0e+00)
HT/FT        1.877467  ==  1X2  0.988165  +  within          0.889301   (diff 0.0e+00)
```

The within-region term contains no trio at all, so **no change to the published
1X2 can move exact score or HT/FT except through the 1X2 itself.** Both track it
one-for-one — visible in the results below, where `ll1x2`, `exact` and `htft`
all shift by the identical +0.000123.

This is also the proof behind `bivariate_goals`'s "every arm is flat": once the
regions are pinned, the only markets with any freedom left are the ones that
**cut across** regions — BTTS and the over/under lines.

## Finding 2 — measured: there, nothing moves

Four candidate forms, each with its parameter fitted on five seasons to the same
objective the deployed `gamma` was chosen for (1X2 RPS) and scored on the sixth:

| form | 1X2 RPS | draw pred | deviation | z | worse than raw |
|---|---|---|---|---|---|
| `none` (`gamma` = 1) | 0.20134 | 0.2561 | +0.0022 | +0.51 | 0/6 |
| `power` **(deployed)** | 0.20030 | 0.2369 | **−0.0170** | **−3.99** | **6/6** |
| `pair` (draw held) | 0.20026 | 0.2561 | +0.0022 | +0.51 | 0/6 |
| `anchor` (base-rate) | 0.20048 | 0.2533 | −0.0006 | −0.13 | 2/6 |
| `pair_draw` (2 params) | 0.20023 | 0.2561 | +0.0022 | +0.51 | 0/6 |

`pair` — sharpen the home/away pair among themselves, leave the draw exactly
where the model put it — removes the entire deficit and **ties** the deployed
form on the 1X2. Against `power`, per held-out season (negative = `pair` better):

| market | mean | seasons better |
|---|---|---|
| 1X2 RPS | −0.000038 | 4/6 |
| 1X2 log loss | +0.000123 | 2/6 |
| exact score | +0.000123 | 2/6 |
| HT/FT | +0.000123 | 2/6 |
| BTTS | +0.000008 | 3/6 |
| O/U 1.5 | −0.000030 | 4/6 |
| O/U 2.5 | +0.000047 | 3/6 |
| O/U 3.5 | −0.000017 | 3/6 |

Nothing clears 5 of 6 in either direction, and every published market lands
inside 5e-05. **The 2.2-point defect is real and nearly free.**

## Finding 3 — given a free hand, the fit leaves draws alone

`pair_draw` adds a multiplier on the draw on top of `pair`, swept over
0.90–1.20. It selects **exactly 1.000 in all six folds.** Free to rescale
draws in either direction, the fit declines. The raw draw rate is right, and
the entire 1X2 gain comes from being more decisive about *which side wins* —
the draw shrinkage in the deployed form is pure side effect.

## The one market signal that does clear the bar

`anchor` — sharpening the deviation from the league base rate rather than the
probability — is the only form that beats the raw baseline on BTTS in 6 of 6
seasons (−0.00022), and it lands the draw rate within 0.0006 of observed. It
pays for that with about a fifth of the 1X2 gain (−0.00086 against −0.00104,
5/6 rather than 6/6). Noted, not pursued: 0.0002 of BTTS against 0.0002 of 1X2
RPS is not a trade this project makes on the headline market.

## Superseded — `pair` is the wrong candidate

`scripts/draw_deficit` (commit 330f54a) found the better form, and this
experiment's own harness confirms it. `p ** gamma` is the map
`(u, v) -> (gamma u, gamma v)` on the two log-odds against the draw; splitting
that into an exponent on the **home-vs-away contrast** and one on the
**draw-vs-rest** axis, and applying the deployed gamma to the contrast only
(nothing fitted), dominates `pair` on every count:

| arm | 1X2 log loss vs deployed | folds | 1X2 RPS | folds | draw O/E |
|---|---|---|---|---|---|
| `power` @ 1.3 (deployed) | — | — | — | — | 1.0951 |
| `pair` @ 1.3 (this file) | −0.00055 | 3/6 | −0.00009 | 4/6 | 0.9915 |
| two-axis (gamma on contrast) | **−0.00126** | **6/6** | **−0.00015** | **6/6** | 1.0285 |

`pair` holds the draw exactly, which **over**-corrects: it gives up the part of
the sharpening that the draw-vs-rest axis legitimately earns, which is why it
only manages 3 of 6 on log loss. Holding the *axis* rather than the probability
keeps that part and still removes most of the deficit.

Two further corrections to this file's reasoning, both from the same source:

* **The objective should have been log loss, not RPS.** The identity proved
  above says exact score and HT/FT move with the 1X2's log loss, so that is
  what a trio map is for. Fitting on RPS — as the runs here do — is nearly
  blind to a draw miscalibration, because the draw is the middle category of a
  cumulative metric. It is why this experiment read a tie where there was a
  6-of-6 gain.
* **`gamma` = 1.3 is stale.** It came from a joint `gamma`/`rho` scan that
  picked it with `rho` = −0.17; `rho` was later smoothed to −0.06 and `gamma`
  was never re-scanned. Scanned here at the current `rho`, the log-loss optimum
  is **1.20**, which matches `draw_deficit` independently.

And one caveat on that supersession, measured here: against `power` re-scanned
to its own optimum (1.20) rather than against the stale 1.3, the two-axis form
is +0.00068 of log loss ahead in only **4 of 6** folds (RPS −0.00012, 5/6). So
roughly half the headline 0.00126 is the stale `gamma`, recoverable by
re-scanning one number at 5/6, and the structural half sits just below this
project's own 5-of-6 bar. Worth separating before either is deployed.

## Caveats

* The lambda cache is from 2026-10-12 and predates `market_level_shift`
  (deployed 2026-10-17). That shift is a common multiplier on both lambdas and
  `scripts/goals_level/README.md` measures it at ≤0.0001 of RPS, so it moves
  the levels this experiment reweights, not the comparison between forms.
* Scores are capped at 10 goals per side (0 matches affected).
* `even`/odd total parity is carried in the output as a diagnostic only — the
  engine publishes no parity market. It is the one column where the deployed
  form beats `pair` clearly (−0.0005, 5/6), which is worth knowing if a parity
  market is ever added, and irrelevant until then.
* The harness reproduces the engine's own published trio to 3.33e-16 and its
  region-mass markets to 6.66e-16, so the arms differ only by the sharpening.
