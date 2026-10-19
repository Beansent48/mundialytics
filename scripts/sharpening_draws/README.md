# The draw deficit in the published 1X2: what it costs, and whether it is fixable

> **Status, 2026-10-19.** Where this file says "the deployed form", it means
> `p_i ** gamma` with gamma = 1.3, which is what was deployed while it was
> written. The deployed map is now the CONTRAST form — `sharpen_contrast_only`,
> v0.63.0 — and gamma stays at **1.3**, not the 1.20 this analysis recommended
> for the power form: 1.20 is that form's optimum, while the contrast form's own
> LOSO optimum is 1.268, and handing one form's constant to another is the error
> the whole investigation is about. The measurements below are unchanged and
> still correct for the form they describe. `scripts/draw_deficit/README.md` has
> the two forms side by side and the grid that decided it.

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
## Superseded, and what the two experiments jointly concluded

This file's `pair` was the wrong candidate, and its `power` @ 1.20 is the right
exponent for the wrong form. `scripts/draw_deficit` found the form; this file
found the stale exponent; the two then disagreed twice about how to pair them,
and converged. What follows replaces three rounds of correction with the joint
result. Every arm below was run against `draw_deficit`'s own `sharpen_trio`,
loaded off its branch and verified **identical to the `two_axis` in this
experiment to 0.00e+00, in both directions** -- the two experiments were always
measuring one form, and only the baseline ever differed.

### The form is settled

`p ** gamma` renormalized is exactly `(u, v) -> (gamma*u, gamma*v)` on the two
log-odds against the draw. One exponent therefore cannot correct the model's
under-confidence on the home-vs-away CONTRAST without draining the draw, which
needed nothing: the raw trio prices draws at 0.2560 against 0.2540 happening.
Scaling the contrast axis alone (`sharpen_contrast_only`) fixes that, and the
gain is robust -- against the baseline it actually displaced, `power` @ 1.3, it
is 5 of 6 or 6 of 6 across the whole range beta 1.18-1.37, worth -0.00085 to
-0.00133 of 1X2 log loss, with the draw's O/E monotonically better. `pair`,
this file's candidate, holds the draw *probability* rather than the *axis*,
which over-corrects, and manages only 3 of 6.

### The exponent is not settled, and each of us got it wrong once

| form | its own fitted optimum |
|---|---|
| `power` | gamma* = 1.2032 (LOSO 1.190-1.215, expanding 1.148-1.199) |
| contrast | beta* = 1.2682 (LOSO 1.246-1.287, expanding 1.182-1.264) |

1.20 belongs to the `power` form. Handing it to the contrast form is the same
error this investigation uncovered in the first place -- gamma 1.3 was stale
because it belonged to rho = -0.17. **Change the form, re-scan the parameter.**
This file proposed exactly that mistake, on the grounds that contrast @ 1.20
scores 5 of 6 against `power` @ 1.20 where contrast @ 1.3 scores 4 of 6.

That fold count will not bear the weight either. Swept against `power` @ 1.20,
the mean gain is flat from 1.18 to 1.37 (-0.00023 to -0.00071, optimum near
1.26) while the fold counts move two folds on 3e-05 of mean log loss:

| beta | log-loss folds | RPS folds |
|---|---|---|
| 1.18-1.20 | 5/6 | 2/6 at 1.18, 3/6 at 1.19, 5/6 at 1.20 |
| **1.21-1.26** | **6/6** | 5/6 at 1.21, **6/6 from 1.22** |
| 1.27-1.28 | 5/6 | 6/6 to 1.27, 5/6 at 1.28 |
| 1.29-1.34 | 4/6 | 5/6 to 1.31, 4/6 below |

So neither 1.20 nor 1.30 is the fold-count optimum -- and the region that is,
1.21-1.26, must NOT be picked for that reason. Selecting a parameter because
the held-out folds liked it is the cherry-pick this grid exists to expose, and
both experiments refuse it on that ground.

### What that leaves

The defensible value is the one fitted on training folds only, which is this
project's standard practice for a deployed constant: **beta = 1.268**. It lands
in the high-fold-count region without having been chosen for it.
`draw_deficit` ships 1.3, on the argument that it is within 5e-05 of 1.268 --
true of the mean, but 1.268 sits at 5/6-6/6 on log loss where 1.30 sits at
4/6. That is the mean-versus-folds conflation again, in the opposite direction
to this file's.

Either way, the form is the decision and the exponent is second-order: against
what was actually deployed, every beta in range clears the bar comfortably.

### Two caveats on reading any of these numbers

* **Absolute levels do not transfer between the two write-ups.** This file's
  `power` @ 1.3 log loss is 0.98769; `draw_deficit`'s is 0.98817, a gap of
  ~5e-04, most likely a different lambda subset or `max_goals`. The deltas
  agree to ~3e-05 and every conclusion rests on deltas and fold counts, so
  compare those and never mix absolute levels across the two files.
* **"5 of 6 on RPS throughout", in an earlier version of this section,
  overstates it.** Against `power` @ 1.20 the RPS count is 5-of-6-or-better
  only inside roughly 1.20-1.31, and falls to 2/6 at beta 1.18.

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
