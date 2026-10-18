# The draw mass the 1X2 sharpening takes out of the grid (2026-10-18)

Experiment scripts. The candidate below **was deployed in v0.63.0** as
`sharpen_contrast_only` (default False; the deployed config turns it on); every
other form here stays research-only. Run from the repo root.

```
python scripts/draw_deficit/score_trio_forms.py all      # ~6 min
python scripts/draw_deficit/score_trio_forms.py gap      # just the measurement
```

## Why

`scripts/bivariate_goals/README.md` went looking for a goals correlation, found
there is none worth modelling, and turned up this on the way — an effect ten
times the size of the one it was chasing.

The real draw rate over the six walk-forward seasons is 0.2540. The scoreline
matrix's **raw** trio says 0.2560: calibrated. The **published** trio, after
`sharpen_gamma_1x2 = 1.3`, says **0.2318**.

| | draw rate |
|---|---|
| observed, the six seasons scored | 0.2540 |
| observed, 27 seasons / 46,138 matches | **0.2587** |
| raw matrix trio | 0.2560 (O/E 0.992) |
| **published trio** | **0.2318 (O/E 1.0957)** |

The gap is +0.0222 with a standard error of 0.0041 — **z = +5.4 — and positive
in 6 of 6 seasons** (+0.0165, +0.0287, +0.0160, +0.0379, +0.0161, +0.0170). It
is not a draw-heavy sample either: the long-run rate is *higher* still, so the
published trio is 0.0269 below it. The mass goes into both tails, near-equally:
home is over-priced by 0.0109 (z −2.2) and away by 0.0113 (z −2.5).

That used to be a 1X2-only matter. Since `coherent_markets` (2026-09-23) every
outcome region of the scoreline matrix is reweighted onto the published trio, so
the missing draw mass is now subtracted from every exact score, BTTS and HT/FT
number the page shows.

**Why it might be stale rather than wrong.** The sharpening exists for a good
reason — the raw engine is under-confident, "says 64% when reality is 71%" — and
it pays for itself on its own market: it takes 1X2 RPS from 0.20143 to 0.20045.
But `prediction_engine.py` records that γ = 1.3 came from a **joint γ/ρ scan
that picked γ = 1.3 together with ρ = −0.17**. ρ was later smoothed to −0.06
(the "always 1-1" fix) and γ was never re-scanned. Section 6 below shows that
scan was self-consistent: at ρ = −0.17 the best γ *is* 1.26, and at ρ = −0.06 it
is 1.20. Moving ρ should have moved γ, and did not.

And underneath that, one exponent is being asked to do two jobs. Writing
u = log(p_h/p_d), v = log(p_a/p_d), the map p^γ is exactly
(u, v) → (γu, γv) — so correcting under-confidence on the home-vs-away
contrast necessarily drains the draw, whether the draw needed it or not. It
did not: O/E 0.992.

## The structure that decides the whole question

With the raw matrix M fixed, coherence makes two of the four matrix markets
*identically* the 1X2 log loss plus a constant:

```
log P(exact score) = log M[cell] − log M[region] + log t[region]
log P(HT/FT path)  = log path    − log ft[res]   + log t[res]
```

Only the last term depends on the trio map `t`. Measured (`identity`):

| trio map | 1X2 ll | exact score ll | cs − 1X2 | HT/FT ll | htft − 1X2 |
|---|---|---|---|---|---|
| γ = 1.0 (raw) | 0.99041 | 2.91103 | 1.92062 | 1.87971 | 0.88930 |
| γ = 1.3 (deployed) | 0.98817 | 2.90878 | 1.92062 | 1.87747 | 0.88930 |
| γ = 1.6 | 0.99750 | 2.91812 | 1.92062 | 1.88680 | 0.88930 |
| two_axis 1.0 / 1.3 | 0.98688 | 2.90750 | 1.92062 | 1.87619 | 0.88930 |
| vector 1.3 / +.2 / −.2 | 1.00046 | 2.92108 | 1.92062 | 1.88977 | 0.88930 |

Constant to **7.8e-16**. Three consequences, and they shape everything below:

* the objective for the trio map is the trio's own **log loss**, not RPS (RPS is
  reported as the guard it is);
* whatever helps the 1X2 helps exact score and HT/FT by *exactly* as much, so
  there is no trade-off to balance between them;
* only over/under and BTTS can disagree, because they are the markets built from
  unions of cells that cross outcome regions — and that is why they barely move
  in any arm.

## The forms

| form | parameters | what it is |
|---|---|---|
| `gamma` | γ | the deployed family, re-scanned |
| `two_axis` | α, β | α on the draw-vs-rest log-odds, β on the home-vs-away contrast. α = β recovers `gamma`; **α = 1 sharpens the contrast and leaves the draw axis alone** |
| `vector` | T, b_h, b_a | textbook vector scaling: T on log p plus a free intercept per outcome. Per-class aggregate calibration comes out for free |
| `decouple` | γ_m | publish the deployed trio for the 1X2, reweight the **matrix** to a trio of its own. Breaks the coherence invariant on purpose |

Fitted by 1X2 log loss (per the identity) on earlier seasons only
(`expanding`, five scorable folds), with leave-one-season-out and a
per-test-season `oracle` beside it as ceilings. In sample, on all six seasons at
once, each family's own best:

| | RPS | 1X2 ll | o25 ll | btts ll | P(draw) | O/E draw |
|---|---|---|---|---|---|---|
| γ = 1.00 (raw) | 0.20143 | 0.99041 | 0.67906 | 0.68445 | 0.2560 | 0.992 |
| γ = 1.20 | 0.20041 | 0.98754 | 0.67893 | 0.68438 | 0.2397 | 1.059 |
| **γ = 1.30 (deployed)** | **0.20045** | **0.98817** | **0.67896** | **0.68440** | **0.2318** | **1.096** |
| `gamma` fitted → 1.203 | 0.20040 | 0.98754 | 0.67893 | 0.68438 | 0.2395 | 1.060 |
| `two_axis` fitted → 0.899 / 1.280 | 0.20028 | 0.98676 | 0.67892 | 0.68434 | 0.2527 | 1.005 |
| `vector` fitted → 1.265 / −.099 / −.116 | 0.20030 | 0.98673 | 0.67901 | 0.68426 | 0.2540 | 1.000 |

γ = 1.3 is past the one-parameter optimum in both directions — the log-loss
optimum is 1.20, the RPS optimum 1.25 — but **no single γ fixes the deficit**:
at its own optimum the draw is still 6% short. A one-parameter family cannot,
because it only has one knob for two jobs. Both two-parameter-or-more families
land the draw within half a point of calibration *and* score better.

## Results — the candidate validates, and `decouple` is dead

Deltas against the deployed γ = 1.3, coherence kept, over six walk-forward
seasons and 10,403 matches. Negative is better; the bar is 5 of 6 folds.

| arm | 1X2 RPS | 1X2 ll | O/U 2.5 | BTTS | exact score | HT/FT |
|---|---|---|---|---|---|---|
| **`two_axis` FIXED (1.0, deployed γ)** | **−0.00015 (6/6)** | **−0.00128 (6/6)** | −0.00006 (3/6) | −0.00006 (4/6) | **−0.00128 (6/6)** | **−0.00128 (6/6)** |
| `two_axis` [loso] | −0.00013 (5/6) | −0.00120 (6/6) | −0.00006 (3/6) | −0.00006 (4/6) | −0.00120 (6/6) | −0.00120 (6/6) |
| `two_axis` [expanding] | −0.00007 (3/5) | −0.00116 (4/5) | −0.00014 (3/5) | +0.00007 (3/5) | −0.00116 (4/5) | −0.00116 (4/5) |
| `two_axis` [oracle] | −0.00022 (6/6) | −0.00185 (6/6) | +0.00004 (3/6) | −0.00008 (4/6) | −0.00185 (6/6) | −0.00185 (6/6) |
| `gamma` [loso] | −0.00003 (4/6) | −0.00056 (5/6) | −0.00003 (3/6) | −0.00002 (4/6) | −0.00056 (5/6) | −0.00056 (5/6) |
| `gamma` [expanding] | +0.00005 (3/5) | −0.00038 (3/5) | −0.00011 (3/5) | +0.00006 (3/5) | −0.00038 (3/5) | −0.00038 (3/5) |
| `vector` [loso] | +0.00006 (4/6) | −0.00064 (5/6) | +0.00006 (3/6) | −0.00014 (5/6) | −0.00064 (5/6) | −0.00064 (5/6) |
| `vector` [expanding] | +0.00045 (1/5) | +0.00044 (3/5) | −0.00007 (3/5) | +0.00003 (3/5) | +0.00044 (3/5) | +0.00044 (3/5) |

Baseline on all 10,403: 1X2 RPS 0.20045, 1X2 ll 0.98817, O/U 2.5 0.67896,
BTTS 0.68440, exact score 2.90878, HT/FT 1.87747.

### The candidate: apply the deployed exponent to one axis instead of two

`two_axis` at **α = 1, β = γ_deployed = 1.3** — "keep the sharpening, stop
applying it to the draw axis". **It introduces no new constant**: β is the γ
already deployed, and α = 1 is not fitted, it is read off section 1 (the raw
trio's draw rate is already calibrated at O/E 0.992, so the draw axis needs no
correction and the contrast does). Nothing is estimated from any fold, so all
six folds are out of sample:

| fold | n | 1X2 RPS | 1X2 ll | O/U 2.5 | BTTS | exact score | HT/FT | O/E draw | was |
|---|---|---|---|---|---|---|---|---|---|
| 2020/21 | 1823 | −0.00018 | −0.00105 | +0.00028 | −0.00038 | −0.00105 | −0.00105 | 1.0119 | 1.0692 |
| 2021/22 | 1823 | −0.00019 | −0.00142 | −0.00022 | +0.00002 | −0.00142 | −0.00142 | 1.0549 | 1.1248 |
| 2022/23 | 1825 | −0.00019 | −0.00103 | −0.00064 | +0.00026 | −0.00103 | −0.00103 | 0.9999 | 1.0707 |
| 2023/24 | 1747 | −0.00018 | −0.00264 | +0.00017 | −0.00013 | −0.00264 | −0.00264 | 1.0908 | 1.1677 |
| 2024/25 | 1444 | −0.00002 | −0.00049 | −0.00020 | −0.00009 | −0.00049 | −0.00049 | 1.0053 | 1.0692 |
| 2025/26 | 1741 | −0.00012 | −0.00093 | +0.00025 | −0.00008 | −0.00093 | −0.00093 | 1.0101 | 1.0719 |
| **all** | 10403 | **−0.00015** | **−0.00128** | −0.00006 | −0.00006 | **−0.00128** | **−0.00128** | **1.0291** | 1.0957 |
| **folds better** | | **6/6** | **6/6** | 3/6 | 4/6 | **6/6** | **6/6** | | |

Every market the trio map controls improves in **every** fold, the 1X2 RPS
guard included, and two thirds of the draw deficit closes (O/E 1.096 → 1.029,
and inside half a point in four of the six folds). The two markets the map does
*not* control — over/under and BTTS, the ones that cross outcome regions — are
flat: −0.00006 each, 3 of 6 and 4 of 6, which is the coin flip you should expect
from a change whose mechanism barely reaches them.

For scale, exact score −0.00128 is about **40% of what the coherence
deployment itself was worth** (−0.0032), and three times the largest effect
anywhere in the bivariate-Poisson experiment.

**It is not just a smaller γ in disguise.** The candidate's mean draw
probability, 0.2468, sits between γ = 1.1 (0.2478) and γ = 1.2 (0.2397) — but
its log loss, 0.98688, beats both (0.98827 and 0.98754) and beats the best γ
anywhere in the family (0.98754 at 1.20). It gives the draw back *while keeping
the full contrast sharpening*, which is exactly what no single exponent can do:
γ = 1.1 would have to give up most of the home/away correction to get the same
draw mass. That is the difference between two knobs and one. **On the
in-sample mean; by season it is 4 of 6 against the best γ at β = 1.3 and 5 of 6
at β = 1.20 — see the correction below.**

**Fitting α costs more than it buys.** `two_axis [loso]` fits α ∈ [0.870, 0.970]
and β ∈ [1.250, 1.301] — tight, and bracketing the fixed point — yet scores
−0.00120 against the fixed −0.00128, and drops the RPS guard from 6/6 to 5/6.
The estimation noise is worth more than the bias it removes. `vector`, with
three parameters, is the clear warning: it has the best in-sample log loss of
any family (0.98673) and under the strict expanding window it is **worse than
the deployed chain** (+0.00044, 1 of 5 on RPS). Two parameters is where this
family stops paying, and one of the two should not be fitted.

### Correction: half of that headline was the stale γ, not the second axis

**Everything above is scored against γ = 1.3, and γ = 1.3 is the thing this
same archaeology showed to be stale.** A parallel session reproduced this
experiment independently, re-scanned γ at the ρ actually served and deployed
**γ = 1.20** on master (`52d94c7`), crediting `scripts/bivariate_goals` for the
deficit. Their harness matches this one to the fourth decimal, their
parameterisation is bit-identical to `f_two_axis` (checked: max difference
0.0e+00), and they were right to ask for a re-measurement. So, against
power(1.20) instead of power(1.3) — `score_trio_forms.py rebase`:

| arm | 1X2 RPS | 1X2 ll | O/U 2.5 | BTTS | exact score | HT/FT | O/E draw |
|---|---|---|---|---|---|---|---|
| power 1.30 (the old baseline) | +0.00004 (2/6) | +0.00062 (1/6) | +0.00003 (3/6) | +0.00002 (2/6) | +0.00062 (1/6) | +0.00062 (1/6) | 1.0957 |
| **two_axis (1.0, 1.20)** | **−0.00002 (5/6)** | **−0.00048 (5/6)** | +0.00000 (3/6) | −0.00002 (4/6) | **−0.00048 (5/6)** | **−0.00048 (5/6)** | **1.0160** |
| two_axis (1.0, 1.25) | −0.00010 (6/6) | −0.00069 (6/6) | −0.00002 (3/6) | −0.00003 (5/6) | −0.00069 (6/6) | −0.00069 (6/6) | 1.0224 |
| two_axis (1.0, 1.30) | −0.00011 (5/6) | −0.00066 (4/6) | −0.00003 (3/6) | −0.00004 (4/6) | −0.00066 (4/6) | −0.00066 (4/6) | 1.0291 |
| two_axis [expanding] | −0.00005 (5/5) | −0.00069 (5/5) | −0.00006 (3/5) | +0.00004 (3/5) | −0.00069 (5/5) | −0.00069 (5/5) | 1.0086 |
| two_axis [loso] | −0.00010 (5/6) | −0.00058 (4/6) | −0.00003 (3/6) | −0.00004 (5/6) | −0.00058 (4/6) | −0.00058 (4/6) | 1.0055 |

Baseline power(1.20): 1X2 RPS 0.20041, 1X2 ll 0.98754, O/U 2.5 0.67893,
BTTS 0.68438, exact score 2.90816, HT/FT 1.87685.

What this changes, and what it does not:

* **The −0.00128 headline double-counts the γ re-scan.** About 46% of it is
  γ 1.3 → 1.20, which master now does on its own. The second axis is worth
  roughly **−0.0005 on top of it**, not −0.0013.
* **The 4-of-6 is about the exponent, not the form** — but the fix is NOT to
  pair the contrast map with master's 1.20. **The two forms want different
  exponents**, fitted on 1X2 log loss on past seasons only:

  | window | power γ* | contrast β* |
  |---|---|---|
  | LOSO, six folds | 1.190 – 1.215 | **1.246 – 1.287** |
  | expanding, five folds | 1.148 – 1.199 | 1.182 – 1.264 |
  | all six | 1.2032 | **1.2682** |

  1.20 is the optimum **of the power form**. Handing it to the contrast form
  repeats exactly the mistake this whole investigation uncovered — γ = 1.3 was
  stale because it belonged to ρ = −0.17. **Change the form, re-scan the
  parameter.**
* **So merging master's γ into this branch would make it worse, not better.**
  `two_axis (1.0, 1.20)` is −0.00048 against power(1.20) where
  `two_axis (1.0, 1.30)` is −0.00066 and the form's own optimum
  `two_axis (1.0, 1.27)` is −0.00071: the naive merge is the weakest of the
  three. The deployed 1.3 is within 5e-05 of this form's own LOSO optimum, i.e.
  already right for it. Raised by the parallel session, which reversed its own
  verdict on seeing it.
* **What is robust, and what is not.** Swept 1.18 → 1.37 against power(1.20),
  the contrast form wins on the mean *everywhere*, −0.00023 to −0.00071, with a
  flat optimum around 1.26. The **fold count is the fragile part**: 5/6 at
  1.18–1.20, 6/6 at 1.21–1.26, 5/6 at 1.27–1.28, 4/6 at 1.29–1.34, 3/6 above —
  so 3e-05 of mean log loss moves two folds. Against power(1.30), the baseline
  this actually displaced, it is 5/6 or 6/6 across that entire range and the
  mean gain is −0.00085 to −0.00133. The honest summary is: **the mean gain and
  the draw calibration are solid, the fold count against a re-scanned power
  baseline is knife-edge, and "6 of 6" is a stronger claim than this form can
  carry against a corrected baseline.**
* β = 1.25–1.27 is better than 1.30 on both counts and is **not proposed here**:
  picking it off this grid is the cherry-pick the grid exists to expose. The
  LOSO fit lands at 1.268 and is the defensible value if anyone re-scans.
* **The claim "beats the best γ anywhere in the family" needs its fold count.**
  It is true on the in-sample mean (0.98688 against 0.98754) and it is 4 of 6
  by season at β = 1.3, 5 of 6 at β = 1.20. The original write-up gave the mean
  and not the folds.
* Nothing changes in the diagnosis, the identity, the `decouple` result or the
  negative arms: those never depended on the baseline γ.

### `decouple` is strictly dominated

The route that looked obvious — publish the sharpened trio but let the matrix
keep a better-calibrated one — is the one the identity kills:

| arm | O/U 2.5 | BTTS | exact score | HT/FT | incoherence reintroduced |
|---|---|---|---|---|---|
| matrix γ = 1.0 (the raw trio) | +0.00011 (3/6) | +0.00004 (2/6) | **+0.00224 (1/6)** | +0.00224 (1/6) | mean 0.0457, max 0.0876 |
| matrix γ = 1.2 | −0.00003 (3/6) | −0.00002 (4/6) | −0.00062 (5/6) | −0.00062 (5/6) | mean 0.0148, max 0.0268 |
| matrix γ fitted [expanding] | −0.00011 (3/5) | +0.00006 (3/5) | −0.00038 (3/5) | −0.00038 (3/5) | mean 0.0195, max 0.0416 |
| deployed (coherent) | — | — | — | — | max 8.9e-16 |

Giving the grid back the **raw** trio makes the exact score 0.0022 *worse*, in 5
of 6 folds. The draw deficit is not the matrix markets paying for a bad trio:
the sharpened trio is the better probabilistic forecast, aggregate draw rate and
all, and the matrix markets want whichever trio has the best log loss. And the
one decoupling that does help — matrix γ = 1.2 — buys exactly what simply
re-scanning γ buys coherently (−0.0006), while reintroducing a **mean 1.5-point,
max 2.7-point** disagreement between the published 1X2 and its own scoreline
grid. That disagreement was the bug coherence was deployed to fix, and the user
saw it on the web. There is no version of decoupling worth paying for it.

Coherence is intact in every other arm: worst disagreement **8.9e-16** across
the matrix and the nine HT/FT paths.

### ρ is a substitute for the draw axis, but leave it alone

Exact-score log loss of the coherent matrix (the full-information likelihood),
each cell with its own trio map fitted in sample:

| ρ | `gamma` | `two_axis` | `vector` |
|---|---|---|---|
| −0.17 | 2.91164 @ 1.26 | 2.91153 @ 1.45 / 1.24 | 2.91125 |
| −0.12 | 2.90892 @ 1.24 | 2.90889 @ 1.16 / 1.25 | 2.90889 |
| **−0.06 (deployed)** | 2.90816 @ 1.20 | **2.90738 @ 0.90 / 1.28** | 2.90735 |
| −0.03 | 2.90871 @ 1.18 | 2.90714 @ 0.80 / 1.29 | 2.90705 |
| 0.00 | 2.90982 @ 1.15 | 2.90720 @ 0.72 / 1.31 | 2.90704 |

Two things. **ρ = −0.06 is right for the family actually deployed** — the
`gamma` column has its minimum there. And once a second axis can handle the
draw, ρ wants to be weaker (−0.03), because Dixon-Coles' low-cell draw boost and
the sharpening's draw axis were doing overlapping work. That is the same
substitution `scripts/bivariate_goals/README.md` saw from the other side (ρ = 0
scoring better *through* the reweighting). It is worth 0.00024 in sample, ρ = 0
costs BTTS in the earlier experiment, and it is a second change in the same
place — so **not proposed**. If the candidate is deployed, re-scan ρ afterwards,
on its own, through the reweighting.

## Recommendation

**Deploy the candidate: apply `sharpen_gamma_1x2` to the home-vs-away contrast
only, not to the draw axis — keeping the exponent at 1.3, which is within
5e-05 of this form's own LOSO optimum of 1.268. Do NOT pair it with the power
form's 1.20.** The numbers that carry it:

* against the γ that was deployed when this was measured: 6 of 6 folds on 1X2
  log loss, exact score and HT/FT (−0.00128 each) and 6 of 6 on the 1X2 RPS
  guard (−0.00015), out of sample, with nothing fitted;
* against power(1.20), the honest baseline and what master now runs:
  **−0.00066 on log loss, exact score and HT/FT (4 of 6) and −0.00011 on RPS
  (5 of 6)**, rising to −0.00071 at this form's own optimum of 1.27. Roughly
  46% of the original headline was the γ re-scan, which master does on its own;
  the structural half is worth about −0.0007 on top of it, with the fold count
  knife-edge in β (see the correction section);
* the draw deficit that motivated it goes from z = +5.4 and O/E 1.096 to
  O/E 1.016 with the two changes together;
* the coherence invariant holds exactly, so no page shows a 1X2 its own grid
  contradicts;
* over/under and BTTS are flat (−0.00006 each), which the identity says is all
  they can be.

Not recommended: re-scanning γ alone (half the gain, 5 of 6, leaves the draw 6%
short), fitting α (worse out of sample than fixing it at 1), `vector` (overfits
on short windows), `decouple` in any form (dominated, and it costs the
invariant), and touching ρ in the same change.

**DEPLOYED in v0.63.0** (2026-10-18), after this measurement, as
`sharpen_contrast_only` in `DEPLOYED_CLUB_ENGINE_KWARGS`. It is deployed on
this branch beside γ = 1.3; master has since re-scanned γ to **1.20** for the
power form (`52d94c7`, `9021b0b`). **The two commits are not additive**: 1.20
is the power form's optimum and this form's is 1.268, so merging them is the
weakest of the three configurations. If this branch lands, the γ commit should
be dropped rather than merged. The flag defaults to
**False**, so everything scored before that date stays reproducible; the
mechanism is `prediction_engine.sharpen_trio` and the tests are
`tests/test_sharpen_contrast_only.py`. On the published walk-forward figures it
moved 1X2 RPS 0.1999 → 0.1998, log loss 0.9857 → 0.9846 and coverage of the
base-rates-to-closing-line span 85% → 86%. The other forms in this file stay
research-only.

## What is worth keeping from this

* **Under coherence, exact score and HT/FT are the 1X2 log loss plus a
  constant.** Anything that reprices the matrix markets through the trio is a
  1X2-calibration question and should be scored and fitted as one, on log loss.
  Two markets that used to need their own validation now follow for free.
* **A sharpening exponent on a 3-outcome trio is two corrections wearing one
  hat.** p^γ scales both log-odds against the draw. Separating the draw axis
  from the home-vs-away contrast costs one line and is worth 0.0013, and it is
  the second time a parameter in this chain has turned out to be stale because
  something it was jointly fitted with later moved (γ with ρ here; `outcome_rho`
  with coherence in `scripts/bivariate_goals/README.md`). **Anything fitted
  jointly needs re-scanning when its partner changes.**
* **The 1X2's own calibration is not monitored, and cannot be.**
  `report_market_drift.py`'s `calibration_drift()` opens with
  `ev[~ev.mercado_id.isin(PICK_ONE)]` and `PICK_ONE = {"1X2", "ht_1x2",
  "ht_ft"}`: a pick-one market logs one selection, not three probabilities, so
  the 1X2's per-outcome calibration is excluded by construction. The drift
  report would never have raised this, however long it ran. It is a cheap fix —
  the log has carried `p_home`/`p_draw`/`p_away` since PR #25 — and it is the
  useful residue here whatever happens to the model change.
* **The live log agrees and cannot prove it.** Of 496 logged 1X2 rows, 286 carry
  the trio and **47** are settled: the draw is priced 0.2400 and happened
  0.2766, the right sign at z = +0.59. Detecting a +0.0222 gap at |z| ≥ 3 needs
  about **3,331** settled matches. This one is settled by the backtest or not at
  all.

## Caveats

* The λ cache is from 2026-10-12 and predates `market_level_shift` (2026-10-17).
  That is a common multiplier on both λ, which barely moves the trio and moves
  every arm here identically.
* The *form* α = 1 was chosen after looking at section 1's measurement, which is
  how it should be — but it is a modelling choice informed by the same six
  seasons it is scored on. What is not fitted on them is the parameter: the fold
  record above estimates nothing. The fitted regimes are reported beside it and
  land in the same place.
* 2026/27 is not a fold: it is not in the walk-forward cache.
