# A real goals correlation vs the Dixon-Coles patch (2026-10-18)

Experiment scripts. **The engine is untouched**: nothing here is wired into
production, and `statistical_core/distributions.py` is byte-identical. Run from
the repo root.

```
python scripts/experiment_bivariate_goals.py all       # ~20 min
python scripts/experiment_bivariate_goals.py corr      # just the measurement
```

## Why

The README's own "Known limitations" opens the item:

> Goals are modelled as **independent** Poisson with the Dixon-Coles correction
> applied to the low-score cells only. That is a patch, not a correlation
> model; a true bivariate Poisson or a copula would be the next step.

The 1X2 is not where that would pay — dependence largely cancels once you
aggregate 121 cells into three outcomes, and the question of beating the
closing line on 1X2 is closed anyway. The markets with something to gain are
the ones that read the matrix: exact score, BTTS, over/under and HT/FT.

So: measure the dependence first, then model it two ways, then score it.

## 1. How much dependence is there, before any model

Over the whole foundation, 46,138 matches and 27 seasons, home and away goals
are **negatively** correlated: Pearson −0.0529, Spearman −0.0375,
Cov −0.0770. That number is the wrong one, and the reason matters.

On the 10,403 walk-forward matches, where the λ actually served is on disk:

| | Cov(home, away) |
|---|---|
| marginal, as observed | **−0.1168** |
| = Cov(λ_h, λ_a) across matches | −0.0852 |
| + Cov conditional on the λ | **+0.0138** |

Almost all of the observed negative correlation is **mismatch, not
dependence**: a strong side's high λ_h comes paired with a low λ_a
(corr(λ_h, λ_a) = −0.5693), and that alone produces a negative marginal
correlation under conditional independence. The only part a model could use is
the conditional covariance, and it is +0.0138 with a bootstrap standard error
of 0.0125 — **t = +1.10**, a residual correlation of +0.0104.

It does not hold up anywhere you look for it:

| fold | n | conditional Cov |
|---|---|---|
| 2020/21 | 1823 | +0.0239 |
| 2021/22 | 1823 | +0.0132 |
| 2022/23 | 1825 | −0.0018 |
| 2023/24 | 1747 | +0.0181 |
| 2024/25 | 1444 | −0.0204 |
| 2025/26 | 1741 | +0.0471 |

Two of six seasons are negative. Conditioning on λ_total gives
+0.052 / +0.060 / +0.002 / −0.041 / −0.005 by quintile and conditioning on
log(λ_h/λ_a) gives +0.024 / +0.019 / +0.017 / −0.019 / +0.028 — no monotone
pattern in either, so there is nothing to make the dependence a function of.
At half-time the conditional covariance is **−0.0018**, which is why
`props/half_time.py` treating the halves as independent was never the weak
point.

**And the patch is already sitting on it.** The deployed scoreline matrix
implies a covariance of its own:

| matrix | implied Cov(H, A) |
|---|---|
| pure independent Poisson | −0.0000 |
| Dixon-Coles −0.06 + temper 1.05 | +0.0066 |
| **as published** (+ 1X2 sharpening + coherent reweighting) | **+0.0152** |

Measured +0.0138, published +0.0152. The prize for modelling the correlation
properly is the difference between those two numbers, and there isn't one.

## 2. Where the signal actually lives

A +0.014 covariance could be two different things, and they want opposite
fixes. Observed / expected scoreline counts under independence with the
deployed marginals:

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | 1.000 | 0.885 | 0.981 | 1.032 |
| **1** | 0.982 | **1.071** | 0.960 | 0.875 |
| **2** | 0.945 | 1.017 | 1.059 | 1.003 |
| **3** | 1.033 | 1.012 | 1.030 | 1.066 |

Draws as a whole come in at O/E **1.0479** — 4.8% more than independence —
and they are spread evenly along the diagonal: 1.0453 on 0-0 and 1-1 against
1.0553 on 2-2 and up. A low-score effect would sit in one corner. This is a
draw effect across the whole diagonal, which is the shape a genuine positive
dependence has, and it is the one honest argument for trying a real
correlation model. Section 4 is what happened when we did.

## 3. The families

`dependence.py` has five, each with one parameter that is zero at
independence (asserted in `tests/test_bivariate_goals.py`, which also checks
that every family at 0 reproduces `scoreline_distribution` exactly):

| family | parameter | note |
|---|---|---|
| `independent` | — | the product-Poisson the engine starts from |
| `dixon_coles` | ρ | the deployed patch, four cells rescaled |
| `bivariate_poisson` | λ₃ | Karlis-Ntzoufras H = X₁+X₃, A = X₂+X₃; Cov = λ₃ exactly |
| `frank_copula` | θ | Frank copula on the two discrete marginals |
| `gaussian_copula` | ρ | Gaussian copula, exact CDF (24-node Gauss-Legendre on the correlation path, agrees with scipy to 1e-16) |

**The shared component can only produce positive covariance** — λ₃ ≥ 0 — which
is the right shape only if the measured dependence is positive. It just about
is. The copulas can go either way.

Marginals are held fixed across families: the copulas preserve theirs by
construction, and the shared component and Dixon-Coles are
iteratively-proportional-fitted back onto the deployed tempered-Poisson
marginals. Row/column scaling leaves every odds ratio alone, so this moves the
marginals without touching the dependence — without it a family would be
scored partly on a marginal change, and the deployed chain already has a
validated marginal correction (`goal_temper` 1.05).

Maximum likelihood on the exact score, all six seasons:

| family | fitted | implied Cov | log-lik gain / match |
|---|---|---|---|
| `bivariate_poisson` | λ₃ = +0.0243 | +0.0220 | +0.000136 |
| `frank_copula` | θ = +0.1578 | +0.0285 | +0.000282 |
| `gaussian_copula` | ρ = +0.0193 | +0.0213 | +0.000141 |
| **`dixon_coles`** | **ρ = −0.0460** | **+0.0051** | **+0.000645** |

The patch takes two to five times the likelihood gain over independence, with
a quarter of the covariance. (Its fitted −0.046 against the deployed −0.06 is also a
cross-check on `scripts/experiment_scoreline_calibration.py`, which is where
−0.06 came from.)

The reason shows up on the one quantity both devices exist to fix. On the raw
trio — before the 1X2 sharpening, before any reweighting — against an observed
draw rate of 0.2540:

| family at its fitted value | P(draw) | O/E draws | 1X2 RPS | 1X2 log loss |
|---|---|---|---|---|
| `independent` | 0.2424 | 1.0479 | 0.20147 | 0.99129 |
| **`dixon_coles` ρ = −0.06** | **0.2560** | **0.9920** | 0.20143 | **0.99041** |
| `dixon_coles` ρ = −0.046 | 0.2528 | 1.0045 | 0.20143 | 0.99052 |
| `frank_copula` θ = +0.1578 | 0.2461 | 1.0320 | 0.20133 | 0.99053 |
| `gaussian_copula` ρ = +0.0193 | 0.2449 | 1.0372 | 0.20139 | 0.99081 |
| `bivariate_poisson` λ₃ = +0.0243 | 0.2446 | 1.0384 | 0.20139 | 0.99084 |

Dixon-Coles closes the draw deficit (O/E 1.048 → 0.992). The dependence
families, at the parameter their own likelihood picks, close **a fifth to a
third of it** and leave 3–4% on the table, because the amount of covariance the data
supports is nowhere near enough to move 2.4 points of draw mass. To get there
a copula needs θ ≈ 0.5 or ρ ≈ 0.08 — four times the fitted value and six times
the measured covariance — and at that setting the exact score is worse than
independence (2.91265 and 2.91287 against 2.91160).

One trap worth recording: **raw 1X2 RPS cannot select this parameter.** It
falls monotonically across the entire grid in every family — Frank runs
0.20252 at θ = −1 down to 0.20082 at θ = +1, straight past the point where the
draw rate is calibrated (O/E 0.998 at θ = 0.5) and on to 5% over-predicting it
(0.950 at θ = 1.0). Pick on RPS and you run to whatever boundary the grid has.
The exact-score likelihood is the criterion with an interior optimum, and
Dixon-Coles is already sitting on it.

And the fitted dependence parameter is not stable — the
per-season oracle for the shared component runs +0.0273, +0.0322, **0.0000**,
+0.0351, **0.0000**, +0.0564, hitting its floor in the two seasons whose
measured covariance was negative.

## Harness

Scored on `walkforward_preds_deployed.csv`: six seasons, each predicted by an
engine fitted only on earlier ones, 10,403 matches with the λ as served. The
dependence parameter is fitted **temporally** — fold k on seasons 1..k−1
(`expanding`, five scorable folds) — with leave-one-season-out and a
per-test-season `oracle` reported beside it as ceilings, not as candidates.

**The published 1X2 is held fixed.** Each candidate matrix is reweighted so
its three outcome regions carry exactly the probabilities the page publishes,
which is what `coherent_markets` already does in production
(`experiment_market_coherence.py`, deployed 2026-09-23). That makes the
comparison fair — only the goal markets can move — and it keeps the invariant
the repo guarantees. Verified for all five families, on the full-time matrix
and on the nine HT/FT paths: worst disagreement **9.99e-16**. The 1X2 question
is then asked separately, by letting each family sharpen and publish its own
trio.

Markets: 1X2 (RPS and log loss), over/under 2.5, BTTS, exact score and the
nine-way HT/FT, with the dependence injected into both halves (a copula
parameter carries over unchanged, the shared component scales with the half's
goal volume).

**The vectorised chain is verified twice.** It reproduces the single-match
reference implementations in `dependence.py` to 1e-11, and the baseline arm
reproduces the engine's own published matrix to **1.11e-16** and
`HalfTimeModel.predict_ht_ft` to **4.12e-13**.

Baseline, 10,403 matches: 1X2 RPS **0.20045**, O/U 2.5 **0.67896**, BTTS
**0.68440**, exact score **2.90878**, HT/FT **1.87747** (log loss except the
first).

## Results — NEGATIVE. Not deployed.

Deltas against the deployed baseline, published 1X2 held fixed; negative is
better, and the bar in this repo is 5 of 6 folds.

| arm | O/U 2.5 | BTTS | exact score | HT/FT |
|---|---|---|---|---|
| `bivariate_poisson` [expanding] | −0.00011 (3/5) | +0.00008 (1/5) | −0.00031 (4/5) | +0.00035 (0/5) |
| `bivariate_poisson` [loso] | −0.00001 (3/6) | +0.00012 (1/6) | +0.00005 (3/6) | +0.00038 (0/6) |
| `frank_copula` [expanding] | −0.00013 (3/5) | +0.00005 (1/5) | +0.00003 (1/5) | +0.00075 (0/5) |
| `frank_copula` [loso] | −0.00005 (3/6) | +0.00008 (1/6) | +0.00019 (2/6) | +0.00063 (0/6) |
| `gaussian_copula` [expanding] | −0.00012 (3/5) | +0.00007 (1/5) | −0.00015 (4/5) | +0.00045 (0/5) |
| `gaussian_copula` [loso] | −0.00002 (3/6) | +0.00011 (1/6) | +0.00007 (3/6) | +0.00037 (0/6) |
| `gaussian_copula` [oracle] | −0.00001 (3/6) | +0.00009 (1/6) | −0.00008 (3/6) | +0.00033 (2/6) |

Not one market clears the bar in any arm. The best cell is the exact score at
4 of 5 — and it is worth −0.00031, which is exactly what simply *removing*
Dixon-Coles gets on 5 of 6 folds (see below), so the shared component on top
of it is adding nothing. Over/under has the most consistent sign and it is 3
of 6 every time. BTTS is **worse in five of six folds in every single arm**.
HT/FT is worse everywhere, including under the oracle fit. The largest effect
anywhere in the table is 0.00075 of log loss, 0.04% of that market's own
level.

And the 1X2, with each family publishing its own sharpened trio instead (these
arms move the trio as well, so the exact-score column mixes both effects):

| arm | 1X2 RPS | 1X2 log loss | exact score |
|---|---|---|---|
| `independent` (ρ = 0) | +0.00026 (0/6) | +0.00285 | +0.00254 |
| `bivariate_poisson` λ₃ = +0.0243 | +0.00022 (0/6) | +0.00228 | +0.00229 |
| `gaussian_copula` ρ = +0.0193 | +0.00022 (0/6) | +0.00222 | +0.00225 |
| `frank_copula` θ = +0.1578 | +0.00020 (0/6) | +0.00192 | +0.00209 |

**No family prices the 1X2 better than the patch, in zero of six folds.** The
copulas recover a quarter of what dropping Dixon-Coles costs and no more.

### Is there any parameter that would have worked?

`sweep` scores every market across the whole parameter range, in sample, on
all six seasons at once — the most generous test there is. The answer is no
cell improves more than one market:

| | implied Cov | O/U 2.5 | BTTS | exact score | HT/FT |
|---|---|---|---|---|---|
| `bivariate_poisson` λ₃ = +0.20 | +0.1824 | **−0.00008** | **−0.00022** | +0.00938 | +0.00664 |
| `frank_copula` θ = +1.0 | +0.1789 | **−0.00014** | **−0.00021** | +0.01041 | +0.00769 |
| `gaussian_copula` ρ = −0.011 | −0.0121 | +0.00002 | +0.00022 | **−0.00039** | **−0.00017** |
| `gaussian_copula` ρ = −0.08 | −0.0879 | +0.00017 | +0.00051 | +0.00095 | **−0.00064** |

BTTS and over/under want an order of magnitude *more* positive dependence than
the data contains — λ₃ = 0.20 against a measured covariance of 0.0138 — and
buying their −0.0002 costs 40 to 50 times as much exact score. The exact score
and HT/FT want a small **negative** dependence, which is the opposite sign to
every maximum-likelihood fit in section 3. There is no parameter the four
markets agree on.

That last row is the only cell in the experiment that clears the bar —
`gaussian_copula` ρ = −0.011 is exact score −0.00039 (5/6) and HT/FT −0.00017
(6/6) — and it is not a result:

* it was chosen by reading all six folds at once, which is the definition of
  in-sample parameter selection;
* BTTS is worse (1/6) and over/under is flat;
* **−0.00031 of the −0.00039 is just removing Dixon-Coles** (`independent`,
  ρ = 0, is exact score −0.00031 on 5 of 6 folds). The copula on top is worth
  −0.00008;
* and every fitting rule that only sees the past lands on a *positive*
  parameter. A knob whose data-fitted value has the wrong sign for the markets
  it is meant to help is not a model.

### The one real finding: under coherence, ρ = −0.06 is slightly too strong

Dropping Dixon-Coles entirely improves the exact score on 5 of 6 folds:

| fold | n | O/U 2.5 | BTTS | exact score |
|---|---|---|---|---|
| 2020/21 | 1823 | +0.00044 | +0.00052 | +0.00142 |
| 2021/22 | 1823 | −0.00017 | +0.00014 | −0.00023 |
| 2022/23 | 1825 | −0.00048 | −0.00015 | −0.00099 |
| 2023/24 | 1747 | +0.00003 | +0.00014 | −0.00057 |
| 2024/25 | 1444 | −0.00009 | +0.00024 | −0.00089 |
| 2025/26 | 1741 | +0.00030 | +0.00020 | −0.00072 |
| **all** | 10403 | +0.00001 (3/6) | +0.00018 (1/6) | **−0.00031 (5/6)** |

This is a side effect of the coherence deployment, and it reverses on the raw
matrix: scored *without* the reweighting, Dixon-Coles has the best exact-score
log loss of any one-parameter family anywhere in this experiment —

| family, raw matrix | best exact-score log loss |
|---|---|
| `dixon_coles` ρ = −0.046 (the ML fit) | **2.91096** |
| `dixon_coles` ρ = −0.06 (deployed) | 2.91103 |
| `frank_copula` θ = +0.15 | 2.91132 |
| `gaussian_copula` ρ = +0.02 | 2.91146 |
| `bivariate_poisson` λ₃ = +0.025 | 2.91147 |
| independence | 2.91160 |

ρ was fitted in 2026-07 on the raw
matrix; coherent markets arrived in 2026-09 and pin each outcome region to the
published 1X2, which cancels most of what ρ was doing to the draw mass and
leaves only its redistribution *within* the regions — pushing mass from 2-2
and up toward 0-0 and 1-1, which the O/E table says is the wrong direction
once the draw region is pinned.

It is still not a deployment: +0.00018 on BTTS against −0.00031 on the exact
score, one fold (2020/21, the empty-stadium season) carrying +0.00142 on its
own, and the whole thing is 0.01% of the market's level. The honest statement
is that **ρ is no longer at its optimum under coherence, and the optimum is
worth 0.0003** — which is a tenth of what the coherence deployment itself was
worth (−0.0032 on exact score).

## What is worth keeping from this

* **The next step named in the README has been measured, and it is not the
  next step.** There is no goals correlation of consequence to model: +0.0138
  conditional covariance at t = 1.10, sign-unstable across seasons, with the
  deployed matrix already implying +0.0152 of it. The item in "Known
  limitations" is now a measurement, not an open question.
* **The patch beats the principled model at the thing both are for.** Fitted
  on the same data with the same single parameter, Dixon-Coles takes 2 to 5
  times the exact-score likelihood gain of a bivariate Poisson or either
  copula, with a quarter of the covariance. "Independent Poisson with a low-score
  patch" is not an approximation to a correlation model — it captures a
  low-cell effect a global dependence parameter cannot reach, and a global
  dependence parameter big enough to help BTTS destroys the scoreline.
* **Coherence changed what ρ means, and nobody re-fitted it.** Worth 0.0003
  of exact score, so it is not urgent, but any future tuning of `outcome_rho`
  has to be scored *through* the reweighting, not on the raw matrix.
* **The published 1X2 under-prices draws by 2.2 points, and the grid inherits
  it.** Observed draw rate 0.2540; the raw trio says 0.2560 (O/E 0.992) and
  the published, sharpened trio says 0.2318 (O/E 1.0957) — a gap of +0.0222
  with a standard error of 0.0041, **z = +5.4**. The γ = 1.3 sharpening buys
  0.00098 of 1X2 RPS with it, so by the score it is paying for itself, but
  coherence now pushes that deficit into every scoreline, BTTS and HT/FT
  number on the page. **No dependence model can touch it** — reweighting pins
  the draw region by construction, which is exactly why every arm above is
  flat. If there is a real 0.002-scale gain left in the matrix markets, it is
  in that interaction and not in the correlation. **Followed up the same day and
  it was: `scripts/draw_deficit/README.md` measures it at 0.00128 on exact score
  and HT/FT in 6 of 6 seasons, three times anything in this experiment.**
* **Nothing is wrong with the halves.** Conditional covariance at half-time is
  −0.0018, and every arm that added dependence to the halves made HT/FT worse,
  0 of 6 folds.

## Caveats

* The λ cache is from 2026-10-12 and so predates `market_level_shift`
  (deployed 2026-10-17). That shift is a common multiplier on both λ, which
  `scripts/goals_level/README.md` measured as ≤0.0001 of RPS, and it leaves
  the dependence structure untouched — it moves every arm here identically.
* 2026/27 is not a fold: it is not in the walk-forward cache, and with 297
  matches it could not settle a t = 1.10 effect anyway.
