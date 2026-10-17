# Goals level from the market (2026-10-17)

**Validated 5/6 folds and DEPLOYED** (`market_level_shift={"k": 70}` in
`DEPLOYED_CLUB_ENGINE_KWARGS`, `src/mundialytics/features/market_level.py`).

## Why this route

Two harnesses have shown that reacting to our **own** in-season residual fails —
shots (v0.56.0, 2/5) and goals (`scripts/goals_level/`, 1/6) — and both fail the
same way: in a normal season the residual wobbles ±2% with no persistence, so any
K fast enough to catch 2026/27's +12.9% buys that noise in the other six.

The problem was never the correction. It was the **estimator**: 47 of our own
matches cannot separate a real level shift from noise. The book's price can.

## The diagnostic, which is the real finding

Inverting Bet365's closing over/under 2.5 through the deployed chain gives, per
match, the level multiplier the market implies. Aggregated per season:

| season | goals / our λ | goals / the book's λ |
|---|---|---|
| 2020/21 | 1.0235 | 0.9983 |
| 2021/22 | 0.9944 | 1.0042 |
| 2022/23 | 0.9820 | 1.0011 |
| 2023/24 | 1.0147 | 1.0194 |
| 2024/25 | 1.0015 | 0.9916 |
| 2025/26 | 1.0193 | 0.9896 |
| **2026/27** | **1.1294** | **1.0447** |

The market's implied level is essentially unbiased every season (0.989–1.019).
Ours wobbles wider, and in 2026/27 we are 12.9% short where the book is 4.5%
short — **the market saw about two thirds of the shift**. The last third nobody
saw: that part is the season finishing above its chances.

## Result

`m` = the book-implied level of this league-season over **earlier dates only**,
shrunk to 1 with K pseudo-matches, clipped to [0.90, 1.10]. The price of the
match being predicted is never used, and neither is any price from its own date.

| K | O/U, 6 historical folds | folds | 1X2 | 2026/27 O/U |
|---|---|---|---|---|
| 40 | −0.00048 ± 0.00029 (t −1.66) | **5/6** | −0.00000 (4/6) | −0.00702 |
| 70 | −0.00047 ± 0.00025 (t −1.90) | **5/6** | −0.00000 (4/6) | −0.00538 |
| 120 | −0.00044 ± 0.00021 (t −2.13) | **5/6** | −0.00000 (4/6) | −0.00387 |

Flat across K=40–400, so the result does not live on a tuned constant. The one
failing fold is 2024/25, the season where the book's bias (0.9916) sat on the
*opposite* side of ours (1.0015), so the correction pushed the wrong way. That is
the honest cost of borrowing someone else's level.

Robust to the odds source, which is what rules out an artifact of one book's
pricing:

| source | O/U, history | folds |
|---|---|---|
| B365 closing | −0.00047 (t −1.90) | 5/6 |
| B365 opening | −0.00057 (t −2.56) | 5/6 |
| market average closing | −0.00051 (t −2.15) | 5/6 |

The opening line does slightly better, which fits: we are averaging a
season-level scalar, not hunting a per-match edge, so the closing line's extra
sharpness buys nothing.

## What it does and does not do

It corrects **our level bias**. It does not close the gap to the book: that was
+0.0374 of O/U log loss on 2026/27, and this is worth ~0.005 there and ~0.0005
in a normal season.

**The line it deliberately does not cross.** Only a league-season level comes
from outside, from matches already played. Feeding the book's per-match number
into λ would score far better and would not be a model — the measured gap is
that in a logistic against the closing line our own weight is +0.042 against the
book's +1.038 (`totals_gap_2026_10_05`). All per-match discrimination stays ours.

Operationally it is free: the daily refresh already downloads these CSVs (step
1/8), and only past matches' prices are needed.

    python scripts/market_level/score_market_level.py [--k 20,40,70,120,200,400]


## How it runs in production

`api/engine.club_engine()` calls `market_levels(df, None, k=70)` and
`engine.set_market_levels(...)` **after** the cached fit, never inside `fit()`:
the fit is cached to disk, so a level baked into it would be served long after
the matches it was measured on (audit finding 5, the same shape). It refreshes
whenever `_club_engine`'s key — the foundation's mtime — moves, which the daily
refresh does. Costs ~19 s per engine load, and it fails soft: any exception
leaves the λ untouched and prints why.

The level needs **our own served λ for matches already played this season**,
because that is what the harness measured against; reconstructing them with
today's fit would not be the same quantity. Two sources:

1. the pre-kickoff log, which has carried `lambda_home`/`lambda_away` since PR
   #25 and **grows every day** by itself (step 7b);
2. `data/processed/served_lambdas_current.csv`, a **one-off backfill** of the
   part of the season played before that (`build_served_lambdas_current.py`,
   step 7a, ~25 min).

**Nothing needs regenerating.** The backfill covers up to the log's first row
and the log covers everything after it, so once the two meet the level
strengthens on its own, and a new season bootstraps itself because round one is
logged before it is played. Step 7a checks and exits in a second when the gap is
closed, so it is safe to run daily and it self-heals a lost or stale cache
(`--force` rebuilds anyway).

The gap mattered on the day: the log alone held 47 settled matches, about 9 per
league, which at K=70 is **11%** of the correction's strength; with the backfill
it was 46%. Both callers now print the match count behind each level, because
this correction can only ever weaken **silently** — if the log stopped carrying
the λ, n would fall and nothing else would complain.

## Levels on the day it was deployed

| league | multiplier |
|---|---|
| Bundesliga | ×1.0140 |
| Premier League | ×1.0294 |
| Serie A | ×1.0297 |
| Ligue 1 | ×1.0373 |
| LaLiga | ×1.0423 |

Measured on 45 (Bundesliga) to 77 (LaLiga) matches, so 39–52% of full strength
and rising with every round.

On a typical LaLiga fixture that is about +2.9pp on P(over 2.5), with the 1X2
moving only as a goal distribution must (slightly fewer draws) — the validation
had 1X2 RPS at −0.00000.

## Rejected variant

Hierarchical shrinkage (league → pooled five-league level → 1), which is better
football since the 2026/27 drift is global and would have helped the thin-data
case: **4/6 folds** at every (K_pool, K_league) tried, O/U −0.00039 to −0.00043
with t −1.14 to −1.50, against the flat form's 5/6 and t −1.90. Better on
2026/27 (−0.0090 to −0.0120) and worse on history — the same trap the in-season
level forms fell into, so not deployed.
