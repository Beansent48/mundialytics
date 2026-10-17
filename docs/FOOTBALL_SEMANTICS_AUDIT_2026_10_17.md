# Football-semantics audit, 2026-10-17

Looking for the class of defect where the **code is right and the football is
wrong** — the kind the drift report had, where pooling by match date silently
pooled model versions because October fixtures can be priced by the September
model.

Nothing here is deployed. Each finding says how to test it.

---

## 1. The xG scale correction is estimated on one provider and applied to another

**Confirmed, quantified. The largest single item.**

`lambda_scale_` (prediction_engine.fit) converts the xG-level λ into goals
units: `clip(goals.mean() / xg.mean()` over the last 730 days`, 0.85, 1.10)`.
Today it is **0.9066**, over a window starting 2024-10-18 that is **90.7%
Understat** data.

But the current season's xG is not Understat's. Understat is dead as a source;
2026/27 is Sofascore for the first 250 matches and ESPN commentary text after
that — and `build_espn_text_xg.align_to_real` *already* rescales the text xG
down onto Sofascore's level, which the daily log confirms fired:

```
aligned to the real xG on 250 shared matches:
{Bundesliga: 0.914, LaLiga: 0.924, Ligue 1: 0.977, Premier League: 0.941, Serie A: 0.937}
```

That script's own docstring states the fact that makes this a bug: *"Text is
calibrated to Understat, which runs ~7% hotter than Sofascore's Opta numbers."*
So the inflation is corrected twice — once at ingestion, once by
`lambda_scale_`.

Measured goals/xG, same metric, different provider:

| data | provider | n | xG | goals | goals/xG |
|---|---|---|---|---|---|
| 2025/26 | Understat | 1741 | 3.099 | 2.768 | **0.893** |
| 2026/27 | Sofascore + aligned text | 297 | 3.162 | 3.094 | **0.979** |

The engine applies 0.9066 where 0.979 is right: a **7.4% excess haircut** on
the xG arm, which carries weight 0.6 → **≈4.8% of λ_total**, about **0.13 of
the 0.354 goals/match** deficit measured in `scripts/goals_level/`. Roughly a
third of the "2026/27 level shift" that an in-season correction failed to fix
is not a level shift at all; it is a unit mismatch.

Worth noting on its own: Understat's goals/xG drifts monotonically from 1.016
(2014/15) to 0.893 (2025/26) **in all five leagues at once**. Finishing rates
do not move 12% in one direction across the big five for a decade; the provider
does. Any xG-scaled quantity fitted at a past cutoff inherits that drift.

### Fixed and deployed

Harmonise forward instead of backward: `build_espn_text_xg.py --align history`
leaves the text xG on the Understat level step 1 already put it on, and
`update_season._on_history_level` lifts the Sofascore rows onto that same level
using the 250 shared matches (1.024–1.094 per league). One scale throughout, so
the xG-rate model is scored at the level it was trained at and `lambda_scale_`
measures the inflation its own data carries.

Measured on the 2026/27 fold, weekly-refit walk-forward (`outputs/xg_level/`):

| | baseline | harmonised | diff |
|---|---|---|---|
| O/U 2.5 log loss | 0.66937 | 0.66549 | **−0.00388 ± 0.00111, t = −3.49** |
| 1X2 RPS | 0.19887 | 0.19893 | +0.00006, t = +0.48 |

Historical rows are the reference level and provably do not move, so there is no
historical fold to validate on — by construction, not by omission. The evidence
is the mechanism plus t = −3.49 on 297 matches, and the change cannot regress a
past fold.

**Correction to the first version of this finding.** It projected ≈4.8% of λ,
about a third of the goals deficit. Wrong: λ_total moves only **1.14%** (2.740
→ 2.771). Two things damp it — a Poisson GLM does not pass a uniform rescale of
its features through one-for-one, and `lambda_scale_` is refitted on the
harmonised data (0.9066 → 0.9009), giving part of it back. The gain is real and
significant but it comes from feeding the model inputs on its training scale,
not from correcting the level. The 2026/27 level deficit is still mostly
unexplained.

---

## 2. A team's "recent form" is read from an undated list, so long-absent clubs are priced on ancient football

**Confirmed, demonstrated, measured.**

`XGRateModel._rate` is `mean(hist[-w:]) if len(hist) >= 3 else fallback`.
`hist` carries no dates, so "the last 5 matches" means the last 5 *rows*,
whenever they happened. The debutant stand-in (`serving/debutants.py`) only
catches a side with **no** history, and `len(hist) >= 3` passes comfortably for
every club below, so the fallback never fires.

2026/27, 13 of 96 clubs have a gap over a year:

| club | last previous top-flight match | gap | matches in the model's history |
|---|---|---|---|
| Coventry | 2001-05-19 | **25 years** | 38 |
| Le Mans | 2010-05-15 | 16 years | 212 |
| Santander | 2012-05-13 | 14 years | 406 |
| Hull | 2017-05-21 | 9 years | 190 |
| Málaga / La Coruña | 2018-05 | 8 years | 595 / 598 |
| Paderborn | 2020-06-27 | 6 years | 68 |
| Schalke 04, Troyes | 2023 | 3 years | 737 / 352 |
| Frosinone | 2024-05-26 | 2 years | 114 |
| Venezia, Monza, Ipswich | 2025-05 | 1 year | 110-114 |

Coventry's xG form, as priced, is Coventry's form in the **2000/01 Premier
League**. The goals-AD arm is immune — a 365-day half-life annihilates a
25-year-old match — so the defect sits entirely in the arm that carries 0.6 of
the weight.

The cost signature is dispersion, not bias. Per-club bias runs both ways
(Málaga 0.63, Schalke 0.65, Troyes 0.71 against Frosinone 1.69, La Coruña 1.57,
Hull 1.51): the signal is not too high or too low, it is arbitrary.

| 2026/27 group | n | 1X2 RPS | mean \|goals − λ\| |
|---|---|---|---|
| a side with form ≥2 years stale | 66 | **0.2203** | 1.512 |
| everything else | 214 | 0.1941 | 1.376 |

Some of that gap is irreducible — a returning club is genuinely harder to
price — so the number is an upper bound, not the prize.

**Fix and test.** Make the form window date-aware: drop entries older than N
months and fall through to the stand-in when fewer than 3 recent ones survive.
Unlike the level correction, **this has power in every fold** — 10-20% of the
matches in every season since 2018/19 involve such a club:

| season | clubs | matches affected | share |
|---|---|---|---|
| 2018/19 | 10 | 360 | 19.7% |
| 2019/20 | 10 | 352 | 20.4% |
| 2020/21 | 9 | 330 | 18.1% |
| 2021/22 | 6 | 216 | 11.8% |
| 2022/23 | 7 | 258 | 14.1% |
| 2023/24 | 5 | 182 | 10.4% |
| 2024/25 | 7 | 252 | 14.4% |
| 2025/26 | 7 | 254 | 14.5% |

This also sits exactly where the Bet365 gap was diagnosed (promoted sides,
matchdays 1-5), and it is independent of the squad-value shift that already
helped there.

### Measured: NEGATIVE, twice. Not deployed.

Two fallbacks, both on the 7-fold harness, both scored on the matches the fix
actually touches.

**A — prune the form to 730 days** (`XGRateModel.form_max_age_days`, opt-in,
off by default). A club without 3 matches inside the window falls back to the
league rate. On the 1,907 matches across 7 folds that involve a stale club:
d_rps **+0.00012**, d_ou **−0.00065**, **3/6 historical folds** on both. Nothing.
(It also cleared a worry: `d_rps` on every *other* match is ±0.00000, so the
pruning really only moves what it should.)

**B — treat a club absent >730 days as a debutant**, priced off last season's
relegated sides, a better prior than the league mean. **1/6 folds**, and on the
matches it touched d_rps +0.00237, d_ou +0.00266 — worse.

Scope limit worth naming on B: it moved only 40 λ of 10,700, far fewer than
intended. `train` includes the season in progress, so the moment a returning
club plays once it is "recent" again and the stand-in stops. B therefore tested
"debutant for its first match only", not for as long as it lacks current form.
A is the test of the full-duration version, with the league mean as the interim
prior, and A failed too.

So the honest reading: Coventry's 2001 form is absurd **as a description** and
no worse **as a predictor** than either replacement. Both are uninformative; a
returning club is simply hard to price, which is what
`promoted_gap_2026_09_29` already found from the other direction. The defect is
real, vivid and — on the goal arm — not worth fixing. What a returning club
needs is a better prior, and the one that measured positive is squad value,
already deployed.

Left unmeasured: the same defect in `TeamPropsModel` (see the correction below).
It is a different model with different fallbacks, so A and B do not settle it.

---

### Correction (same day): the root cause is in the props layer too

`props/team_props.TeamPropsModel._latest_team_state` does the same thing —
`g["ev_for"].tail(w).mean()`, the last *w* rows whatever their date, with the
same `len(g) < 3` guard that a club with 38 ancient rows sails past. That is
the model the **board's event markets are actually priced with** (`tp_*` in the
log), so Coventry's shots, corners and cards form is also its 2001 form.

Its league-position features are built correctly, per `(competition, season)`.

Unlike the goal side this is **not measurable from the live log**: only 11
settled 2026/27 matches carry both a stale club and an event expectation, and
they say nothing (shots error 6.16 vs 4.79, corners 1.87 vs 2.43, yellows 1.82
vs 1.60). Testing it needs `backtest_team_props.py` across folds.

## 3. No rolling window resets at the season boundary

**Real, mostly inert where I first pointed it, unmeasured.**

Correction to the first version of this finding: the deployed board does not
read the engine's `EventLambdaModel` (`exp_*` in the log) — `TeamPropsModel`
prices the event markets, and the goal path drops the GL model whenever the
xG-rate model is ready. So `features/team_features.py` feeds almost nothing
served, and the live exposure of this finding is the two form states covered by
finding 2.

`features/team_features.add_pre_match_rolling_features` and
`XGRateModel._add_rolling` both group by `team` alone. On matchday 1,
`goals_for_last5` and `xg_for_r5` are the previous **May** — a different squad
after a transfer window and ten weeks without competitive football — presented
to the model as current form. `team_match_count_pre` is likewise a career
count, so the shrinkage in `apply_rolling_feature_shrinkage` treats a club in
its first match of a season as fully observed.

Finding 2 is the extreme case of this same root cause. The milder case is
genuinely arguable — end-of-season form does carry information about a squad
that mostly stayed — so the point is not that it is wrong but that it is an
accident rather than a measured choice. The deployed goal path does not use the
`team_features` rollings at all (GL is dropped whenever the xG-rate model is
ready), so there the exposure is the **event markets**; the xG arm is exposed
through `_add_rolling`.

---

## 4. The competition layer cannot represent a points deduction

**Gap, not a bug in the code as written.**

`competition/standings.py` computes points purely from played matches. There is
no field anywhere in `src/` or the data for an administrative deduction, so for
a sanctioned club the live table and every title / top-four / relegation
probability built on it is off by the full deduction. Juventus −15 (2022/23),
Everton −8 then −2 and Nottingham Forest −4 (2023/24) are all inside the
backtest window.

It only bites when a deduction exists, which is why it has gone unnoticed; it
is cheap insurance to add a per-club, per-season adjustment column that
`standings` adds after summing results.

---

## 5. The fit cache key did not cover the xG it was fitted on

**Found while deploying finding 1. Confirmed, fixed.**

`api/engine._cached_fit` keyed the joblib fit on row count, last date, a code
fingerprint and the squad-value mtime:

```
clubs_46138_2026-10-12_d15e3ed7_1792142627
```

Nothing in there moves when xG **values** change while the row count and the
last date stay put — which is exactly what a provider re-score, a backfill, or
finding 1's harmonisation does. The stale fit would be reloaded from disk and
the new numbers would reach no request at all. Same shape as the drift report's
bug: a key that does not capture what it is keying.

No cache file happened to exist when finding 1 landed, so today's deployment was
not affected. The key now carries a cheap xG content digest (non-null count plus
rounded sum, O(n) over 46k rows); a permutation of the same values is correctly
still a cache hit.

## Checked and correct

Worth recording, because each of these is a place the football could easily
have been wrong and is not:

* **League tiebreakers are per competition.** Points → GD → GF for the Premier
  League, Bundesliga and Ligue 1; head-to-head first for La Liga and Serie A,
  applied only when the tied set is fully decided by their mutual results.
* **Two-legged European ties** resolve on aggregate, then extra time at the
  second leg (λ/3, home advantage retained), then a 50/50 shootout — with **no
  away-goals rule**, which UEFA abolished in 2021.
* **Elo resets 40% toward the mean between seasons**, and the goals-AD model's
  365-day half-life handles the summer gap properly.
* **The fit is league-only**: the foundation holds the five domestic leagues,
  regular season, no neutral venues, so no cup or European rotation leaks into
  per-league attack/defence strengths. European form was separately measured and
  rejected.
* **Debutant stand-ins** are last season's relegated clubs — a sound football
  choice for pricing a promoted side (its real flaw is finding 2: a club that
  never qualifies as a debutant because it played in 2001).

---

## Suggested order

1. **Finding 1** — confirmed, isolated, and the estimator already exists from
   250 paired matches. It also partially settles the question
   `scripts/goals_level/` closed negatively.
2. **Finding 2** — biggest measured effect, and the only one with a clean test
   in all six folds.
3. **Finding 4** — small, and insurance against a silently wrong table.
4. **Finding 3** — measure before changing; the plausible version of it is
   already covered by 2.
