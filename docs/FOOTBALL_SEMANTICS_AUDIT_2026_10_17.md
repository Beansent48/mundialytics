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

**Fix and test.** Estimate the scale per provider (or harmonise at ingestion and
drop the correction). The provider offset is already measured on 250 paired
matches — 0.9066 / 0.939 ≈ 0.966 — which is a far better estimator than a
season residual. Unlike an in-season level term this is validatable: a
provider's scale is a property of the provider, not of the season.

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
