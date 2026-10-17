# npxG: a column that said non-penalty xG and contained xG (2026-10-18)

**Data fix deployed. The modelling idea it unblocked is NEGATIVE, twice.**

## The defect

`home_npxg` was a byte-for-byte copy of `home_xg` for **all twelve historical
seasons** — 100% of rows identical, penalty share 0.0000. Only 2026/27 had real
npxG, because it comes from a different provider.

`build_understat_xg_matches.py` is written correctly
(`npxg = xg.where(~is_penalty, 0)`), but it looks for `situation == "penalty"`
and **Understat leaves `situation` empty on a penalty**. `astype(str)` turns
that NaN into the string `"nan"`, `isin({"penalty"})` is never true, and nothing
is ever zeroed. The 6,677 shots with no situation have mean xG 0.7564 and 6,674
of them sit in [0.74, 0.79]: Understat prices every penalty at 0.761169.

The same quirk had already been found from another direction (the props layer
recorded "pens were in shots file all along, situation=NaN") and nobody went
back to the npxG aggregation.

Fixed, with the assumption checked rather than trusted: if a future scrape
starts leaving `situation` empty for something else, the mean xG of that group
leaves the penalty band and the build says so instead of mislabelling quietly.
Penalties are now **8.33%** of xG, 7.4–10.2% by season (2020/21 peaks at 10.24%
— the VAR-and-empty-stadiums season, a real effect).

The fix changes **no served λ**: verified 0.0e+00 across fixtures on the broken
and fixed foundations, because nothing in the deployed goal path reads npxG.
It is a correctness fix so that the next person reaching for npxG gets npxG.

## The idea it unblocked, and why it fails

`XGRateModel` — the arm carrying 0.6 of λ — builds its form windows from xG
*including penalties*. A penalty is 0.76 xG of pure noise for next week, so
npxG should be the better form signal. Two formulations, 7 folds each:

| arm | 1X2 RPS (hist) | folds | O/U (hist) | folds | λ error |
|---|---|---|---|---|---|
| `npxg` (windows only, target xG) | +0.00002 | 2/6 | +0.00074 | 2/6 | worse in 6/6 |
| `npxg_full` (windows + target + scale) | +0.00008 | 3/6 | +0.00084 | 2/6 | worse in 6/6 |

Both fail, and the bias column says why. The first **raises** λ in all six
historical folds (features drop ~8%, the target does not, so the GLM answers by
raising coefficients); the second **lowers** it in all six. Each time, the only
fold that gains is 2026/27 — whose level was wrong by +12.9% to begin with, so a
level shift in the lucky direction flatters it. That is not discrimination.

**The λ absolute error gets worse in every historical fold in both arms**, and
that is the real answer: the textbook claim that npxG predicts better than xG is
about predicting future *performance*, not future *goals*. Goals include penalty
goals. For an engine whose target is goals, penalties are legitimate if noisy
signal, and removing them throws away something real.

`form_source` stays in the model, defaulting to `"xg"`, documented with both
results so nobody has to rebuild the harness to re-ask.

    python scripts/goals_level/wf_lambdas.py SEASON OUT.csv --src foundation \
        --found <fixed foundation> [--npxg-form | --npxg-full]
    python scripts/goals_level/compare_arms.py outputs/npxg_full --base outputs/npxg/base
