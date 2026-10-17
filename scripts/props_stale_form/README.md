# Stale event state in the props layer (2026-10-17)

**NEGATIVE. Nothing deployed.** The defect is real; it is already neutralised
downstream, for exactly the markets where it would matter.

This closes the last open item of
`docs/FOOTBALL_SEMANTICS_AUDIT_2026_10_17.md` (finding 2's props half).

## The defect

`TeamPropsModel._latest_team_state` takes `g["ev_for"].tail(w).mean()` — the
last *w* rows whatever their date — past a `len(g) < 3` guard that a club with
38 ancient rows sails through. That is the model the board's event markets are
priced with, and `_side_lambda` reads the team's `for_*` **and the opponent's**
`against_*`, so one stale club spoils its own λ and its opponents'.

## The screen, which is the part worth keeping

Per-team error at predicting the club's own rate this season, 58 club-seasons
with a gap over two years, against the league-level drift over the same span:

| market | its ancient form | its league's level | league drift 2014/15 → 2026/27 |
|---|---|---|---|
| **fouls** | **2.433** | **0.887** | 13.90 → 11.69 (−16%, monotonic) |
| yellows | 0.440 | 0.294 | 2.08 → 1.84 |
| corners | 0.905 | 0.891 | 5.05 → 4.88 |
| shots | 1.586 | 1.797 | 12.46 → 14.13 (not monotonic) |
| sot | 0.655 | 0.873 | — |

The two columns rank the same way, and that is the mechanism: **where the
league level drifts, a club's ancient rate measures a refereeing era that no
longer exists; where it does not, the rate is a real team trait worth keeping.**
A club last seen in 2010 fouls the way 2010 was refereed.

It also explains a split the repo had already measured from the other side: a
league-season level term passed **for fouls only** (−0.0013, 5/5, deployed) and
failed for shots. Same axis, found independently.

**Re-basing** the ancient form onto today's league level — keep the style, fix
the era — is the elegant idea and it loses: fouls 2.433 → 1.464 (−40%) but
still far behind the plain league level's 0.887, and shots/sot get ~2% worse.

## The arm, and why it fails

Replace only `fouls` and `yellows`, and only while the club lacks 19 matches
inside two years, which makes it self-limiting (9 stale clubs in the first
blocks of a season, 1 by the sixth, 0 after). Seven folds, `TeamPropsModel`
refitted every 28 days, both arms scored through `predict_fixture`, so this is
what the board would have quoted. 130,968 quoted lines, 9.2% of them touchable.

| market | n | log loss base | arm | delta | t | folds |
|---|---|---|---|---|---|---|
| fouls | 1,998 | 0.60323 | 0.59806 | −0.00516 | −1.13 | **3/6** |
| yellows | 2,997 | 0.61378 | 0.61290 | −0.00089 | −0.73 | 5/6 |
| corners / shots / sot | — | — | — | 0.00000 | — | 0/6 |

Fouls has the size and not the consistency (one fold carries it: 2023/24
−0.0447, with three folds positive). Yellows is consistent in direction but the
effect is −0.00089 on the touched subset and −0.000081 across the board, with
t = −0.73 — indistinguishable from nothing. Neither passes.

Untouched rows were checked rather than assumed: max |p_base − p_arm| = 0, so
the patch moved only what it should.

## Why the big raw error does not become a market gain

```
LEVEL_K = {"fouls": 70}      # league-season level feature, deployed
RES_K   = {"yellows": 120}   # league-season residual correction, deployed
```

**Both markets the screen identifies as league-driven already carry a deployed
league-level correction**, so the model already absorbs the era drift my patch
would have removed. And the markets with no such protection — shots, sot,
corners — are exactly the ones where the stale state is *fine* (the ancient
form wins or ties).

So the system is already right here, market by market. The defect survives in
the state and is paid for nowhere, which is why it never showed up in the live
log either (11 settled 2026/27 matches touch it and say nothing).

    python scripts/props_stale_form/wf_stale_state.py SEASON OUT.csv   # ~6 min/fold
    python scripts/props_stale_form/score.py
