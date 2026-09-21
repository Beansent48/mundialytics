# Mundialytics

[![tests](https://github.com/Beansent48/mundialytics/actions/workflows/tests.yml/badge.svg)](https://github.com/Beansent48/mundialytics/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A football match-prediction engine: probabilistic forecasts for 1X2, over/under,
BTTS, half-time markets and player/team props, fitted on **more than 46,000
matches across 27 seasons (2000/01–2026/27)** of the Big 5 European leagues — England, Spain,
Italy, Germany, France — and benchmarked against bookmaker closing odds. A
second engine covers national teams on international results from 2010.

Built and operated solo. It runs a live forward test: every prediction is written
to [`predictions_log.csv`](data/processed/logs/predictions_log.csv) **before
kick-off** and settled against the real result afterwards, so the track record
cannot be back-fitted.

![Match view](docs/img/app.png)

*One fixture in the web app: 1X2, expected goals, likely scorers, goal markets and
the likeliest scorelines. The match markets are all read off the same score
distribution, so they cannot disagree with each other.*

## The headline result

Measured on **10,080 Big 5 matches, 2020/21–2025/26**, against Bet365's
**closing** odds (100% closing coverage, proportionally de-vigged). Odds are
never model inputs — they are only the yardstick.

Where the engine sits, in ranked probability score (lower is better):

| Predictor | RPS (1X2) |
|---|---|
| Uniform 1/3 — no information at all | 0.2356 |
| League base rates — home advantage only | 0.2308 |
| **This engine** | **0.2008** |
| Bet365 closing odds | 0.1946 |

Between knowing nothing and the best-informed price on the market there is
0.0362 of RPS. **The engine covers 83% of it.**

Head to head with the closing line:

| | RPS (1X2) | Log loss (1X2) | Log loss (O/U 2.5) |
|---|---|---|---|
| This engine | 0.2008 | 0.9881 | 0.6795 |
| Bet365 closing | **0.1946** | **0.9680** | **0.6710** |
| gap | +0.0062 | +0.0201 | +0.0085 |

It does **not** beat the closing line — about 3% behind, and the gap holds in
every season and every league (worst Premier League +0.0083, best Bundesliga and
LaLiga +0.0039), which is what makes it a ceiling rather than noise. Bet365's
closing price carries injury news, confirmed lineups and the weight of informed
money; this engine has public data and nothing else.

Reproduce all of it (the benchmark scores the deployed walk-forward cache, so
generate it first):

```bash
python scripts/generate_deployed_walkforward.py   # writes the deployed cache
python scripts/benchmark_vs_bet365.py
```

## Is it calibrated?

Being close to the market is one question; saying 30% and being right 30% of
the time is another. It is:

![1X2 reliability diagram](docs/img/calibration_1x2.png)

Home and away curves track the diagonal across the whole range. Draws never get
predicted above ~35% — a known property of the sport, not a defect: draws are
genuinely rarely the favourite. The figure scores all 10,403 walk-forward
predictions, so its RPS reads 0.2013; the benchmark above uses the 10,080 of them
that have Bet365 odds attached. Regenerate with
`python scripts/plot_calibration.py`.

## How it works

```
football-data.co.uk ──┐
Understat / Sofascore  ├─→ canonical match schema ─→ features ─→ models ─→ markets
StatsBomb / ClubElo    │      (entity resolution)        │          │
FBref / ESPN ──────────┘                                 │          ├─ 1X2 / O-U / BTTS
                             internal Elo ───────────────┤          ├─ half-time markets
                             walk-forward form ──────────┤          ├─ team props
                             xG-rate predictor ──────────┘          └─ player props
```

- **Goal model** — Dixon-Coles double Poisson: per-team attack/defence strengths
  fitted jointly by maximum likelihood with exponential time-decay weights, plus
  the Dixon-Coles low-score correction and an internal Elo prior. The low-score
  correction (and a light sub-Poisson temper) are tuned walk-forward to the
  exact-score distribution rather than hand-set — a milder draw correction fits
  the real scorelines better on all 5 folds (scoreline log-loss 2.926 → 2.916,
  1X2 RPS and O/U 2.5 nudged the same way, none regressing;
  `scripts/experiment_scoreline_calibration.py`).
- **Form** — rolling team features computed strictly walk-forward: at every
  point in the backtest, only matches already played are visible. The single
  biggest accuracy gain in the project — roughly 0.004 of RPS, improving in 5 of
  5 temporal folds.
- **xG** — a dedicated model predicts each side's expected-goal *rate* rather
  than using raw historical xG as a feature; the raw feature turned out to be
  largely redundant with the strength estimates.
- **Calibration** — Platt scaling per market, validated on held-out seasons.
- **Scopes** — club and national-team models are fitted and used separately.
  Strength parameters are not comparable across contexts, so mixing them would
  quietly corrupt both.
- **Markets** — the score matrix is derived once and every market (totals,
  BTTS, half-time, correct score) is read off it, so they stay mutually
  consistent by construction. The headline "likely score" is reported *per
  outcome* (most likely home win, draw, away win), because the single global
  modal score of a product-Poisson matrix is a near-constant low draw and says
  little about who is favoured.

Validation is **temporal out-of-sample throughout**: train on seasons up to
year *N*, test on *N+1*. Never k-fold — shuffled folds leak the future into the
training set and inflate football models badly.

## What didn't work

Recorded because negative results are the expensive part of the project:

- **Beating the closing line.** Four experiments across price levels, leagues,
  seasons and divisions. Not a calibration problem, not home advantage, not
  missing lineups (oracle-capped at ~3% gain), not lower divisions, not opening
  prices. Bet365 was ahead everywhere. Conclusion: stop.
- **xG as a direct feature.** Redundant with the maximum-likelihood strength
  estimates. Only the derived xG-*rate* predictor added signal.
- **Per-team first-half share.** Measured correlation r = −0.118 — noise. The
  half-time markets use a global scaling instead.
- **Shrinking per-competition parameters.** `AttackDefenseModel` fits μ and home
  advantage independently per competition, so small ones over-fit: held out on
  internationals since 2023, the AFC Asian Cup over-predicted both sides by
  ~0.9 goals off ~115 training matches. Credibility-shrinking each competition
  toward the sample-weighted pooled fit does exactly what it was designed to do
  — Asian Cup bias +0.92 → +0.05 — and looked like a small RPS gain on that one
  window. Across five temporal folds it does not survive: one better, two worse,
  two tied, with the away-goal bias improving in two and worsening in two. Kept
  as an opt-in `competition_shrinkage_k`, defaulting to 0, and not deployed.
  Sweep it with `scripts/experiment_competition_shrinkage.py`.

## Conclusions

- **The club engine works.** Calibrated 1X2 probabilities, 83% of the way from
  an uninformed baseline to the closing line, validated out-of-sample in time.
  That is the part with an external yardstick behind it.
- **As a betting edge, it does not.** That question was asked properly and the
  answer was no, so that line of work is closed.
- **What it is now:** an analytics engine — simulate a season from the current
  table, price a hypothetical squad, quantify the uncertainty in a fixture —
  shipped with a measured statement of how far to trust it.

## Known limitations

- Goals are modelled as **independent** Poisson with the Dixon-Coles correction
  applied to the low-score cells only. That is a patch, not a correlation model;
  a true bivariate Poisson or a copula would be the next step.
- **Team level.** The main model sees no lineups and no injuries. An oracle test
  — handing the model perfect lineup knowledge — capped that loss at ~3%.
- **One bookmaker.** Bet365 alone. A multi-book consensus, or Pinnacle, would be
  a harder yardstick.
- **Big 5 leagues only** for the trained model; no cross-league scale.

## Status

Split by how strong the evidence is, because "it works" means three different
things below and collapsing them would be dishonest.

### Measured out-of-sample

| Area | Evidence |
|---|---|
| Club match prediction — 1X2, O/U 2.5 | 10,080 matches against Bet365 closing odds · `scripts/benchmark_vs_bet365.py` |
| Club 1X2 calibration | reliability diagram over 10,403 walk-forward predictions · `scripts/plot_calibration.py` |
| European competitions | held-out season, RPS 0.2044 vs 0.2327 base rates · `scripts/evaluate_european_layer.py` |
| National teams | 1,425 internationals held out, RPS 0.1725 vs 0.2317 base rates; calibrated on the bulk, with a small-sample competition bias · `scripts/evaluate_national_engine.py` |
| Player props — 6 markets | 263,551 player-matches, beats naive and position baselines in 5 of 5 temporal folds, ECE 0.0009–0.0130 · `scripts/backtest_player_props.py` |
| League forecasting | 388 team-seasons forecast from matchday 19; Brier 0.0168 / 0.0425 / 0.0684 for champion / top 4 / relegation against 0.0489 / 0.1637 / 0.1307 base rates · `scripts/evaluate_league_forecast.py` |
| Half-time markets | 10,403 walk-forward matches; HT 1X2 RPS 0.1968 vs 0.2084 base rates, ECE 0.008–0.013 on the totals · `scripts/evaluate_half_time.py` |
| Team props — 5 event totals | 8,844 matches; beats the league base rate in 4–5 of 5 folds depending on market, ECE 0.007–0.048 · `scripts/backtest_team_props.py` |

The club rows are the strongest evidence in the project: the yardstick is a
sharp bookmaker's closing price, not a baseline.

The European layer is an Elo→λ model, and its shipped calibration was fitted on
most of the results available, so scoring it on those would be in-sample. The
script refits on the earlier seasons only and scores the held-out one (410
Champions and Europa League matches, priced with point-in-time ClubElo).
Goal levels come out close — mean λ 1.72–1.31 against 1.82–1.33 actual — while
the outcome split leans slightly wrong: 20.8% draws predicted against 17.6%
actual, 46.6% home wins against 51.5%. On one season that gap is about two
standard errors, so treat it as a lead rather than an established bias. It is
also measured on the 84% of fixtures both of whose clubs have a ClubElo history
— see [European fixture coverage](docs/EUROPEAN_ELO_COVERAGE.md).

Player props are scored on appearances, bookmaker-style, against three baselines
— a global rate, a position-group rate, and a naive career-rate model. All six
markets (anytime scorer, 2+ goals, shots 1.5/2.5, assist, yellow card) beat all
three in every one of the five temporal folds, and the calibration error stays
under 1.3% everywhere. This is the best-validated layer after the club engine.

League forecasts are scored by stopping four Big 5 seasons at matchday 19 and
simulating the rest. Leakage is controlled twice: the engine is fitted only on
matches played before that season started, and the league state only sees
results before the cutoff. Against the base rate a table gives you for free
(1/20 champion, 4/20 top four, 3/20 relegated) the Brier score is 66%, 74% and
48% lower. Mean predicted probability matches the realised rate to three
decimals in all three events, though that part is partly structural — each
simulated season awards exactly one title, four top-four places and three
relegations — so read it as a sanity check rather than as calibration evidence.

Half-time markets are a stateless transformation of the full-time lambdas — a
global first-half share, because a per-team share was measured and turned out to
be noise (r = −0.118). Fed the same walk-forward lambdas the club benchmark
uses, HT 1X2 lands at RPS 0.1968 against 0.2084 for the base rates, with the
aggregate split almost exact (0.343/0.399/0.258 predicted, 0.338/0.401/0.262
actual). The over/under gains are small — 0.005 of log loss — which is what a
global share should be expected to deliver.

The national engine is scored the same way: trained to 2022, held out on the
1,425 internationals since. RPS 0.1725 against 0.2317 for the base rates, draws
at 23.0% predicted against 22.5% actual, and home λ exact at 1.63.

Away λ is not: 1.28 predicted against 1.16 actual, roughly four standard errors,
so a real bias rather than noise. Decomposing it is more useful than the
aggregate. World Cup qualification, 63% of the sample, is well calibrated
(−0.13 home, −0.01 away), as is the Nations League. The bias concentrates in
competitions with small fits: Euro qualification runs +0.31/+0.42 and the AFC
Asian Cup +0.92/+0.94 on 51 matches. `AttackDefenseModel` fits μ and home
advantage *per competition*, and the Asian Cup gets about 115 training matches
to do it with — small-sample over-fitting, not a broken home-advantage term.
Shrinking those per-competition parameters toward the pooled fit was the obvious
next experiment; it fixes the Asian Cup but does not survive five temporal folds
(see What didn't work).

One caveat that cost a day to learn: `AttackDefenseModel` keeps a
per-competition μ and home advantage, and an unrecognised competition name falls
silently to index 0. Price an international as `"unknown"` and it is scored with
AFC Asian Cup parameters — λ near 3.0 and the home/away order inverted. Always
pass the real competition.

Team props are the weakest of the measured rows and the README should say so.
They do beat the league base rate, but by less: log-loss deltas run from −0.056
on fouls down to −0.003 on corners, and the calibration error, 0.007 to 0.048,
is several times the player-props level. **Corners specifically are marginal** —
a ~0.004 edge that survives only 4 of 5 folds. Treat those lines as barely
better than the base rate.

### Infrastructure — exercised and tested, nothing to score

These emit no probabilities of their own, so there is nothing to score the way
the rows above are scored. They run, and the test suite covers them. Player
ratings are the one partial exception: they are validated indirectly, through
the SquadLab calibration reported under In progress.

| Area | What works |
|---|---|
| Player ratings | 13,297 rated players, position-scoped roles, cross-era baseline |
| Data quality | canonical team registry, entity guardrails, leakage-safe snapshots |
| Odds layer | odds contract and readiness checks, de-vigging, value, pick policy |
| Evaluation | RPS, Brier, log loss, walk-forward backtesting, calibration search |
| Live logging | every prediction written pre-kickoff and settled afterwards |

The odds layer is dormant by choice: built while the betting question was still
open, kept because the benchmark above depends on it.

### In progress

Listed because half-finished work is normal and hiding it helps nobody.

- **SquadLab** — now a playable draft, not just a simulator. Open tiered packs,
  draft an eleven from ACTUAL / PRIME / ICON player cards (with re-rolls), pick a
  formation, and play the **real Champions League** with your side dropped into an
  actual bracket, resolved match by match on the same engine. Simulator, calendar
  and the historical all-time catalogue all work. The
  card axes are mapped per position-quantile (attack/defence/creation live on
  different scales), and the squad's rating maps onto team strength through a
  calibration that is precise on attack (R² = 0.68) and modest on defence
  (R² = 0.35): the public stats support a narrower defensive spread than an
  attacking one, so closing that gap needs better data rather than a better fit.
  Reasoning in
  [`calibration_constants.py`](src/mundialytics/statistical_core/squadlab/calibration_constants.py).
- **Competition layer** — leagues are done. Other tournament formats, and props
  aggregated over a whole competition, are not.
- **European fixture coverage** — 84% of UEFA fixtures have a ClubElo history
  for both clubs; the other 16% cannot be priced. This is **not** an alias
  problem, which was the first guess and was wrong: none of the 46 blocked clubs
  is present on disk under any spelling. They are champions of smaller
  associations — Cyprus, Israel, Bulgaria, Czechia, the Nordics — whose Elo
  histories were never downloaded, and the ClubElo API has been returning 502
  since early September (still down on 2026-09-17). Audit it with `python scripts/audit_european_elo_coverage.py`; full
  write-up in [`docs/EUROPEAN_ELO_COVERAGE.md`](docs/EUROPEAN_ELO_COVERAGE.md).
- **xG coverage** — 97% of matches since 2014/15, where Understat's data starts
  (Bundesliga 2024/25 is the notable hole, an upstream scraper bug rather than a
  missing source). Understat stopped
  publishing after 2025/26, so current-season xG now comes live from Sofascore's
  per-shot data, aggregated to match level and feeding the same rolling xG-rate
  form (`scripts/build_sofascore_xg_matches.py`; 100% of played 2026/27 matches).
- **SquadLab special cards** (award and memorable-match player variants) need
  season-split player data that isn't built yet.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest tests/                    # clean-checkout suite

# Only for the daily data refresh (brings soccerdata + Selenium):
pip install -e ".[data]"

# The UI is a web app (Next.js in web/) over the FastAPI in api/ — two processes:
uvicorn api.main:app --port 8000
npm --prefix web install
npm --prefix web run dev         # http://localhost:3000
```

The web app has pages for matchday, single-competition simulation, league
forecasting from the current table, player and team props, European
competitions, results and track record, individual awards, and SquadLab —
where you draft an eleven and play the real Champions League.

Keeping it fresh is one command — `python scripts/update_season.py` downloads new
results, rebuilds the modelling foundation (with an integrity check and rollback),
settles the logged markets and writes the upcoming round's predictions before
kick-off. `scripts/run_update.ps1` wraps it for a daily Windows scheduled task, so
results and forward predictions stay current without a manual run.

## Layout

```
src/mundialytics/
├── statistical_core/   probability engine, simulators, squadlab, competitions
├── models/             goal, result and minutes models, xG rate
├── features/           rolling team features, team match stats
├── ratings/            internal Elo, local ClubElo
├── props/              player, team and half-time markets
├── evaluation/         RPS, Brier, log loss, walk-forward backtesting
├── serving/            track record, scorer race, SquadLab rosters (for the API)
├── betting/            odds contract, de-vig, value, pick policy (dormant)
├── data/               canonical schema, provider adapters, loaders
├── data_quality/       team registry, entity guardrails, leakage-safe snapshots
├── identity/           player and team identity, current squads
├── enrichment/         xG, ClubElo and advanced-stat joins
├── providers/          ESPN fixtures, external API configuration
└── matchday/, reports/ earlier matchday builder and paper-ledger reports

api/                    FastAPI service the web app reads from
web/                    Next.js front end — see web/README.md
scripts/                ~200 CLI entry points — see scripts/README.md
tests/                  200 tests green on a clean checkout; 102 more
                        skip unless the local dataset is built
docs/                   README figures, one current note, archived history
```

## Scope and data

Research project, run in **paper mode** — no money has ever been staked on it.
Everything comes from free public sources: football-data.co.uk (results and
odds), Understat and StatsBomb Open Data (historical xG and events), Sofascore
(current-season xG, from its shot data), ClubElo, FBref, and ESPN's public JSON
(current-season fixtures and per-player match stats).

No source is trusted alone. Every one of them has failed at some point — an API
returning 502 for days, a season never published, a scraper stalling mid-fetch,
a fixture host going dark — so the fixture calendar and the player-settlement
layer each read from two providers, and cross-source agreement is a test rather
than an assumption.

Release notes: [`CHANGELOG.md`](CHANGELOG.md). The earlier design notes,
per-version specs and the long chronological README are kept, partly in Spanish,
under [`docs/archive/`](docs/README.md#archive).

## License

MIT — see [LICENSE](LICENSE).
