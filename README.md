# Mundialytics

[![tests](https://github.com/Beansent48/mundialytics/actions/workflows/tests.yml/badge.svg)](https://github.com/Beansent48/mundialytics/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A football match-prediction engine: probabilistic forecasts for 1X2, over/under,
BTTS, half-time markets and player/team props, fitted on **45,938 matches across
27 seasons (2000/01–2026/27)** of the Big 5 European leagues — England, Spain,
Italy, Germany, France — and benchmarked against bookmaker closing odds. A
second engine covers national teams; it ranks well but is not yet calibrated
(see [Status](#status)).

Built and operated solo. It runs a live forward test: every prediction is written
to [`predictions_log.csv`](data/processed/logs/predictions_log.csv) **before
kick-off** and settled against the real result afterwards, so the track record
cannot be back-fitted.

![Match view](docs/img/app.png)

*One fixture: 1X2, the full score matrix, goal markets and expected team stats —
every one of them read off the same score distribution, so they cannot disagree
with each other.*

## The headline result

Measured on **10,080 Big 5 matches, 2020/21–2025/26**, against Bet365's
**closing** odds (100% closing coverage, proportionally de-vigged). Odds are
never model inputs — they are only the yardstick.

Where the engine sits, in ranked probability score (lower is better):

| Predictor | RPS (1X2) |
|---|---|
| Uniform 1/3 — no information at all | 0.2356 |
| League base rates — home advantage only | 0.2308 |
| **This engine** | **0.2025** |
| Bet365 closing odds | 0.1946 |

Between knowing nothing and the best-informed price on the market there is
0.0362 of RPS. **The engine covers 78% of it.**

Head to head with the closing line:

| | RPS (1X2) | Log loss (1X2) | Log loss (O/U 2.5) |
|---|---|---|---|
| This engine | 0.2025 | 0.9933 | 0.6806 |
| Bet365 closing | **0.1946** | **0.9680** | **0.6710** |
| gap | +0.0080 | +0.0253 | +0.0095 |

It does **not** beat the closing line — about 4% behind, and the gap holds in
every season and every league (worst Premier League +0.0099, best Bundesliga
+0.0060), which is what makes it a ceiling rather than noise. Bet365's closing
price carries injury news, confirmed lineups and the weight of informed money;
this engine has public data and nothing else.

Reproduce all of it:

```bash
python scripts/benchmark_vs_bet365.py
```

## Is it calibrated?

Being close to the market is one question; saying 30% and being right 30% of
the time is another. It is:

![1X2 reliability diagram](docs/img/calibration_1x2.png)

Home and away curves track the diagonal across the whole range. Draws never get
predicted above ~35% — a known property of the sport, not a defect: draws are
genuinely rarely the favourite. The figure scores all 10,403 walk-forward
predictions, so its RPS reads 0.2029; the benchmark above uses the 10,080 of them
that have Bet365 odds attached. Regenerate with
`python scripts/plot_calibration.py`.

## How it works

```
football-data.co.uk ─┐
Understat / StatsBomb ├─→ canonical match schema ─→ features ─→ models ─→ markets
ClubElo / FBref      ─┘      (entity resolution)        │          │
                                                        │          ├─ 1X2 / O-U / BTTS
                              internal Elo ─────────────┤          ├─ half-time markets
                              walk-forward form ────────┤          ├─ team props
                              xG-rate predictor ────────┘          └─ player props
```

- **Goal model** — Dixon-Coles double Poisson: per-team attack/defence strengths
  fitted jointly by maximum likelihood with exponential time-decay weights, plus
  the Dixon-Coles low-score correction and an internal Elo prior.
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
  consistent by construction.

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

## Conclusions

- **The club engine works.** Calibrated 1X2 probabilities, 78% of the way from
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
| European competitions | held-out season, RPS 0.2104 vs 0.2340 base rates · `scripts/evaluate_european_layer.py` |
| Player props — 6 markets | 263,551 player-matches, beats naive and position baselines in 5 of 5 temporal folds, ECE 0.0009–0.0130 · `scripts/backtest_player_props.py` |
| League forecasting | 388 team-seasons forecast from matchday 19; Brier 0.0168 / 0.0425 / 0.0684 for champion / top 4 / relegation against 0.0489 / 0.1637 / 0.1307 base rates · `scripts/evaluate_league_forecast.py` |

The club rows are the strongest evidence in the project: the yardstick is a
sharp bookmaker's closing price, not a baseline.

The European layer is an Elo→λ model, and its shipped calibration was fitted on
most of the results available, so scoring it on those would be in-sample. The
script refits on the earlier seasons only and scores the held-out one (358
Champions and Europa League matches, priced with point-in-time ClubElo).
Goal levels come out close — mean λ 1.70–1.30 against 1.81–1.31 actual — while
the outcome split leans slightly wrong: 21.0% draws predicted against 16.8%
actual, 46.5% home wins against 51.7%. On one season that gap is about two
standard errors, so treat it as a lead rather than an established bias.

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

### Built and exercised, but not benchmarked

Each of these runs, returns sensible output, and is covered by the test suite.
None has been scored against an external reference, so read the ordering they
produce and not the exact probabilities.

| Area | What works |
|---|---|
| Half-time markets | HT result, HT over/under, HT-FT |
| Team props | totals, team-side lines, booking points |
| Player ratings | 11,063 profiles, position-scoped roles, cross-era baseline |
| Data quality | canonical team registry, entity guardrails, leakage-safe snapshots |
| Odds layer | ingestion, de-vigging, market mapping, value, staking |
| Evaluation | RPS, Brier, log loss, walk-forward backtesting, calibration search |
| Live logging | every prediction written pre-kickoff and settled afterwards |

The odds layer is dormant by choice: built while the betting question was still
open, kept because the benchmark above depends on it.

### In progress

Listed because half-finished work is normal and hiding it helps nobody.

- **National teams** — a second engine on international results from 2010. It
  *discriminates* well: trained to 2022 and scored on the 1,425 internationals
  since, RPS 0.1792 against 0.2317 for the base rates. But it is **not
  calibrated** and must not be read as a probability: goals are over-predicted
  (mean λ 2.24–2.37 against 1.63–1.16 actual), the home/away asymmetry comes out
  inverted, and draws land at 15.2% predicted against 22.5% actual. The
  home-advantage term for the national scope is the first place to look.
  Reproduce with `python scripts/evaluate_national_engine.py`.
- **Tournament simulation and top scorer** — the Monte Carlo itself is correct:
  group stage, knockout and Golden Boot all run and rank plausibly. But every
  tournament preset is a national competition, so it runs on the engine above
  and inherits its miscalibration. **Mechanically done, not calibrated** — a
  top-scorer probability from it is a ranking, not a number to trust. Fixing the
  national engine fixes this too.
- **SquadLab** — assemble a squad, real or historical, and simulate its season
  match by match. Simulator, calendar, player-rating layer and the all-time
  squad catalogue work. The calibration mapping player ratings onto team
  strength is precise on attack (R² = 0.68) and only modest on defence
  (R² = 0.35): the public stats available support a narrower defensive spread
  than an attacking one, so closing that gap needs better data rather than a
  better fit. Reasoning in
  [`calibration_constants.py`](src/mundialytics/statistical_core/squadlab/calibration_constants.py).
- **Competition layer** — leagues are done. Other tournament formats, and props
  aggregated over a whole competition, are not.
- **European name resolution** — only 80% of UEFA fixtures resolve to a ClubElo
  club, so a fifth of European matches cannot be priced at all. Alias gaps,
  mostly transliteration, and the single most mechanical fix on this list.
- **xG coverage** — ~97% of matches. Bundesliga 2024/25 is the notable hole, an
  upstream scraper bug rather than a missing source.
- **SquadLab special cards** (award and memorable-match player variants) need
  season-split player data that isn't built yet.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest tests/                    # 159 pass on a clean checkout
streamlit run app/streamlit_app.py
```

The Streamlit app has eight pages: matchday, single-competition simulation,
league forecasting from the current table, player and team props, European
competitions, results and track record, individual awards, and SquadLab.

## Layout

```
src/mundialytics/
├── statistical_core/   probability engine, simulator, squadlab, competitions
├── models/             goals, events, minutes, xG rate
├── features/           rolling team features, player baselines
├── ratings/            internal Elo, local ClubElo
├── evaluation/         RPS, Brier, log loss, walk-forward backtesting
├── props/              player and team prop markets
├── betting/            odds, de-vig, value, staking, market mapping
├── data/               canonical schema, provider adapters, loaders
├── data_quality/       team registry, entity guardrails, leakage-safe snapshots
├── identity/           player and team identity resolution across providers
├── enrichment/         xG, ClubElo and advanced-stat joins
├── providers/          external API configuration
├── matchday/           per-matchday orchestration
├── reports/            daily picks, paper ledger, match reports
└── simulation/         tournament Monte Carlo

scripts/                ~200 CLI entry points — see scripts/README.md
tests/                  159 tests green on a clean checkout; 32 more
                        skip unless the local dataset is built
docs/                   design docs and full version history
```

## Scope and data

Research project, run in **paper mode** — no money has ever been staked on it.
Everything comes from free public sources: football-data.co.uk (results and
odds), Understat and StatsBomb Open Data (xG and events), ClubElo, FBref.

Full version-by-version history: [`docs/README_FULL.md`](docs/README_FULL.md)
and [`CHANGELOG.md`](CHANGELOG.md). Design decisions:
[`docs/MODEL_DESIGN.md`](docs/MODEL_DESIGN.md) and
[`docs/DECISIONS.md`](docs/DECISIONS.md).

## License

MIT — see [LICENSE](LICENSE).
