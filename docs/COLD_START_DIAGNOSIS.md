# Cold start: what is actually slow (2026-10-21)

Task **A1** of the [platform roadmap](PLATFORM_ROADMAP.md). The complaint is
that the web and the API take too long; asked where, the answer was **the cold
start** — from launching the two processes to the first page being usable.

The first half of this document is the diagnosis as it was written, before
anything was applied. The [results](#what-the-stopwatch-found) at the end are
the before/after table, measured with `scripts/measure_cold_start.py`, and they
contradict two of the guesses below — which is what a measurement is for.

## The one thing to get out of the way first

The Python side is not slow. Measured on this machine, with the joblib fit
cache warm (the state the daily refresh leaves it in):

```
import pandas              0.79s
import fastapi+uvicorn     0.52s
import prediction_engine   1.55s
load_clubs_data            0.24s  (46,186 rows)
joblib.load engine         0.60s  (29 MB)
joblib.load props          0.06s
TOTAL                      3.76s
```

Three and a half seconds, not three minutes. So the headline in
`api/main.py` — "the first request pays the model fit (~3 min)" — describes the
**cache-miss** path, and the cache is warm almost all the time. Which leaves
four suspects, in order of how much they can plausibly cost:

> **This total was wrong, by a factor of nine.** It times `joblib.load`
> directly, and `club_engine()` does one more thing after the fit comes back off
> disk: `market_level_detail`, the 2026-10-17 market-level deployment, which is
> **30.8 s**. Measured step by step, `club_engine()` on a warm joblib cache is
> 35 s, not 3.8 s. See [the correction](#correction-the-python-path-is-35-s-not-38-s)
> below.

### 1. The API pays those 4 seconds on the first *request*, not at startup

There is no `lifespan` or startup hook anywhere in `api/` (grep for
`lifespan|on_event|startup`: nothing). `club_engine()` is an `lru_cache`d
function called lazily from the endpoints. So uvicorn prints its ready line
while holding nothing, and the first visitor waits for imports, the data load
and a 29 MB `joblib.load` — and so does the *second* visitor, concurrently,
because nothing is holding a lock.

### 2. `next dev` compiles each route the first time it is visited

`.claude/launch.json` and the README both start the web with `npm run dev`.
Next.js 16 in dev compiles per route on demand: the first visit to
`/es/matchday` compiles that route, the first visit to `/es/squadlab` compiles
that one. There is no production build anywhere in the loop, so this cost is
paid again after every restart, on every page, for ever. This is the suspect
with the highest prior and the cheapest fix.

### 3. The refit, when it does hit

The joblib key folds in the row count, the last date, the engine code
fingerprint, `squad_values.csv`'s mtime and an xG digest (`api/engine.py`). Any
change to engine code invalidates it, and then the next first request pays a
~3-minute fit. That is correct behaviour — a stale fit would serve numbers that
no longer match the logged ones — but it means **a session that touches the
engine makes the next page load three minutes long**, and the web client gives
up at 20 s.

### 4. The 20-second client abort turns slowness into errors

`web/lib/api.ts` aborts every call at 20 s. This has already bitten once: the
docstring of `scripts/warm_api_caches.py` records the league page rendering as a
404 and the awards page as an error because the first visitor after a refresh
paid a 35-second recomputation. Slow is one failure; *wrong* is another, and the
second is worse.

## Measured the same day: 83 seconds to the first matchday answer

The user then reported that it is specifically **matchday information** that takes
minutes. Measured end to end — uvicorn on a spare port, nothing warmed, the first
requests being the two the matchday screen actually fires from the browser
(`web/components/matchday/matchday-client.tsx` calls `api.fixturesCalendar()` and
`api.fixturesDay(selected)`):

| Request | Cold | Warm |
|---|---|---|
| `/fixtures/calendar?days_back=14&days_forward=21` | **83.5 s** | 4.4 s |
| `/fixtures/day` (in parallel, as the page does) | **83.4 s** | 3.6 s |
| `/fixtures?competition=laliga` | — | 0.07 s |
| `/fixtures/upcoming?days=21` | — | 8.5 s |

Three findings behind that number.

**The fit cache is not to blame.** The key the API computes right now matches the
file on disk (`engine_clubs_46186_2026-10-19_749e24c9_1792571960_42694x60782.6`),
so no refit is happening. Confirmed, not assumed.

**The calendar is fetched live from ESPN on every cold start, and nothing
persists it.** `cat.season_calendar` → `fetch_season_fixtures`
(`src/mundialytics/providers/espn_fixtures.py`) has **no disk cache at all**:
two HTTP calls per league-season (ESPN stopped accepting a date range in
September 2026, so a July-June season needs `dates=2026` *and* `dates=2027`,
`limit=1000` each). Measured today, with ESPN healthy:

```
laliga           2.32s   380 fixtures
premier-league   2.00s   380 fixtures
serie-a          2.04s   380 fixtures
bundesliga       1.90s   306 fixtures
ligue-1          2.15s   306 fixtures
TOTAL           10.41s   (10 HTTP calls, no disk cache)
second call      0.0000s  (in-process lru_cache, 30-minute bucket)
```

There is no fixture snapshot anywhere on disk. `data/processed/competition_cache/`
holds *forecast* caches, and `espn_matches_current.csv` holds played matches —
neither is the calendar. So the whole 10 s is paid again after every restart, and
on a bad ESPN day the ceiling is `45 s × 10 calls` with no retry budget: minutes,
which is exactly the complaint.

**Concurrent cold requests duplicate all of it.** `lru_cache` does not
de-duplicate calls in flight, and FastAPI runs sync endpoints in a threadpool. The
two requests the page fires together each loaded the engine and each fetched all
five calendars — 14 s of work became 83 s of contention. This is why the cold
number is six times the sum of its parts.

## The fix the numbers pointed at: persist the calendar

This is the user's own proposal (2026-10-21: "keep the current state in some
store and only fetch what is new — have the minimum needed to start instantly"),
and the measurement above said they were right. It came **before** every
candidate on the original shortlist, and it is what was built first.

The shape, and why it is a file and not a database:

- **The daily pass already downloads this.** `scripts/update_season.py` runs at
  10:00 and talks to ESPN anyway. It should write the five league calendars to
  disk — one CSV per league-season, ~380 rows each, tens of kilobytes — and
  `cat.season_calendar` should read that file. ESPN stops being the serving
  source and becomes the refresh source, exactly as the UEFA calendar already
  works (ESPN data backbone, September 2026: "the cache becomes the fallback,
  not the source").
- **A file, not Postgres.** One writer, read-mostly, no queries beyond "this
  league, this season", and the project's whole invalidation idiom is already an
  mtime (`data_version()` in `api/engine.py`). A database service would add a
  container and a migration to protect 1,750 rows that change once a day. The
  Postgres arriving in **B1** is for *user* data; the football snapshot stays
  files, which also keeps the API container database-free in **A2**.
- **Stale-while-revalidate, never block a visitor.** Serve the snapshot
  immediately, and refresh in the background when it is older than the window
  (the current 30-minute `lru_cache` bucket becomes a background refresh). A
  live score during a matchday is the only case that needs minutes-fresh data,
  and even then the right answer is to re-fetch *behind* a served response.
- **Single-flight.** Whatever stays live must be de-duplicated: two concurrent
  cold requests each did the whole job. A lock around the loader turns the worst
  case back into one cost instead of N.
- **The failure mode gets better, not just the speed.** With a snapshot on disk,
  ESPN being down or slow stops being visible at all; before it was the
  difference between 10 s and minutes, and `fetch_season_fixtures` swallowing
  the error returns an *empty* calendar, which is how a matchday renders blank.

All five were built as described. What the measurement added to the plan: the
same single-flight argument applies one level up, to the payload cache and to
the model load, and the day and calendar payload caches were additionally
*self-invalidating* — see the warm table below.

## What the stopwatch found

Everything below was measured with `scripts/measure_cold_start.py`, on this
machine, on 2026-10-21, with the joblib fit cache warm unless a row says
otherwise. "Cold" is one freshly spawned uvicorn per endpoint: `boot` is the
spawn-to-port-accepting time, `first` is the request on that fresh process, and
`cold` is the sum — what a visitor actually waits for. Uvicorn runs the lifespan
*before* it binds the socket, so work moved into startup leaves `first` and
lands in `boot`.

### Correction: the Python path is 35 s, not 3.8 s

The 3.76 s at the top of this document times `joblib.load`. `club_engine()`
does one more thing, and it is the biggest single item in the whole cold start:

```
imports (pandas, fastapi, prediction_engine)   3.1 s
load_clubs_data (46,186 rows)                  0.2 s
joblib.load engine (29 MB)                     0.8 s
market_level_detail(df, None, k=70)           30.8 s   <-- missing from the diagnosis
props_models (joblib)                          0.2 s
                                              ------
                                              ~35 s
```

`market_level_detail` re-prices every logged lambda of the season against the
book's over-2.5 line, fourteen Newton steps per match. It is deliberately
**not** baked into the engine's joblib — `api/engine.py` explains why: a level
frozen into a fit would be served long after the matches it was measured on — so
it ran on every process start and again after every daily refresh, and the fit
cache could not help it. A full cache MISS (no joblib at all) is **119 s**, not
the "~3 min" the docstrings claimed.

### The API, cold: one fresh process per endpoint

| Endpoint | Before `cold` | After `cold` | After `boot` | After `first` |
|---|---|---|---|---|
| `/health` | 39.0 s | **5.4 s** | 5.3 | 0.05 |
| `/catalogue` | 33.2 s | **5.3 s** | 5.3 | 0.05 |
| `/fixtures/day` | 55.1 s | **14.0 s** | 6.4 | 7.7 |
| `/fixtures/calendar?days_back=14&days_forward=21` | 53.4 s | **11.8 s** | 7.9 | 3.9 |
| `/fixtures/upcoming?days=21` | 56.3 s | **27.9 s** | 7.4 | 20.5 |
| `/fixtures?competition=laliga` | 41.2 s | **9.3 s** | 9.0 | 0.33 |
| `/match/{comp}/{fixture}` | 31.5 s | **8.0 s** | 7.4 | 0.63 |
| `/league/laliga` | 29.2 s | **9.1 s** | 9.0 | 0.12 |
| `/awards/laliga` | 28.2 s | **13.3 s** | 13.2 | 0.10 |
| `/track-record` | 46.5 s | **20.0 s** | 6.9 | 13.2 |
| `/leagues` | 31.3 s | **5.4 s** | 5.3 | 0.08 |
| `/competitions` | 3.8 s | 5.9 s | 5.8 | 0.05 |
| `/squadlab/pool` | 3.7 s | 5.4 s | 4.8 | 0.61 |
| `/health`, **joblib cache deleted** | 118.9 s | 123.3 s | **123.2** | **0.06** |

Read the last row first: it is the point of the whole exercise. The two-minute
refit did not get faster — it moved. Before, `boot` was 3.7 s and the first
request was 115 s, past the web client's 20 s abort, which is how a slow start
became a 404. After, the refit is 120 s of `boot`, announced in the log
(`models ready in 119.6s`), and the first request is 60 milliseconds.

Two rows got slower, and honestly so: `/competitions` and `/squadlab/pool` read
only disk and never needed the engine, so they used to answer in 0.03 s on a
process holding nothing. Now they wait for the startup that every other
endpoint benefits from. Paying 2 s on two endpoints to take 25 s off the other
eleven is the trade, and it is the right way round.

`boot` varies between 4.8 s and 13.2 s in the table. The lifespan itself is
consistent — `models ready in 1.6s` to `2.0s` in every log — and the spread is
Python's own imports competing with the background payload warm-up and with the
teardown of the previous process in the sweep. A quiet start is about 5.3 s.

### The API, warm: what each endpoint costs once the models are loaded

`first` is the first call on an already-warmed process; `cached` is the median
of three repeats.

| Endpoint | Before `first` | Before `cached` | After `first` | After `cached` |
|---|---|---|---|---|
| `/health` | 0.00 s | 0.013 s | 0.03 s | 0.004 s |
| `/catalogue` | 0.03 s | 0.031 s | 0.04 s | 0.025 s |
| `/fixtures/day` | 12.82 s | **6.67 s** | 5.31 s | **0.051 s** |
| `/fixtures/calendar` | 6.45 s | **3.58 s** | 0.19 s | **0.049 s** |
| `/fixtures/upcoming?days=21` | 15.39 s | 0.005 s | 13.60 s | 0.007 s |
| `/fixtures?competition=laliga` | 0.14 s | 0.134 s | 0.15 s | 0.137 s |
| `/match/{comp}/{fixture}` | 0.39 s | 0.031 s | 0.25 s | 0.043 s |
| `/league/laliga` | 0.02 s | 0.009 s | 0.01 s | 0.009 s |
| `/awards/laliga` | 0.03 s | 0.011 s | 0.06 s | 0.056 s |
| `/track-record` | 12.29 s | 0.005 s | 10.49 s | 0.004 s |
| `/leagues` | 0.11 s | 0.092 s | 0.07 s | 0.059 s |
| `/squadlab/pool` | 0.54 s | 0.140 s | 0.46 s | 0.141 s |

The two bold columns are a bug, not a slow path. `/fixtures/day` and
`/fixtures/calendar` **never hit their own cache**: `_cached`'s key includes
`european.fixtures_fingerprint()`, the mtimes of the UEFA `raw_*.csv` files, and
building the payload calls `european.fetch_season_fixtures`, which refetches
from ESPN and **rewrites those same files**. Computing the value moved the key
that was supposed to find it. Three identical requests left three cache entries
and paid ~4 s for each. Verified directly: the fingerprint changes between two
consecutive calls with no new data.

### The web: `next dev` against `next build && next start`

Both measured against an already-warm API, so these are Next.js numbers. First
visit, then the median of three repeats.

| Route | `next dev` first | `next dev` second | `next start` first | `next start` second |
|---|---|---|---|---|
| `/` | 3.94 s | 0.179 s | 0.10 s | 0.008 s |
| `/matchday` | 1.11 s | 0.136 s | 0.01 s | 0.006 s |
| `/leagues` | 1.15 s | 0.160 s | 0.01 s | 0.006 s |
| `/leagues/laliga` | 1.44 s | 0.273 s | 0.19 s | 0.034 s |
| `/competitions` | 0.98 s | 0.137 s | 0.01 s | 0.005 s |
| `/results` | 12.99 s | 0.180 s | 0.03 s | 0.004 s |
| `/awards` | 1.24 s | 0.125 s | 0.01 s | 0.004 s |
| `/squadlab` | 1.04 s | 0.129 s | 0.01 s | 0.004 s |
| `/es` | 0.22 s | 0.158 s | 0.03 s | 0.003 s |
| server boot | 1.6 s | — | 1.1 s | — |
| `next build` | — | — | 25 s (35 s with `standalone`) | — |

**This is the suspect the diagnosis got wrong.** It called `next dev`
compilation "the suspect with the highest prior". It is real — 1 to 4 s per
route, paid again after every restart — but the API was costing twenty-five
seconds on the same page. Production mode is still worth having: it is a 25 s
build against 1-4 s per route per restart, and it turns every page into
milliseconds.

`/results` at 12.99 s in dev is not Next.js: the page server-renders
`/track-record`, and that call was cold.

### The fan-out: candidate 5 is discarded

A full LaLiga round, priced:

```
matchday 10: 10 fixtures, 10 _predict calls, all 10 priced
first call 101 ms (the ESPN season calendar), the rest 88 ms median
0.91 s for the whole round, of a 0.91 s render
```

88 ms per `_predict`, not the 20 ms the diagnosis guessed — but 0.91 s per round
is noise beside 35 s, and batching it would be work with nothing to show. Then
the obvious follow-up: where do those 88 ms go?

```
cat.season_calendar                       0.19 ms/call
set(home) | set(away) over the calendar   1.28 ms/call
predict_match_or_proxy                   94.17 ms/call
_predict (whole thing)                   95.09 ms/call
```

All of it is `predict_match_or_proxy` — model code, which this task may not
touch. So the fan-out cannot be made cheaper per fixture at all; the only lever
is not making a visitor wait for it, which is what the startup warm-up does.
That is why `/fixtures/upcoming?days=21` is still 13.6 s warm-first and 0.007 s
once cached: it prices every unplayed fixture in three weeks, around 140 of
them, at 95 ms each.

## What was done

In the order the measurement justified, not the order the candidate list had.

1. **The league calendar is served from disk** (`api/catalogue.py`). One CSV per
   league-season under `data/processed/fixture_calendars`, ~380 rows and 16 KB,
   written through a temporary file and swapped in. A fresh snapshot is served
   outright; a stale one is served too and refreshed in a background thread;
   only a league with no snapshot at all blocks, and then at most once per
   window however often the provider fails. An empty provider answer never
   replaces a snapshot — `fetch_season_fixtures` returns an empty frame for
   every failure, which is how a matchday renders blank. The daily pass is the
   writer: `scripts/warm_api_caches.py` already talks to ESPN in step 7m, and it
   now reports a league whose snapshot it could not refresh.
2. **A `lifespan` that loads the models before uvicorn binds its port**
   (`api/main.py`). Keyed on `data_version()` exactly as before, so warming is
   not process-lifetime state and the daily refresh still reaches a running API.
3. **A disk memo for the market level** (`api/engine.py`), keyed on everything
   `market_level_detail` reads: its own source fingerprint, `k`, the
   foundation's size and last date, both lambda logs and every football-data
   file. `club_engine()` goes from 35 s to 1.6 s. The daily refresh warms it on
   the way past, so the API start after a refresh reads it off disk.
4. **Single-flight, with a fast 503 instead of a queue.** One caller loads the
   models and anything arriving meanwhile gets `503` with `Retry-After: 5`; one
   caller builds each payload-cache entry; one caller refreshes each calendar.
   Four tabs on a cold API used to wait four times for the same work.
5. **A half-hour cache for the European state** (`api/european.py`), which was
   not cached at all — 1.2 s per competition per call — and which is what made
   the day and calendar payload caches self-invalidating.
6. **A local production mode**: `npm --prefix web run prod`, plus a
   `mundialytics-web-prod` launch configuration. `prod` refuses to build while
   the API is down, because every data page is SSG and a build without an API
   bakes the "model is warming up" box into the static HTML — measured, not
   theorised: /en/leagues built to 52 KB of warming notice against 68 KB of real
   standings.
7. **`output: "standalone"`** in `web/next.config.ts`. 37 MB against ~500 MB of
   `node_modules`, for the A2 image. Not free after all: the build goes from
   25 s to 35 s for the file tracing. Serving is unaffected.
8. **The matchday screen retries by itself** (`web/lib/matchday-state.ts`).
   It fetches from the browser, so a cold API left the panel on "could not
   load" until the tab regained focus. Now every five seconds for two minutes.

### What was discarded, and why

- **Batching the matchday predictions** (candidate 5). Measured at 0.91 s for a
  ten-fixture round, and 94 of the 95 ms per fixture are inside
  `predict_match_or_proxy`, which is model code this task may not touch. There
  is nothing to batch: the cost is not per-call overhead.
- **A "warming up" page state** (part of candidate 4). The pages already have
  the copy — `offlineTitle` is literally "The model is warming up" — and in
  production they are prerendered, so they do not call the API at request time
  at all. What was missing was not a state but a recovery: the API now fails
  fast with a 503 and the browser-side panel retries. Adding a second visual
  state would have been new copy in two locales for a case the existing box
  already describes.
- **Making the fit cache stickier.** Explicitly out of bounds, and unnecessary:
  the 119 s refit is correct behaviour and moving it into `boot` was enough.

### What was not fixed

- `/fixtures/upcoming?days=21` is 13.6 s on a warm process the first time. That
  is ~140 fixtures at 95 ms of model code each. The startup thread warms it, so
  a visitor normally gets the cached 0.007 s, but someone who opens the
  favourites tab in the first twenty seconds after a restart waits for it.
- `/track-record` is 10.5 s warm-first for the same reason, and it is warmed
  last because the matchday payloads come first.
- `boot` is 5.3 s of which 3.1 s is `import pandas` and friends. Nothing here
  addresses that.
- The `_CACHE` dict in `api/main.py` still has no eviction. It was unbounded
  before and it is unbounded now; the entries are a few kilobytes each.

## The constraints, and how they were checked

- **The served numbers must not change.** 31 payloads were captured from the
  endpoint functions before any change and again after all of them — 15 match
  pages across the five leagues (upcoming and played), the five rounds, the day,
  the calendar, the three-week fixture list, the standings, the five league
  forecasts, the five scorer races, the catalogue and the track record. All 31
  are **byte-for-byte identical**. The market-level multipliers printed at
  startup are unchanged to four decimals (LaLiga x1.0516, Premier League
  x1.0394, Serie A x1.0401, Ligue 1 x1.0463, Bundesliga x1.0242).
- **The joblib key must keep invalidating on data and code.** Untouched. The new
  market-level memo follows the same discipline and `tests/test_cold_start_warmup.py`
  pins it: a new logged lambda, new foundation rows, a different `k` and a new
  football-data file each recompute it.
- **The daily refresh must still reach the site without a restart.** `_ready()`
  is keyed on `data_version()` and reloads when it moves;
  `test_a_refresh_reloads_rather_than_serving_the_old_fit` pins that warming at
  startup did not turn the models into process-lifetime state.
- **`pytest tests/` stays green**, and the README's "Running it" section
  describes the new way to start it.

### A trap for anyone working in a worktree of this repo

Git for Windows ships `core.autocrlf=true` in its **system** config, and this
repo's blobs already hold CRLF. A fresh `git worktree add` therefore writes
CR-CR-LF, the engine sources differ from the main checkout byte for byte, and
`engine_code_fingerprint()` differs with them (`de4e09f3` here against
`749e24c9` in the main checkout). The joblib key misses — and `api/engine.py`'s
cleanup glob then **deletes the other checkout's warm fit**. It cost the main
checkout one 119 s refit before it was noticed. If a session links `data/` into
a worktree, give it its own `data/processed/cache`.
