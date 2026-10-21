# Cold start: what is actually slow (2026-10-21)

Task **A1** of the [platform roadmap](PLATFORM_ROADMAP.md). The complaint is
that the web and the API take too long; asked where, the answer was **the cold
start** — from launching the two processes to the first page being usable.

This is the measurement that has to come before any fix, plus the shortlist of
candidates. **Nothing here is applied yet.**

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

## What to measure next

Record the numbers in this file as a table, then fix, then record them again.

1. **Time to first usable page, by route**, for `next dev` vs
   `next build && next start`. Walk every route in `web/app/[locale]/`
   (`/`, matchday, leagues, competitions, results, awards, squadlab) and time
   the first visit and a second visit.
2. **The API's own latency, endpoint by endpoint**, on a cold process and on a
   warm one. `/health`, `/catalogue`, `/fixtures/day`, `/fixtures/calendar`,
   `/round`, a match page, `/league`, `/awards`, the SquadLab endpoints.
3. **The per-page fan-out.** `_fixture_payload(..., with_probs=True)` calls
   `_predict` once per fixture and each call reaches `cat.season_calendar`
   (30-minute `lru_cache`, ESPN fetch on a miss). Count the calls a single
   matchday render makes and time them together.
4. **The cache-miss path**, once, deliberately: delete the engine joblib and
   time the first request. That is the worst case a visitor can hit, and it is
   what any "warming" UX has to cover.

## The fix the numbers point at: persist the calendar

This is the user's own proposal (2026-10-21: "keep the current state in some
store and only fetch what is new — have the minimum needed to start instantly"),
and the measurement above says they are right. It comes **before** every
candidate in the next section.

The shape, and why it is a file and not a database:

- **The daily pass already downloads this.** `scripts/update_season.py` runs at
  10:00 and talks to ESPN anyway. It should write the five league calendars to
  disk — `data/processed/fixture_calendar.parquet` or one CSV per
  league-season, ~380 rows each, tens of kilobytes — and `cat.season_calendar`
  should read that file. ESPN stops being the serving source and becomes the
  refresh source, exactly as the UEFA calendar already works (ESPN data backbone,
  September 2026: "the cache becomes the fallback, not the source").
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
- **Single-flight.** Whatever stays live must be de-duplicated: today two
  concurrent cold requests each did the whole job. A lock around the loader turns
  the worst case back into one cost instead of N.
- **The failure mode gets better, not just the speed.** With a snapshot on disk,
  ESPN being down or slow stops being visible at all; today it is the difference
  between 10 s and minutes, and `fetch_season_fixtures` swallowing the error
  returns an *empty* calendar, which is how a matchday renders blank.

What this does **not** cover: the engine fit (already a 0.6 s joblib snapshot)
and `next dev` compiling routes. Those are the other two legs, below.

## Candidate fixes

In the order they should be tried, cheapest and safest first — after the
calendar snapshot above, which is now the first job.

1. **A `lifespan` that warms the engine at startup.** Load `club_engine()` and
   `props_models()` before uvicorn reports ready. Costs ~4 s of boot, removes
   them from the first visitor, and on a cache miss it moves a 3-minute wait
   from a page load to a startup log line where it belongs.
2. **A local production mode.** A `prod` script (`next build && next start`) and
   a second `.claude/launch.json` configuration, so day-to-day use is a
   precompiled build and `dev` is only for editing the web app. Measure the
   build time too: it is paid once per change, not once per page.
3. **`output: "standalone"` in `next.config.ts`.** Does nothing for speed; makes
   the Docker image in **A2** small and is free to add now.
4. **Fail fast instead of hanging while warming.** If a request arrives while
   the models are still loading, answer 503 with a retry hint rather than
   blocking for three minutes, and let the page render a "warming up" state. The
   20 s abort then becomes a visible, honest state instead of a 404.
5. **Batch the matchday predictions** if the fan-out measurement justifies it —
   and only then. One round is ~10 fixtures; if each `_predict` is 20 ms this is
   noise and should be left alone.

## Hard constraints for this task

- **The served numbers must not change.** This is a latency task. The deployed
  baseline rule applies in full: no engine kwargs, no model code, no
  recalibration. A before/after check that `/match/...` returns byte-identical
  probabilities is part of the work, not an optional extra.
- **The joblib key must keep invalidating on data and code.** Making the cache
  stickier to save time would serve numbers that disagree with the pre-kickoff
  log, which is the one thing the project cannot afford.
- **The daily refresh must still reach the site without a restart.** The
  `data_version()` mtime key in `api/engine.py` is what makes that work; a
  warm-at-startup change must not turn the models into process-lifetime state
  again.
- `pytest tests/` stays green, and the README's "Running it" section has to
  describe whatever the new way to start it is.
