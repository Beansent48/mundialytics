# Dockerisation and deployment: the plan, and what was built (2026-11-03)

Task **A2** of the [platform roadmap](PLATFORM_ROADMAP.md), and deliberately
**last**. Written as a plan on 2026-10-21; **built on 2026-11-03**, which is
when the measured numbers below replaced the estimates and three of the plan's
assumptions turned out to be wrong.

The user has still not decided to publish. Nothing here is deployed, no VPS was
rented and no domain was bought: the deliverable is a stack that comes up with
`docker compose up` on this machine and a documented path to a server for the
day the decision flips.

Target when it happens, unchanged: **one VPS, ~15 €/month**, 4-8 GB RAM,
everything in one `docker-compose` stack, plus a domain. No paid managed
services.

---

## 1. The thing that makes this non-trivial: the data

### What the tree actually is

Measured 2026-11-03 (`du -sm data/*`), and the first correction to this plan:

```
data/raw         23.9 GB   <-- the 2026-10-21 version of this plan missed it
data/external     2.1 GB
data/processed    481 MB
the rest            ~5 MB
                 --------
data/            26.5 GB
outputs/          7.6 GB   (ignored by git, not served)
```

The plan said 2.7 GB. It counted `data/processed` and `data/external` and left
out `data/raw` entirely — 17,256 files, mostly StatsBomb and Wyscout event
dumps from one-off studies.

### The serving bundle, measured

`scripts/measure_serving_bundle.py` installs an audit hook on `open` *before*
the app is imported, then drives a real uvicorn in a thread and calls every
route the API declares: `/health`, `/catalogue`, `/leagues`, `/competitions`,
`/track-record`, `/fixtures/day`, `/fixtures/calendar`,
`/fixtures/upcoming?days=21`, then per competition `/league`, `/awards` and
`/fixtures`, then `/competition/{slug}` for the three UEFA ones, three played
and three upcoming match pages, and the whole of SquadLab including a full
Champions season. The hook sees what the code does rather than what it says:
pandas, joblib and numpy all open through CPython's `open`.

```
17 routes exercised, 373 files read, 224.5 MB

group              files        read     of the tree on disk
data/external        240    152.0 MB     2.1 GB
data/processed        35     71.2 MB     508.2 MB
src/mundialytics      86      1.1 MB     1.8 MB     (code — in the image)
api/*.py               6    142.0 KB     —          (code — in the image)
data/curated           4     31.3 KB     46.8 KB
data/identity          1      6.2 KB     7.9 KB
web/data               1      1.9 KB     1.9 KB     (code — in the web image)
data/reference         1       533 B     1.6 KB
```

The largest readers are the ones the plan guessed right:

| Size | File | Read |
|---|---|---|
| 101.9 MB | `external/advanced/understat/understat_player_match.csv` | ×2 |
| 27.9 MB | `processed/cache/engine_clubs_*.joblib` | ×1 |
| 18.1 MB | `processed/logs/predictions_log.csv` | ×2 |
| 15.0 MB | `processed/foundation_big5_multi_season.csv` | ×1 |
| 3.6 MB | `processed/understat_team_match_xg.csv` | ×2 |

Note what is **not** there: `understat_shots.csv` (121 MB), which the plan
listed as needed. The props fit reads it when it *fits*; the serving path reads
the fitted joblib. It is a refresh input, not a serving one.

`scripts/build_serving_bundle.py` turns the manifest into the directory a
deployment mounts. It drops the code paths (they travel in the images) and
`data/processed/cache/`, because the fit cache is keyed on a hash of the engine
source and a container's copy of that source is not byte-identical to a Windows
checkout's: a copied cache could never be hit. The container pays one cold fit
on its first boot instead, which is what the healthcheck's `start_period` is
for.

### The read set was not the whole bundle, and only the container said so

The first bundle was **277 files, 194.5 MB**, and it was wrong. Running the API
against it printed:

```
market level unavailable, lambdas left alone
```

That is the fail-soft branch of `api/engine._club_engine`, so the container
served every match **with no market level at all** — probabilities quietly
different from the deployed ones, out of a bundle that looked complete. The
2026-10-17 market-level deployment is part of the published model; silently
dropping it is exactly the kind of divergence this project does not accept.

The mechanism, and it is a general trap rather than a one-off:

1. `_market_levels` memoises the level on disk **inside
   `data/processed/cache/`** — the directory the bundle deliberately excludes.
2. Its cache key is built by *stat-ing* every file in
   `data/raw/football_data/` through a glob.
3. **A glob is `os.scandir`, not `open`.** With the memo warm, the measurement
   read the memo and never opened a single football-data CSV, so they never
   entered the read set.
4. In the container the memo missed (no cache directory), the level recomputed,
   and its inputs were not there.

Any on-disk memo hides its own inputs this way. So the measurement script now
records directory listings as well as opens and prints them separately —
"listed (not every file opened)", `walked` in the manifest — which is where the
next gap of this shape will show up before a container finds it. And
`build_serving_bundle.py` has an explicit `ALWAYS_INCLUDE` for
`data/raw/football_data` with the reason written next to it.

**The bundle is therefore 583 files, 234.7 MB** — 201 football-data CSVs and
41 MB more than measured, and still **0.9% of the 26.5 GB tree.**

    python scripts/measure_serving_bundle.py --out outputs/serving_bundle
    python scripts/build_serving_bundle.py --to outputs/bundle_data --clean
    DATA_DIR=outputs/bundle_data docker compose up -d --wait api

That last line is the test the bundle exists for: a stack serving from nothing
but the measured files is what proves the list complete. A missing file is a 500
on the page that needs it, not a silent fallback.

Re-run the measurement after anything that changes what the serving path reads
— a new market, a new page, a new source.

### Two corrections that follow

**The data is a volume, never an image layer.** Unchanged, and now with a second
reason: 26.5 GB in a build context is 26.5 GB uploaded to the daemon before a
layer is built. `.dockerignore` keeps `data/` and `outputs/` out.

**The mount cannot be read-only**, which the plan assumed it could. Measured
writes while *merely serving*:

```
data/external/uefa/raw_champions-league_2026.csv      the fixture fetch rewrites
data/external/uefa/raw_europa-league_2026.csv          these on every refresh of
data/external/uefa/raw_conference-league_2026.csv      the European calendar
data/processed/competition_cache/live_*.json          league forecasts, per league
data/processed/competition_cache/scorers_*.json       the Golden Boot races
data/processed/competition_cache/uefa_*.json
data/processed/fixture_calendars/*.csv(.tmp)          A1's on-disk snapshots
data/processed/cache/engine_clubs_*.joblib            on a cold boot
data/processed/logs/                                  uvicorn's own log
```

The cache files are only rewritten when stale, so two consecutive measurements
report different write sets; the union above is what the mount has to allow.

---

## 2. Where the refresh runs — decided

The plan put this to the user as the one question a session could not answer,
and recommended (a). **On 2026-11-02 they chose (b): the refresh moves into the
stack.** The site is to be self-sufficient rather than as fresh as a laptop
being switched on.

Built as `scripts/refresh_loop.py` and the `refresh` service: a process that
sleeps until `REFRESH_AT` (10:00, local time per `TZ`), runs
`scripts/update_season.py`, and goes back to sleep. It is `run_update.ps1`'s
contract, since a container has no Task Scheduler:

- `--full` only on `REFRESH_FULL_DAY` (Monday), because the walk-forward
  regeneration behind the Resultados page is ~15 minutes;
- stdout teed to the container log, stderr filed. A `--full` run emits hundreds
  of thousands of sklearn and soccerdata warning lines, which in the container
  log would bury the summary; only their tail is printed, and only when the run
  failed;
- `last_run.json` in the same place and the same shape, so anything that reads
  it on the PC reads it here;
- `/api/revalidate` afterwards, then the lineup watcher;
- 30 runs of logs kept.

Its own image stage, because soccerdata brings Selenium along and the serving
image has no business carrying a browser driver.

Three things the Windows version did not have to solve:

- **a malformed `REFRESH_AT` must not kill the process.** With
  `restart: unless-stopped` a typo in `.env` would be a crash loop; it falls
  back to 10:00 and says so;
- **it wakes every minute** instead of sleeping the whole gap, so a suspended
  host or a clock change cannot leave it asleep past its slot, and a
  `docker compose down` is not waited out;
- **`REVALIDATE_SECRET` is effectively required.** `/api/revalidate` accepts an
  unauthorised call only from a localhost `Host` header, and from the refresh
  container the Host is `web`. Without the secret the post-refresh invalidation
  answers 401 and the new matchday waits out each page's own revalidate window.
  The container says so in its log at startup rather than leaving it to be found.

### What (b) does not move, and the cost

**The track record's git commit.** The prediction log and the drift report are
the only two refreshed files that live in git. The container has no checkout and
no credentials, and nothing in this project pushes unattended, so
`scripts/commit_track_record.py` does not run there: in the stack those files
accumulate on the data volume. Getting them into git stays a job for the machine
that holds the repository — `git pull` the volume's copy, or keep running that
one script on the PC against a synced copy. It is the one honest gap in the
user's choice and it is not a blocker: nothing on the site reads git.

**The refresh needs more than the serving bundle.** Seeding a server with the
194.5 MB bundle is enough to *serve*, and is not enough to *refresh*:
`update_season.py` rebuilds the foundation from the raw football-data CSVs and
appends to the Understat files (`understat_shots.csv`, 121 MB, among them). The
first transfer to a server therefore carries the bundle plus the refresh's own
inputs; the honest way to size that is to run the refresh container once against
the bundle and read what it says is missing, which is a step for the day the
server exists.

---

## 3. The stack, as built

| Service | Image | Notes |
|---|---|---|
| `caddy` | `caddy:2-alpine` | the only public port; `/engine/*` to the API, everything else to Next; automatic TLS once `SITE_ADDRESS` is a hostname |
| `web` | multi-stage `node:22-alpine` | `.next/standalone` only (37 MB against ~500 MB of node_modules); `next build`, never `next dev` |
| `api` | `python:3.14-slim` + `requirements-lock.txt` | code only, `data/` mounted; healthcheck with a 420 s `start_period` |
| `migrate` | the web image's `source` stage | applies the Drizzle migrations and exits; gates `web` |
| `postgres` | `postgres:17-alpine` | named volume; exists since **B1**, which is why B1 came first |
| `refresh` | the api image's `refresh` stage | the daily pass (section 2); + soccerdata |

`python:3.14-slim` and not 3.12: `requirements-lock.txt` is the exact set the
full suite last passed with, on 3.14. CI additionally runs the ranges in
`requirements.txt` on 3.10 and 3.12; the image does not have to.

### What the images weigh, and why

Measured 2026-11-03, built in 97 s (the two Python images) on this machine:

| Image | Size | Where it goes |
|---|---|---|
| `mundialytics-api` | **692 MB** | 386 MB dependency layer (numpy/scipy/pandas/sklearn), ~295 MB interpreter and base, **5.2 MB of this project's code** |
| `mundialytics-refresh` | **1.01 GB** | the api image plus soccerdata |
| `mundialytics-web` | see below | `.next/standalone` only |
| `mundialytics-web-tools` | **1.88 GB** | the `source` stage: full `node_modules` plus sources. A one-shot that runs the migrations and exits, so its size buys nothing and costs nothing |

Two numbers worth keeping. **The project's own code is 5.2 MB** of a 692 MB
image (`api/` 180 kB, `src/` 2.25 MB, `scripts/` 2.76 MB, `config/` 33 kB) —
everything else is the scientific Python stack, which is why there is no point
optimising the Dockerfile and every point in keeping the data out of it.

And **filtering three entries out of the lock saves 320 MB**: `soccerdata`
brings `seleniumbase`, which brings Selenium, PyAutoGUI, behave, pytest-xdist
and rich. The split between the `api` and `refresh` stages is that number.

### The healthcheck's `start_period`, which is load-bearing

Measured by A1 on 2026-10-21: uvicorn runs the lifespan — which loads the
models — **before** it binds the socket, so a probe during a cold start gets a
refused connection. Not a slow answer, not a 503: nothing to answer with. A
healthcheck without `start_period` therefore marks the container unhealthy,
restarts it, and the restart begins the same fit again. It never comes up.

**The first value for it was wrong, and the container said so.** 420 s was
sized from A1's 119 s host measurement; the first container actually measured
reported `models ready in 394.7s` — 25 seconds of margin, on a 22-vCPU machine.
A1's 119 s is a fit with the props joblib already warm, and a freshly seeded
volume has no caches at all: engine fit, then the market level recomputed
(28 s), then the whole props refit. On two slow vCPUs that is minutes past
420 s, the container would be marked unhealthy, and the restart would begin the
same fit again — the precise failure this section exists to prevent, caused by
sizing it from the wrong measurement.

**1200 s**: 395 s measured cold in a container, times about three for two slow
vCPUs. After it, a genuinely broken container still fails in 4 × 15 s. This is
the price of a *cold seed* and it is paid once — with `data/processed/cache`
warm the same start is seconds, and the refresh container rebuilds those caches
on its way past.

The probe checks `ready`, not just a 200. `/health` answers 200 while the models
load — it has to, since a probe triggering the load was the bug A1 fixed — and a
process in that state serves the "the model is warming up" box.

### One origin, which is how CORS stops mattering

The plan left this open with a lean towards routing everything through Next.
Auditing `web/lib/api.ts` found five client components that call the API **from
the browser** — the matchday screen, the competition browser, the awards
browser, the historic tournament and SquadLab, including the POST that plays a
Champions season — so it was a real fork.

Taken: **one origin, terminated at Caddy.** `handle_path /engine/*` strips the
prefix and forwards to `api:8000`; everything else goes to Next, its own
`/api/*` route handlers included. The API needs no public hostname, publishes no
port to the world, and is never called cross-origin.

Not through a Next route handler, which the plan called cleaner: it reaches the
same same-origin result with an extra Node hop on every client call and a
catch-all proxy to maintain, for no gain.

What that needed in the app:

- `web/lib/api-base.ts` — one place that answers "where is the API" twice,
  because the same module runs server-side and in the browser. The server uses
  `API_INTERNAL_URL` (`http://api:8000`); the browser uses
  `NEXT_PUBLIC_API_URL`, which in a deployment is a *path* (`/engine`). A
  relative base cannot work server-side: there is no origin to resolve it
  against. With neither set, both answer `http://localhost:8000` and
  development is untouched.
- `MUNDIALYTICS_CORS_ORIGINS` in `api/main.py`. The development default (any
  `localhost:*` port) must not ship, and this is the knob that settles it:
  **unset** means development, **set and empty** means no cross-origin caller
  at all (what the container sets), a **list** means exactly those. Empty is not
  unset — `tests/test_cors_origins.py` pins that down, because collapsing the
  two is how the regex would ship anyway.

### The build needs a ready API

Every data page is prerendered (`generateStaticParams` over the locales), so
`next build` renders each one against whatever the API answers at build time. A
build against a down API bakes the warming-up box into static HTML; A1 added
`web/scripts/require-api.mjs` to refuse that, and it runs inside the image
build too.

The builder has no route to the compose network, so it borrows the host's
(`build: network: host`) and reaches the port `api` publishes on loopback. That
is why the API publishes one at all. Consequence: **the first bring-up is two
moves**, and so is any deployment that rebuilds the web image.

    docker compose up -d --wait postgres api     # api is healthy = models in
    docker compose build web
    docker compose up -d

After that, `docker compose up -d` is the whole command.

### Migrations

A one-shot `migrate` service from the web image's `source` stage — which still
has `drizzle-kit`, a devDependency the standalone runner drops — running
`npx drizzle-kit migrate`, with `web` gated on
`service_completed_successfully`. So: **before the app that needs them, every
time the stack comes up.** Not in web's entrypoint, where two replicas would
race, and not by hand, which is the step everybody forgets.

### Details that would otherwise be discovered on the server

- **`name: mundialytics` is pinned in the compose file.** A1 found B1's Postgres
  invisible to `docker compose` run from the main checkout: it had been started
  from a worktree, so the project name came from the directory and the command
  exited 0 having done nothing.
- **`user: DATA_UID:DATA_GID`.** A bind mount keeps the host's ownership and the
  image's own uid (10001) owns nothing on a Linux host. Docker Desktop hides
  this completely — its file sharing makes everything writable — so on the VPS
  either set these to the owner of `DATA_DIR` or `chown -R 10001:10001 data/`.
- **`HOME` and `SOCCERDATA_DIR` point into the volume**, because the container
  may run as a uid with no home directory, and a scraper that cannot write its
  cache fails in a way that reads like a network error.
- **An empty `BETTER_AUTH_SECRET` stops the web container** with a message,
  rather than letting Better Auth fall back to a published default. As a
  `command` guard and not a `${VAR:?}`: compose interpolates the whole file
  whatever you ask it to start, so a required variable would also break
  `docker compose up -d postgres`, which is how B1's database comes up alone.
- **`TZ` on every service.** Kickoff rendering and the "today" window of
  `/fixtures/day` are date-sensitive: a container on UTC shows a 21:00 Sunday
  kickoff as Monday's match.
- **`.gitattributes` pins LF** for `*.sh`, the Dockerfiles, the Caddyfile and
  the compose file. This machine has `core.autocrlf=true` system-wide, and a
  shell script whose shebang ends in CR fails on the server with "bad
  interpreter".

### What did not work, so nobody tries it again

- **A read-only `data/` mount**, which this plan assumed. Measured impossible
  (§1): serving a page writes. A middle road — read-only plus a writable volume
  over `data/processed/cache` — does not survive either, because the UEFA
  fixture fetch rewrites `data/external/uefa/raw_*.csv` on the way to answering
  `/fixtures/day`.
- **`python:3.12-slim` with `requirements-lock.txt`**, which is what this plan
  specified. The lock is the set the suite last passed with **on 3.14** and its
  own header says some of those pins do not exist for 3.12, so it would have
  meant installing ranges and losing the only reproducible dependency set the
  project has. The image is 3.14; CI keeps testing the ranges on 3.10 and 3.12.
- **Installing the whole lock in the serving image.** It pulls `soccerdata`
  (and therefore Selenium), `matplotlib` and `pytest` into a container that
  serves JSON. The serving stage installs the lock minus those three and the
  refresh stage adds them back at the same versions; checked against the read
  set, which shows none of the 92 modules the serving path imports touching any
  of them.
- **A web image that builds without a running API.** The alternative to the
  two-move bring-up is to stop prerendering the data pages, which would undo
  A1's measured 4-48 ms routes and make every page render per request. Not
  worth avoiding one extra command in a deployment.
- **Proxying the API through a Next route handler**, which this plan called the
  cleaner of the two options. It reaches the same same-origin result as the
  Caddy path does, with a Node hop on every client call and a catch-all handler
  to maintain. Rejected on those grounds, not tried.

---

## 4. Operations, minimal but real

- **Backups.** `docker/backup-postgres.sh`: nightly `pg_dump | gzip`, 14 days
  kept, written to a `.part` and renamed so an interrupted dump is never
  mistaken for a backup, credentials read from the same `.env` the stack uses.
  One crontab line, which the script carries, at 04:30 — after the 10:00
  refresh is long done and well before the next one. Postgres is the only thing
  that needs this: `data/` is reproducible from the refresh, `outputs/` is not
  served, the images rebuild from the repository, and the accounts are the only
  bytes in the deployment that exist nowhere else. The restore command is in
  the script, with the warning that it goes into an *empty* database.
- **CI.** The `images` job builds the API image whole and the web image up to
  `source`, and checks that the compose file parses. It cannot build the web
  image's final stage: that runs `next build`, which needs a ready API, which
  needs ~195 MB of data that git does not carry. The `build` stage is exercised
  on the machine that has the data.
- **Logs** go to stdout, so `docker compose logs -f` is the one place to look.
  uvicorn also keeps writing `data/processed/logs/uvicorn.log`, which is now a
  file on the volume rather than the only copy.
- **Sizing.** One uvicorn worker on purpose: a second would hold its own copy of
  the fit (29 MB of joblib plus the foundation in pandas) and pay its own cold
  start. 4 GB is enough to serve; option (b) puts the refresh on the same box,
  so **8 GB** is the number to budget for — the `--full` pass fits models over
  46k matches.

---

## 5. The open questions, as they stand

| Question | State |
|---|---|
| Does the refresh stay on the PC (a) or move to the server (b)? | **Decided 2026-11-02: (b).** Built; section 2 has what it costs. |
| Is the site to be indexed? | **Decided 2026-11-02: not yet.** `robots.txt` disallows and the pages send `noindex`; `SITE_INDEXABLE=1` flips both. |
| Domain name, and which registrar? | **Open.** Naming and payment, so it stays the user's. Nothing was bought. Once it exists: point an A record at the VPS, set `SITE_ADDRESS` to the hostname and `BETTER_AUTH_URL` to `https://<host>`, and Caddy gets the certificate on the next start. |
| Provider? | **Open.** Hetzner is the cheapest at this size (a CX22, 2 vCPU / 4 GB, is about 4 €/month; 8 GB about 8 €), which leaves room in the 15 € for the domain and the backups. Nothing was rented. |

### Going to a server, when that day comes

1. A VPS with Docker and compose; a user who owns the data directory.
2. `git clone`, then `cp .env.example .env` and fill `BETTER_AUTH_SECRET`,
   `REVALIDATE_SECRET`, `DATA_UID`/`DATA_GID`, `SITE_ADDRESS`, `ACME_EMAIL`,
   `BETTER_AUTH_URL`.
3. Seed `DATA_DIR`: the serving bundle (194.5 MB) plus the refresh's inputs
   (section 2) — `rsync -a` from this machine.
4. `docker compose up -d --wait postgres api`, then `docker compose build web`,
   then `docker compose up -d`.
5. The crontab line from `docker/backup-postgres.sh`.
6. Open the site. The first visitor after a restart may wait for a cold fit;
   the `start_period` is there for exactly that.

Updating it later is `git pull`, then the same two-move sequence (the web image
has to be rebuilt because its pages are prerendered), then
`docker compose up -d`.

---

## 6. What the containers proved, and what they corrected

The stack was built and run on 2026-11-03/04. That mattered more than expected:
**six defects turned up that reading the code had not**, and not one of them
raised an exception. Every one left the system answering with a credible number,
which is the only reason they had survived.

| # | What was wrong | How it looked |
|---|---|---|
| 1 | the measured bundle could not recompute the market level (41 MB of inputs hidden behind a memo keyed by `scandir`) | one log line, every match priced with no market shift |
| 2 | `_market_levels` **memoised that failure** under a key describing the missing files | later starts "succeeded" in under a second, still with no level |
| 3 | `start_period` 420 s against a 394.7 s and then 544.1 s cold start | 25 s of margin; on a slow VPS, a restart loop |
| 4 | the API image excluded `web/data/benchmark_vs_bet365.json`, which the API reads | `/track-record` served `benchmark: null` |
| 5 | `web/lib/api.ts` typed that nullable field as non-nullable | `/en/results` threw during prerender: **the web image could not be built** |
| 6 | the props bundle was missing `understat_shots.csv`, tolerated by `if ... .exists()` | both prop models load, every price plausible, the pen-taker split gone |

Numbers 2, 5 and 6 are product defects rather than packaging ones, and they are
in this document because the pattern is the point: a cache that stores a
failure, a type that denies a null the server really sends, and an input whose
absence is swallowed by an `exists()` check.

### What the running stack does

```
docker compose ps        api/web/postgres/caddy/refresh healthy, migrate exited 0
all-healthy, caches warm                      21 s   (models ready in 2.0s)
all-healthy, cold seed                 394.7 / 544.1 s
market level, in-container      Bundesliga x1.0257 · LaLiga x1.0451 · Ligue 1
                                x1.0316 · Premier League x1.0369 · Serie A x1.0546
/engine/health            200 through Caddy, on the site's own origin
/engine/fixtures/day      200, today's card priced
/ , /es, /es/{results,matchday,leagues,squadlab,awards,competitions}
                          200, 58-111 KB of real content each
/robots.txt               200, 27 bytes — the Disallow
sign-up through the proxy 200, user row created, session cookie resolves
migrations                applied by the one-shot, exit 0, gating web
```

`/en` answers 307 to `/`, which is `localePrefix: "as-needed"` with
`defaultLocale: "en"` doing what it is configured to do, not a proxy fault.

### Two local traps worth knowing

- **`DATA_DIR` belongs in `.env`, not on the command line.** Passing it to one
  compose call and not the next changes the service's configuration, so compose
  silently *recreates* the container against the other directory. That is how
  defect 2 was produced: a container that mounted the wrong tree wrote the
  poisoned memo.
- **Docker Desktop does not follow NTFS junctions.** A worktree whose `data/` is
  junctioned to the main checkout (which is how a worktree gets the 26.5 GB at
  all) mounts as a tree where `data/external` and `data/raw` exist and cannot be
  traversed. So on this machine the serving-bundle copy is not an optimisation,
  it is the only mountable form of the data.

### Still not done

- **The refresh container has not run a refresh.** It starts, schedules and
  logs; `REFRESH_ON_START=1` against a real season download was not run, and
  the bundle is a serving bundle, so it would be the wrong input anyway
  (section 2). That is the one piece of A2 still owed a measurement.
- Nothing is deployed. No VPS, no domain, no push of an image anywhere.
