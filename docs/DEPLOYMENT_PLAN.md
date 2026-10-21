# Dockerisation and deployment: the plan (2026-10-21)

Task **A2** of the [platform roadmap](PLATFORM_ROADMAP.md), and deliberately
**last**. The user has decided not to publish yet; this exists so that when the
decision flips, nobody starts from a blank page — and so that **A1** and **B1**
leave the right things behind.

Target when it happens: **one VPS, ~15 €/month**, 4-8 GB RAM, everything in one
`docker-compose` stack, plus a domain. No paid managed services.

---

## 1. The thing that makes this non-trivial: the data

There is no 100 MB image here. Measured 2026-10-21:

```
data/processed   489 MB
data/external    2.2 GB
outputs/         7.6 GB   (ignored by git, not served)
```

The ten largest files in `data/external/`:

| Size | File | Needed to serve? |
|---|---|---|
| 416 MB | `transfermarkt/game_lineups.csv` | No — squad-value research |
| 246 MB | `transfermarkt/archive.zip` | No — raw download |
| 191 MB | `wyscout_2017_18/events_Italy.json` | No — one-off event study |
| 189 MB | `wyscout_2017_18/events_England.json` | No |
| 186 MB | `wyscout_2017_18/events_France.json` | No |
| 184 MB | `wyscout_2017_18/events_Spain.json` | No |
| 153 MB | `wyscout_2017_18/events_Germany.json` | No |
| 121 MB | `advanced/understat/understat_shots.csv` | **Yes** — the player-props fit reads it |
| 107 MB | `advanced/understat/understat_player_match.csv` | **Yes** — same |
| 40 MB | `transfermarkt/player_valuations.csv` | No (`squad_values.csv` is the derived input) |

So roughly **1.2 GB of the 2.2 GB is research material the serving path never
opens**, and the first job of A2 is to establish exactly what it does open —
by instrumenting, not by guessing. The method is cheap: start the API with a
read hook (or run the test suite plus a crawl of every page) and record every
path actually read. That list becomes the **serving bundle**.

Consequence for the image: **the data is a volume, never an image layer.** The
image carries code and dependencies; `data/` is mounted. An image that bakes in
a CSV has to be rebuilt by the daily refresh, which is absurd.

## 2. The real architectural question: where does the refresh run?

`scripts/update_season.py` downloads from ESPN and friends, rebuilds the
foundation, settles markets and logs the upcoming round; on this machine it is a
Windows scheduled task at 10:00 that also warms the caches, revalidates the
web's cache and commits the track record. Two options, and this one is the
user's call, not a session's:

**(a) The refresh stays on the Windows PC and pushes data to the VPS.** After a
clean run, rsync the serving bundle over SSH; the API container notices via the
`data_version()` mtime key it already has. Cheapest VPS (data only, no scraping
stack), keeps a pipeline that demonstrably works, and the daily commit of the
track record stays where the git credentials are. Cost: the site's data is only
as fresh as the PC being on.

**(b) The refresh moves into the stack**, as a scheduled container. The site is
then self-sufficient. Cost: the scraping dependencies (`soccerdata`, Selenium)
and their RAM move to the VPS, the image grows a lot, and the whole
`update_season` failure handling — notifications, rollback, `last_run.json` —
has to be re-expressed somewhere other than a Windows notification.

Recommendation: **(a) first**, because it is a two-hour job and it decouples
publishing from re-engineering the pipeline; (b) later, if the site ever has to
be independent of one laptop.

## 3. The stack

| Service | Image | Notes |
|---|---|---|
| `api` | `python:3.12-slim` + `requirements-lock.txt` | `data/` mounted read-only. Healthcheck on `/health` — **with a `start_period` of at least 3 minutes**, see below. |
| `web` | `node:22-alpine`, multi-stage | Needs `output: "standalone"` in `next.config.ts` (A1 adds it) or the image carries all of `node_modules`. **`next build`, never `next dev`.** |
| `postgres` | `postgres:17-alpine` | Named volume. Exists from **B1**; this is why B1 comes before A2. |
| `caddy` | `caddy:2` | Reverse proxy and automatic TLS. Routes `/` to `web`; the API is **not** exposed publicly unless something needs it directly. |

**The healthcheck needs a `start_period`, and no amount of API work removes
that.** Measured by A1 on 2026-10-21: uvicorn runs the lifespan *before* it
binds the socket, and the lifespan loads the models, so during a cold fit the
probe gets a refused connection — not a slow answer, not a 503, nothing to
answer with. With the joblib cache cold that fit is about two minutes. A
healthcheck without `start_period` therefore marks the container unhealthy,
restarts it, and the restart begins the fit again: it never comes up. `/health`
was made a pure process-state probe in the same PR (200 whenever the process is
up, with a `ready` flag), which covers every *live* case — a reload triggered by
the daily refresh, a request arriving while another holds the load, a start whose
warm-up failed — but it cannot cover a port that is not listening yet.

Notes that will save an afternoon:

- `NEXT_PUBLIC_API_URL` is baked in at build time, and the browser does call the
  API directly today. Decide early whether the browser keeps calling the API
  cross-origin (then it needs a public hostname and the CORS list in
  `api/main.py` needs the production origin) or whether everything goes through
  Next (then CORS stops mattering and the API can stay on the internal
  network). The second is cleaner; it means auditing the client-side calls in
  `web/lib/api.ts` first.
- The CORS block in `api/main.py` currently allows any `localhost:*` origin by
  regex. That is right for development and must not be what ships.
- Timezone: the containers should run with an explicit `TZ`, because kickoff
  rendering and the "today" window in `/fixtures/day` are date-sensitive.

## 4. Operations, minimal but real

- **Backups**: only Postgres truly needs them — `data/` is reproducible from the
  refresh and `outputs/` is not served. A nightly `pg_dump` to object storage or
  to the PC that already holds everything else.
- **CI**: the repo already runs the test suite on GitHub Actions. A2 adds
  building both images; deployment stays a manual `docker compose pull && up -d`
  until there is a reason for more.
- **Sizing**: the engine fit peaks in pandas over 46k matches and the joblib it
  produces is 29 MB. 4 GB is probably enough to *serve* (option a); 8 GB if the
  refresh ever runs there (option b).
- **Logs**: `uvicorn.log.err` currently lands in `data/processed/logs/`.
  Containers log to stdout; keep one place to look.

## 5. Open questions for the user, when A2 starts

| Question | Why it cannot be decided by a session |
|---|---|
| Does the daily refresh stay on the PC (a) or move to the server (b)? | It is a trade between cost, reliability and how much the pipeline gets rebuilt. |
| Domain name, and which registrar? | Naming and payment. |
| Is the site to be indexed by search engines, or quietly public? | Changes `robots.txt`, metadata and how much the privacy page matters. |
| Provider: Hetzner (cheapest at this size) or another? | Payment and preference. |
