# Platform roadmap — performance, deployment and user accounts

Written 2026-10-21. This is the index for two fronts that are not about the
model at all: **making the app fast to start and run**, and **giving it user
accounts**. Everything the project has done so far assumed a single operator on
one machine who starts two processes by hand. Both fronts are about stopping
that from being true.

Each task below is one session (one chat). They are ordered, and the order is a
dependency order, not a preference.

---

## Decisions already taken (2026-10-21)

These came from the user and are not open for a session to revisit on its own.

| Question | Decision |
|---|---|
| Publish now? | **No.** Local speed first; Docker and a domain when the rest is done. |
| Audience, eventually | **Public site, optional accounts.** Everything readable without logging in; the account only unlocks saving. |
| Budget, when it ships | **~15 €/month**: one VPS (4-8 GB) + a domain. No paid managed services. |
| Login | **Google + email, via a library** (Auth.js / Better Auth) inside Next.js, users in **our own Postgres**. Not Clerk, not Supabase. |
| What the account stores | Favourites and settings, SquadLab progress, match alerts — and, later, **achievements, goals and user levels to gamify SquadLab**. |
| Order | Performance → accounts → Docker/deploy. |

The gamification intent matters now even though it is built last: the SquadLab
schema in task **B2** has to be born with somewhere to hang an achievement, or
it gets rewritten six months later.

---

## The tasks

| # | Task | Session | Depends on | Detail |
|---|---|---|---|---|
| **A1** | Cold start: measure and fix | started 2026-10-21 | — | [COLD_START_DIAGNOSIS.md](COLD_START_DIAGNOSIS.md) |
| **B1** | Accounts: auth + Postgres foundation | built 2026-10-23 (v0.65.0) | A1 (touches `web/` layout and config) | [USER_ACCOUNTS_DESIGN.md](USER_ACCOUNTS_DESIGN.md) §1-3, and §8-9 for what was built |
| **B2** | User data on the server: favourites, settings, SquadLab progress | built 2026-10-23 (v0.66.0) | B1 (needs a session and a user id) | [USER_ACCOUNTS_DESIGN.md](USER_ACCOUNTS_DESIGN.md) §4-6, and §10 for what was built |
| **A2** | Dockerisation and deployment | queued | A1 (what warm start needs decides the entrypoint), B1 (Postgres is a container) | [DEPLOYMENT_PLAN.md](DEPLOYMENT_PLAN.md) |

A1 was started on 2026-10-21; B1, B2 and A2 exist as task cards in Claude Code
waiting to be started, and those cards expire in a week. **These files, not the
cards, are the durable record**: the brief for each unstarted task — scope,
traps, done-when, and the line to paste to start it — is in
[PLATFORM_TASK_BRIEFS.md](PLATFORM_TASK_BRIEFS.md), so a lapsed card costs
nothing.

Why this order and not another:

- **A1 first** because it is the only one whose benefit is felt today, on the
  machine the project actually runs on, and because every later task would
  otherwise inherit a slow feedback loop.
- **B1 before A2** because dockerising twice — once without a database and once
  with it — is the one avoidable piece of work in the whole plan. By the time
  the compose file is written, Postgres should already be a known quantity.
- **B2 after B1** because there is no point designing where a user's SquadLab
  run is stored before there is a user id to key it on.
- **A2 last** because publishing is a decision the user has explicitly deferred,
  and a deployment built before the app is fast is a deployment of a slow app.

---

## Backlog, after the four above

Not sessions yet. They are written down so they are not re-discovered.

1. **Match alerts** (requested). Email when a followed team plays, or when a
   prediction moves after the lineups land. Needs B1 (an address to write to),
   B2 (favourites server-side) and a watcher process — which in practice means
   A2, because a notifier that only runs while a laptop is open is not one.
2. **Gamification proper** (requested). Achievements, objectives, user levels
   and XP over SquadLab. B2 lays the table for it; this is the game design and
   the award rules on top.
3. **Own-bet tracking.** Deliberately left out of the account scope for now: it
   is a new module, not persistence of something that already exists.
4. **`outputs/` is 7.6 GB** and `data/external/` is 2.2 GB, most of it never
   read by the serving path. A2 has to answer this to produce an image that is
   not absurd; see the serving-bundle section of the deployment plan.

---

## What is true today, for whoever picks these up

- **Two processes, by hand.** `uvicorn api.main:app --port 8000` and
  `npm --prefix web run dev`. There is no production build in the loop, and
  `.claude/launch.json` starts the web in dev mode too.
- **The API is stateless and anonymous.** 3,000 lines over `api/`, reading CSVs
  and joblib-cached fits. It has no concept of a user and, if the plan in
  [USER_ACCOUNTS_DESIGN.md](USER_ACCOUNTS_DESIGN.md) holds, it never will:
  accounts live in Next.js route handlers against Postgres.
- **Favourites were already shaped like an account**, and since v0.66.0 they
  use it: `web/lib/favorites.ts` has two backends, `localStorage` signed out
  and the server signed in, and `useFavorites()` is unchanged — **no component
  was touched**, which is what that docstring had been promising since before
  accounts existed.
- **A SquadLab run is now a row.** It used to live in React state, with a
  warning on the way out that leaving would lose the team; signed out it still
  does. Gamification (backlog 2) can read `squadlab_run` as history rather than
  needing runs replayed.
- **The daily refresh already warms caches.** `scripts/update_season.py` step
  7m runs `scripts/warm_api_caches.py`, and `run_update.ps1` then calls the
  web's `/api/revalidate`. Cold start is not a cache-warming problem — see A1.
- **Next.js 16 renamed `middleware.ts` to `proxy.ts`** (`web/proxy.ts`, locale
  routing). Every auth guide on the internet says `middleware.ts`. B1 will trip
  over this if nobody says it first. *(It did not: B1 left `proxy.ts` alone,
  because nothing on the site is gated and Next's own docs say Proxy is not a
  session-management layer. Auth checks are in the route handlers and in the
  one page that reads a session.)*
- **Accounts are built but optional.** Since v0.65.0 the web app can sign
  somebody in, with the users in a Postgres started by `docker-compose.yml`.
  Skipping all of it leaves the site exactly as it was — which is also what a
  visitor without an account sees.
