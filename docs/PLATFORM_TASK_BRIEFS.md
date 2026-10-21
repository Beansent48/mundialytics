# Platform tasks: ready-to-start briefs

The three tasks of the [platform roadmap](PLATFORM_ROADMAP.md) that have not
been started: **B1**, **B2** and **A2**, in that order. A1 (cold start) was
started on 2026-10-21 and is not here.

Each section is everything a fresh session needs. They exist as task cards in
Claude Code too, but **those cards expire after a week and this file does not**
— if a card is gone, open a new session and paste the line under *How to start
it*. The reasoning behind each task lives in the linked design document; these
are the operational briefs, not the design.

Order is a dependency order. B2 needs B1 merged. A2 needs B1 merged and A1
merged. Running two of them at once means two sessions writing the same files
under `web/`.

---

## B1 — Accounts: authentication + Postgres

**Prerequisite:** A1 merged (it touches `web/package.json`,
`web/next.config.ts`, `.claude/launch.json` and `api/main.py`).
**Design:** [USER_ACCOUNTS_DESIGN.md](USER_ACCOUNTS_DESIGN.md), sections 1-3 and 7.

### How to start it

> Lee `docs/PLATFORM_TASK_BRIEFS.md` sección B1 y `docs/USER_ACCOUNTS_DESIGN.md`,
> y monta la base de cuentas de usuario.

### Scope

1. **Postgres locally via Docker**: a compose file with one `postgres:17-alpine`
   service and a named volume. It is the first piece of the stack A2 will
   finish, so write it as something that grows, not as a stopgap.
2. **Data layer**: Drizzle, migrations committed to the repo, and a committed
   `web/.env.example` listing every key with no value (`DATABASE_URL`, the
   Google OAuth client, the session secret). `web/.env.local` is never
   committed.
3. **The auth library**: choose between Better Auth and Auth.js v5 **by reading
   their current documentation** — both moved fast through 2026 — with a bias to
   Better Auth for first-class Drizzle support and email+password without a
   separate adapter. Record why in the design document.
4. **Google + email**, with email verification if the library ships it. The user
   has to create the OAuth client in Google Cloud themselves: write down the
   exact steps and the redirect URLs they need instead of guessing them.
5. **The session in the UI**: sign in/out in the header, a minimal account page
   (who you are, sign out, **delete account**), strings in both
   `web/messages/en.json` and `es.json`.
6. **The guard**, if one is needed, inside the existing `web/proxy.ts` after the
   locale rewrite, or in the route handlers.

### Traps

- Next.js 16 renamed `middleware.ts` to **`proxy.ts`**. Every auth guide says
  `middleware.ts`; creating that file gives you a guard Next never runs, failing
  silently.
- `web/lib/api.ts` caches every call with a shared `tags: ["data"]`. **No
  per-user response may go through it**, or one user's data is served to
  another. User data gets its own module, no `data` tag, `cache: "no-store"`.
- No public route may become sign-in-only. The account unlocks **saving**, not
  **seeing**.
- Delete account must really delete: cascade in the schema plus a test.
- Add nothing analytics-shaped. A strictly necessary session cookie needs no
  consent banner; anything else does.

### Done when

Sign-in works with Google and with email against the Docker Postgres; the
session survives a reload; sign out and delete account work; a visitor without
an account sees exactly what they saw before; `pytest tests/` and
`npm --prefix web run lint` green; the design document updated with what was
actually built (library chosen, real schema, Google Cloud steps) and the README
with how to start the database. Commit by topic, PR against `master`, no merge
without the user saying so.

---

## B2 — User data on the server

**Prerequisite:** B1 merged. Without a `user_id` there is nothing to key on — if
B1 is not in `master`, stop and say so.
**Design:** [USER_ACCOUNTS_DESIGN.md](USER_ACCOUNTS_DESIGN.md), sections 4-6.

### How to start it

> Lee `docs/PLATFORM_TASK_BRIEFS.md` sección B2 y `docs/USER_ACCOUNTS_DESIGN.md`,
> y saca del navegador los favoritos, los ajustes y el progreso de SquadLab.

### Scope

1. **Our tables** in Drizzle, in a committed migration: `user_settings`,
   `user_favorites` (composite PK), `squadlab_run`, and the empty gamification
   ones (`user_progress`, `achievement`, `user_achievement`). Everything
   cascades on user delete, with a test for it.
2. **Favourites with two backends**: `web/lib/favorites.ts` keeps its public API
   exactly as it is and, when there is a session, reads and writes the server
   with `localStorage` as an optimistic cache; with no session it behaves
   exactly as today.
3. **First-login migration**: read `mv.favorites.v1` and **merge, never
   replace** (someone who followed Valencia on their phone and Betis on their
   laptop ends with both), write server-side, mark the local copy migrated
   without deleting it — signing out cannot be data loss.
4. **Account settings**: locale, theme and odds format saved and applied when
   signing in from another device. The cookie stays authoritative for the
   request in flight.
5. **SquadLab progress**: save and continue a run. Store the inputs (seed,
   names, rerolls) **and the result of a finished run**, because pools and
   ratings move with the data: a finished run keeps its outcome and its engine
   fingerprint and is never silently re-simulated. Same reasoning as the
   pre-kickoff prediction log.
6. New strings in both `web/messages/en.json` and `es.json`.

### Out of scope

Inventing achievement codes, XP curves or level thresholds — empty tables and a
documented shape; the game rules are their own task. And the Python API, which
stays stateless and anonymous.

### Done when

With a session, favourites follow the user across browsers; with no session
everything works as before; first login merges instead of overwriting; a
SquadLab run can be left half-played and continued from another device;
deleting the account takes it all; `pytest tests/` and
`npm --prefix web run lint` green; the design document updated with the real
schema and with whichever open questions got settled. Commit by topic, PR
against `master`, no merge without the user saying so.

---

## A2 — Dockerisation and deployment

**Prerequisites:** A1 merged (it defines what container readiness means) and B1
merged (Postgres is a container). **And the user's decision to publish** — as of
2026-10-21 they have explicitly deferred it, so this one waits for them.
**Design:** [DEPLOYMENT_PLAN.md](DEPLOYMENT_PLAN.md).

### How to start it

> Lee `docs/PLATFORM_TASK_BRIEFS.md` sección A2 y `docs/DEPLOYMENT_PLAN.md`,
> y deja el proyecto listo para vivir en un servidor.

### First, ask

Before writing a single Dockerfile, put the four open questions in section 5 of
the deployment plan to the user. Two of them change the architecture: **where
the daily refresh runs** (stays on the PC and pushes data to the server, or
moves into the stack with Selenium and all), the domain name and registrar,
whether the site is to be indexed, and the provider.

### Scope

1. **Establish what the serving path actually reads.** Measured: `data/processed`
   is 489 MB and `data/external` 2.2 GB, of which ~1.2 GB is research material
   never opened in production — but `understat_shots.csv` (121 MB) and
   `understat_player_match.csv` (107 MB) *are* read by the props fit. Instrument
   the reads (an open hook, or the suite plus a crawl of every page) and
   document the real list. That list is the **serving bundle**.
2. **Data is a volume, never an image layer.** An image with a CSV inside would
   have to be rebuilt by the daily refresh.
3. **Two images**: `api` on `python:3.12-slim` with `requirements-lock.txt` and a
   healthcheck on `/health`; `web` multi-stage on `node:22-alpine` with
   `output: "standalone"`, built with `next build` and never `next dev`.
4. **The full compose**: `api`, `web`, `postgres` (from B1) and `caddy` for
   reverse proxy and automatic TLS. Decide early whether the browser keeps
   calling the API cross-origin — then it needs a public hostname and the CORS
   list in `api/main.py` needs the production origin — or whether everything
   goes through Next, which is cleaner and makes CORS stop mattering. The regex
   allowing any `localhost:*` must not be what ships.
5. **Minimal real operations**: nightly `pg_dump` (only Postgres needs backup),
   an explicit `TZ` in the containers because kickoff rendering and the "today"
   window of `/fixtures/day` are date-sensitive, logs to stdout, and both images
   built by the CI that already runs the suite.

### Done when

`docker compose up` brings the whole site up locally with the data mounted and
the accounts Postgres running; the image does not carry gigabytes of CSV; how
fresh data reaches the server is written down and tested; the deployment plan
reflects what was built, with the serving bundle and its real size; and the
README says how to deploy it and how to run it locally. Commit by topic, PR
against `master`, no merge without the user saying so.
