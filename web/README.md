# web/

The Mundialytics front end: Next.js 16 (App Router, Turbopack) with next-intl
(English at `/`, Spanish at `/es`), Tailwind 4 and next-themes. It holds no
model logic — every number comes from the FastAPI service in [`../api`](../api).

## Running it

The API has to be up first; the first request after a data update pays the
engine fit (~3 min), and pages show a "warming up" state until it lands.

```bash
uvicorn api.main:app --port 8000     # from the repo root
npm --prefix web install
npm --prefix web run dev             # http://localhost:3000
```

| Variable | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Where the API lives |
| `REVALIDATE_SECRET` | unset | Guards `GET /api/revalidate`; unset means localhost callers only |

Every API fetch is tagged `data`, so the daily refresh
(`scripts/run_update.ps1`) calls `/api/revalidate` once and the new matchday
shows up without restarting the server.

## Accounts

Optional, and only here — the FastAPI service has no notion of a user and is
not meant to grow one. Every page on the site renders for a visitor who never
signs in; an account exists so that what you follow can be saved. [Better
Auth](https://www.better-auth.com) over our own Postgres, with
[Drizzle](https://orm.drizzle.team) for the schema and the migrations. The
design, and the reasoning behind both choices, is in
[`../docs/USER_ACCOUNTS_DESIGN.md`](../docs/USER_ACCOUNTS_DESIGN.md).

Nothing below is needed to run the site. Skip it and the sign-in page is the
only thing that does not work.

```bash
npm --prefix web install             # first, even on an existing checkout
docker compose up -d postgres        # from the repo root
cp web/.env.example web/.env.local   # then fill it in, see below
npm --prefix web run db:migrate      # create the tables
```

The install line is not boilerplate. This release added `better-auth`,
`drizzle-orm`, `pg` and `drizzle-kit`, so a checkout that was already running
before it has a `node_modules` without them — and the symptom is not a helpful
message about a missing dependency but `MODULE_NOT_FOUND` from the server, or
`db:migrate` failing before it reaches the database.

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | `postgresql://mundialytics:mundialytics@localhost:5432/mundialytics` for the compose defaults |
| `BETTER_AUTH_SECRET` | Signs session cookies. `node -e "console.log(require('node:crypto').randomBytes(32).toString('base64url'))"` |
| `BETTER_AUTH_URL` | `http://localhost:3000` in development |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | From a Google Cloud OAuth client — the exact steps and the redirect URI are in [§9 of the design](../docs/USER_ACCOUNTS_DESIGN.md). Leave empty and the Google button is simply not offered; email sign-in still works |
| `AUTH_REQUIRE_EMAIL_VERIFICATION` | `1` refuses sign-in until the address is confirmed. Leave empty in development: there is no mail transport yet, so verification links are printed to the server log |

Changing the schema is three commands, and all three outputs are committed:

```bash
npm --prefix web run db:schema       # rewrite lib/db/schema.ts from lib/auth.ts
npm --prefix web run db:generate     # diff it into SQL under lib/db/migrations
npm --prefix web run db:migrate      # apply it
```

What the account holds: followed teams and competitions, theme and language,
and SquadLab runs. Signed out, all three behave exactly as they did before
accounts existed — favourites in `localStorage`, preferences in the browser,
and a SquadLab draft that is lost when you leave. Nothing became sign-in-only.

Two tests need the database up and the migrations applied. The second also
needs the web app running, because what it checks — whether a session reaches
the right rows, and whether one account can see another's — lives in the seam
between the route handler, Better Auth and Postgres:

```bash
node --test --test-reporter=spec web/tests/account-cascade.test.mjs
node --test --test-reporter=spec web/tests/user-data.test.mjs
```

Both skip rather than fail when nothing is listening, which is why CI (lint and
tsc, no database) stays green without them.

> **Adding a table?** Put it in `lib/db/app-schema.ts`, never in
> `lib/db/schema.ts`. The latter is generated from the auth config and
> `npm run db:schema` overwrites it, so a table added there disappears the next
> time anyone regenerates the auth schema — with no error, because the
> generator does not know it was not the only author.

## Layout

```
app/[locale]/        pages: matchday + match view, leagues, competitions,
                     results (track record), awards, squadlab, and
                     sign-in / sign-up / account
app/api/revalidate/  cache invalidation hook for the data refresh
app/api/auth/        Better Auth: sign-in, sign-out, the Google callback
lib/auth*.ts         the auth instance, the browser client, error messages
lib/db/              Drizzle schema, committed migrations, the pg pool
components/          one folder per page area, plus ui/ and layout/
data/                benchmark_vs_bet365.json, written by
                     scripts/benchmark_vs_bet365.py; the landing page and the
                     API both read it. Never edit it by hand.
lib/api.ts           typed API client (timeouts, revalidate windows)
messages/            en.json / es.json — same keys in both
i18n/, proxy.ts      locale routing
```

## Checks

```bash
npm run lint
npx tsc --noEmit
```

Both run in CI on every push.
