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

## Layout

```
app/[locale]/        pages: matchday + match view, leagues, competitions,
                     results (track record), awards, squadlab
app/api/revalidate/  cache invalidation hook for the data refresh
components/          one folder per page area, plus ui/ and layout/
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
