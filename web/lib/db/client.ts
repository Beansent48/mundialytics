import { drizzle, type NodePgDatabase } from "drizzle-orm/node-postgres";
import { Pool } from "pg";

import * as schema from "./schema";

/**
 * The Postgres connection the account layer reads and writes.
 *
 * Only Next.js talks to this database. The Python API in `api/` stays
 * stateless and anonymous — see docs/USER_ACCOUNTS_DESIGN.md §2 — so nothing
 * here is shared with the model server.
 *
 * Two deliberate choices:
 *
 * - **Lazy.** The connection is built on first use, not on import, so a page
 *   that never reads a user never needs `DATABASE_URL` — which is what keeps
 *   `next build` able to prerender the football pages with no database around.
 * - **Cached on `globalThis`.** `next dev` re-evaluates modules on every hot
 *   reload, and a fresh pool per reload leaks connections until Postgres
 *   refuses new ones.
 */
type Db = NodePgDatabase<typeof schema>;

const globalForDb = globalThis as unknown as { mundialyticsDb?: Db };

function connect(): Db {
  const url = process.env.DATABASE_URL;
  if (!url) {
    throw new Error(
      "DATABASE_URL is not set. Copy web/.env.example to web/.env.local and " +
        "start Postgres with `docker compose up -d postgres`.",
    );
  }
  return drizzle(new Pool({ connectionString: url }), { schema });
}

export function getDb(): Db {
  globalForDb.mundialyticsDb ??= connect();
  return globalForDb.mundialyticsDb;
}
