import { revalidateTag } from "next/cache";
import type { NextRequest } from "next/server";

/**
 * On-demand cache invalidation for the daily data refresh.
 *
 * Every API response is fetched with the shared "data" tag (see lib/api.ts).
 * The scheduled update (scripts/run_update.ps1, step after the Python refresh)
 * calls this route once it has rewritten fixtures, results and the ESPN event
 * feed, so the site shows the new matchday — scores, timelines, standings —
 * the moment the refresh finishes instead of waiting for each page's own
 * revalidate window to lapse. `{ expire: 0 }` drops the stale copies right away,
 * which is what we want for an out-of-band trigger (updateTag is Server-Action
 * only, per the Next 16 docs).
 *
 * Auth: if REVALIDATE_SECRET is set it must match (?secret= or the
 * x-revalidate-secret header); otherwise only localhost callers are allowed, so
 * the local single-operator setup needs zero configuration while a public
 * deployment is protected the moment the secret is present.
 */
function authorized(request: NextRequest): boolean {
  const secret = process.env.REVALIDATE_SECRET;
  if (secret) {
    const provided =
      request.nextUrl.searchParams.get("secret") ??
      request.headers.get("x-revalidate-secret");
    return provided === secret;
  }
  const host = (request.headers.get("host") ?? "").split(":")[0];
  return host === "localhost" || host === "127.0.0.1" || host === "::1";
}

export async function GET(request: NextRequest) {
  if (!authorized(request)) {
    return Response.json({ revalidated: false, error: "unauthorized" }, { status: 401 });
  }
  revalidateTag("data", { expire: 0 });
  return Response.json({ revalidated: true, tag: "data", now: Date.now() });
}
