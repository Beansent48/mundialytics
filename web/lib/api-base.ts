/**
 * Where the API is — which is two different answers, and that is the point.
 *
 * The same `lib/api.ts` runs in a Server Component and in the browser, and in a
 * deployment those two have different routes to the model server:
 *
 * - **the server** talks to it directly on the container network
 *   (`API_INTERNAL_URL=http://api:8000`): one hop, no proxy, and the API needs
 *   no public hostname;
 * - **the browser** calls it on the site's own origin, as a path
 *   (`NEXT_PUBLIC_API_URL=/engine`), which Caddy hands to the API. Same origin
 *   means no CORS at all — the development allow-list of any `localhost:*`
 *   port, which the deployment plan said must not ship, simply stops applying.
 *
 * `NEXT_PUBLIC_*` is inlined at build time, so the browser's answer is fixed
 * when the image is built; the server's is read per call, so it is the runtime
 * environment's. In development neither variable is set and both are
 * `http://localhost:8000`, exactly as before — nothing about running
 * `uvicorn` and `next dev` by hand changes.
 *
 * A relative base only works in the browser: a server-side `fetch("/engine/…")`
 * has no origin to resolve against. That is why the server prefers
 * `API_INTERNAL_URL` and why this is one module rather than a constant copied
 * into three files.
 */

const BROWSER_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** The base URL to use from here: internal on the server, same-origin in the browser. */
export function apiBase(): string {
  if (typeof window === "undefined") {
    return process.env.API_INTERNAL_URL ?? BROWSER_BASE;
  }
  return BROWSER_BASE;
}
