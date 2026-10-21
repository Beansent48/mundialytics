/**
 * Refuse to build the site while the API is down.
 *
 * Every data page here is prerendered: `generateStaticParams` over the locales
 * makes /matchday, /leagues, /results, /awards and /squadlab SSG, so
 * `next build` renders each one against whatever the API answers *at build
 * time*. With the API down each page is built from its `catch` branch — the
 * "the model is warming up" box — and `next start` then serves that box as
 * static HTML until the revalidate window lapses or /api/revalidate is called.
 *
 * Measured on 2026-10-21: a build with the API down produced a 52 KB
 * /en/leagues holding the warming notice, against 68 KB of real standings with
 * the API up, and `next start` served the notice to every visitor. A build that
 * silently ships the error page is worse than one that refuses to run.
 */
const base = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const url = `${base.replace(/\/$/, "")}/health`;

try {
  const res = await fetch(url, { signal: AbortSignal.timeout(10_000) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const body = await res.json();
  console.log(`api ok: ${body.matches} matches through ${body.dataThrough}`);
} catch (err) {
  console.error(`\n  The API is not answering at ${url} (${err.message}).`);
  console.error("  A production build prerenders the data pages, so building");
  console.error("  now would bake the \"model is warming up\" box into them.\n");
  console.error("  Start it first:  .venv/Scripts/python.exe -m uvicorn api.main:app --port 8000");
  console.error("  Then:            npm --prefix web run prod\n");
  process.exit(1);
}
