import assert from "node:assert/strict";
import { test } from "node:test";

/**
 * The account layer, over HTTP, with two real accounts.
 *
 * Deliberately not unit tests over `lib/user-data.ts`: the questions worth
 * answering here are "does a session reach the right rows" and "can one
 * account see another's", and both live in the seam between the route handler,
 * Better Auth and Postgres. Calling the functions directly would skip exactly
 * the part that can be wrong.
 *
 * Needs the web app running against the compose Postgres with the migrations
 * applied:
 *
 *   docker compose up -d postgres
 *   npm --prefix web run db:migrate
 *   uvicorn api.main:app --port 8000      # the build guard wants it
 *   npm --prefix web run prod
 *   node --test --test-reporter=spec web/tests/user-data.test.mjs
 *
 * Skips rather than fails when nothing is listening, because CI runs lint and
 * tsc and has neither a server nor a database.
 */
const BASE = process.env.WEB_URL ?? "http://localhost:3000";

async function reachable() {
  try {
    const res = await fetch(`${BASE}/api/auth/get-session`, {
      signal: AbortSignal.timeout(2000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * One signed-in account, holding its own cookie — two of these never mix.
 *
 * Every request carries `Origin`, because Better Auth refuses one without it
 * (`MISSING_OR_NULL_ORIGIN`, 403) and a browser always sends it. Leaving it off
 * here would be testing a client no browser resembles.
 */
async function signUp() {
  const email = `test-${Date.now()}-${Math.random().toString(36).slice(2, 8)}@example.invalid`;
  const password = "a-password-long-enough";
  const res = await fetch(`${BASE}/api/auth/sign-up/email`, {
    method: "POST",
    headers: { "content-type": "application/json", origin: BASE },
    body: JSON.stringify({ name: "Test Account", email, password }),
  });
  if (res.status === 429) {
    // Better Auth rate-limits its own endpoints, which is a property worth
    // having and makes this file flaky when it is run repeatedly. Say so,
    // rather than leaving a bare 429 to look like a bug in the account layer.
    throw new RateLimited();
  }
  assert.equal(res.status, 200, "sign-up should succeed");
  const cookie = (res.headers.getSetCookie?.() ?? [])
    .map((c) => c.split(";")[0])
    .join("; ");
  assert.ok(cookie, "sign-up should set a session cookie");

  const call = async (path, init = {}) =>
    fetch(`${BASE}${path}`, {
      ...init,
      headers: {
        ...(init.headers ?? {}),
        cookie,
        origin: BASE,
        "content-type": "application/json",
      },
    });

  return { email, password, cookie, call };
}

const json = (body) => ({ body: JSON.stringify(body) });

/** Not a failure of the thing under test — the auth library defending itself. */
class RateLimited extends Error {}

test("the account layer keeps each account to its own rows", async (t) => {
  if (!(await reachable())) {
    t.skip(`nothing answering at ${BASE}; start the web app first`);
    return;
  }

  let a;
  let b;
  try {
    a = await signUp();
    b = await signUp();
  } catch (error) {
    if (error instanceof RateLimited) {
      t.skip("Better Auth is rate-limiting sign-up; wait a minute and re-run");
      return;
    }
    throw error;
  }

  await t.test("signed out is 401, not a redirect to HTML", async () => {
    const res = await fetch(`${BASE}/api/user/favorites`);
    assert.equal(res.status, 401);
    assert.match(res.headers.get("content-type") ?? "", /json/);
  });

  await t.test("a user's own data is never cacheable", async () => {
    const res = await a.call("/api/user/favorites");
    assert.equal(res.status, 200);
    // The whole point of the separate module: this must never be stored by a
    // shared cache and handed to the next person.
    assert.match(res.headers.get("cache-control") ?? "", /no-store/);
    assert.match(res.headers.get("cache-control") ?? "", /private/);
  });

  await t.test("favourites round-trip and stay sorted", async () => {
    const res = await a.call("/api/user/favorites", {
      method: "PUT",
      ...json({ teams: ["Valencia", "Betis"], competitions: ["laliga"] }),
    });
    assert.equal(res.status, 200);
    assert.deepEqual(await res.json(), {
      teams: ["Betis", "Valencia"],
      competitions: ["laliga"],
    });
  });

  await t.test("merge is a union, never a replace (design §5)", async () => {
    // The phone-and-laptop case: this browser knows one team the server does
    // not, and the server knows two it does not.
    const res = await a.call("/api/user/favorites", {
      method: "POST",
      ...json({ teams: ["Sevilla"], competitions: [] }),
    });
    const merged = await res.json();
    assert.deepEqual(merged.teams, ["Betis", "Sevilla", "Valencia"]);
    assert.deepEqual(merged.competitions, ["laliga"], "a merge drops nothing");
  });

  await t.test("one account's favourites are invisible to another", async () => {
    const res = await b.call("/api/user/favorites");
    assert.deepEqual(await res.json(), { teams: [], competitions: [] });
  });

  await t.test("a bad body is a 400, not a 500", async () => {
    const res = await a.call("/api/user/favorites", {
      method: "PUT",
      ...json({ teams: "Valencia" }),
    });
    assert.equal(res.status, 400);
  });

  await t.test("settings patch one field without clearing the others", async () => {
    await a.call("/api/user/settings", { method: "PUT", ...json({ theme: "dark" }) });
    const res = await a.call("/api/user/settings", {
      method: "PUT",
      ...json({ locale: "es" }),
    });
    const settings = await res.json();
    assert.equal(settings.theme, "dark", "the theme survived a locale write");
    assert.equal(settings.locale, "es");
  });

  await t.test("settings reject a value that is not ours", async () => {
    const res = await a.call("/api/user/settings", {
      method: "PUT",
      ...json({ locale: "fr" }),
    });
    assert.equal(res.status, 400);
  });

  let runId;

  await t.test("a run saves, and a finished one keeps its result", async () => {
    runId = crypto.randomUUID();
    const draft = await a.call("/api/user/squadlab/runs", {
      method: "POST",
      ...json({
        id: runId,
        mode: "sandbox",
        seed: 4000000000,
        squad: { Forward_0: { player: "Someone", kind: "actual" } },
        rerolls: {},
        status: "in_progress",
        result: null,
      }),
    });
    assert.equal(draft.status, 200, "sandbox is a real mode, and a big seed fits");

    const finished = await a.call("/api/user/squadlab/runs", {
      method: "POST",
      ...json({
        id: runId,
        mode: "sandbox",
        seed: 4000000000,
        squad: { Forward_0: { player: "Someone", kind: "actual" } },
        rerolls: {},
        status: "finished",
        result: { stage: "Campeón", teamName: "Test XI" },
      }),
    });
    assert.equal(finished.status, 200);

    const list = await (await a.call("/api/user/squadlab/runs")).json();
    assert.equal(list.length, 1, "the draft and the season are one row, not two");
    assert.equal(list[0].status, "finished");
    assert.equal(list[0].stage, "Campeón");
    assert.equal(list[0].seed, 4000000000, "the seed survived the bigint column");
  });

  await t.test("a finished run is never overwritten", async () => {
    await a.call("/api/user/squadlab/runs", {
      method: "POST",
      ...json({
        id: runId,
        mode: "sandbox",
        seed: 1,
        squad: {},
        rerolls: {},
        status: "finished",
        result: { stage: "Eliminado", teamName: "Rewritten" },
      }),
    });
    const run = await (await a.call(`/api/user/squadlab/runs/${runId}`)).json();
    assert.equal(run.result.stage, "Campeón", "replaying cannot rewrite history");
  });

  await t.test("a finished run must carry a result", async () => {
    const res = await a.call("/api/user/squadlab/runs", {
      method: "POST",
      ...json({
        id: crypto.randomUUID(),
        mode: "draft",
        seed: 1,
        squad: {},
        rerolls: {},
        status: "finished",
        result: null,
      }),
    });
    assert.equal(res.status, 400);
  });

  await t.test("another account's run is a 404, not a 403", async () => {
    // 403 would confirm it exists, which is itself the leak.
    assert.equal((await b.call(`/api/user/squadlab/runs/${runId}`)).status, 404);
    assert.equal(
      (await b.call(`/api/user/squadlab/runs/${runId}`, { method: "DELETE" })).status,
      404,
    );
    // And it is still there afterwards.
    assert.equal((await a.call(`/api/user/squadlab/runs/${runId}`)).status, 200);
  });

  await t.test("deleting the account takes the lot", async () => {
    const res = await a.call("/api/auth/delete-user", {
      method: "POST",
      ...json({ password: a.password }),
    });
    assert.equal(res.status, 200, "delete-user should succeed with the password");

    // The session is gone with the user, so the only thing left to check is
    // that nothing answers for it any more.
    assert.equal((await a.call("/api/user/favorites")).status, 401);

    // B is untouched: a cascade must not reach past its own user.
    assert.equal((await b.call("/api/user/favorites")).status, 200);

    await b.call("/api/auth/delete-user", {
      method: "POST",
      ...json({ password: b.password }),
    });
  });
});
