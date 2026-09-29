import assert from "node:assert/strict";
import { test } from "node:test";
import { createQuery } from "../lib/matchday-cache.ts";

test("returning to a loaded list reuses the snapshot without another request", async () => {
  const query = createQuery();
  let calls = 0;
  const load = async () => { calls++; return { count: 24 }; };
  await query.refresh(load, 300_000);
  const snapshot = query.read();
  await query.refresh(load, 300_000);
  assert.equal(calls, 1);
  assert.equal(query.read(), snapshot);
  assert.equal(snapshot.data.count, 24);
});

test("concurrent mounts share the same pending request", async () => {
  const query = createQuery();
  let resolve;
  let calls = 0;
  const load = () => { calls++; return new Promise((done) => { resolve = done; }); };
  const first = query.refresh(load, 300_000);
  const second = query.refresh(load, 300_000);
  await Promise.resolve();
  assert.equal(first, second);
  assert.equal(calls, 1);
  resolve(["match"]);
  await first;
  assert.deepEqual(query.read().data, ["match"]);
});

test("expired data stays visible during refresh and after a failed refresh", async () => {
  const query = createQuery();
  await query.refresh(async () => ["old result"], -1);
  let reject;
  const refreshing = query.refresh(() => new Promise((_, fail) => { reject = fail; }), 300_000);
  await Promise.resolve();
  assert.deepEqual(query.read().data, ["old result"]);
  reject(new Error("offline"));
  await refreshing;
  assert.equal(query.read().failed, true);
  assert.deepEqual(query.read().data, ["old result"]);
  await query.refresh(async () => ["updated result"], 300_000);
  assert.deepEqual(query.read(), { data: ["updated result"], failed: false });
});

test("a failed first request can retry and the server snapshot never exposes client data", async () => {
  const query = createQuery();
  await query.refresh(async () => { throw new Error("offline"); }, 300_000);
  assert.equal(query.read().failed, true);
  await query.refresh(async () => ["match"], 300_000);
  assert.deepEqual(query.read().data, ["match"]);
  assert.deepEqual(query.server(), { failed: false });
});
