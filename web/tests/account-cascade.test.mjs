import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { Client } from "pg";

/**
 * "Delete my account" has to mean it.
 *
 * Nothing in the application deletes a session or a provider link when a user
 * goes; the database does, through `ON DELETE CASCADE` in
 * `lib/db/schema.ts`. That is deliberate — application code can forget a
 * table, and task B2 is about to add several — so this test checks the
 * constraint itself as well as the behaviour.
 *
 * It needs the real Postgres, with the migrations applied:
 *
 *   docker compose up -d postgres
 *   npm --prefix web run db:migrate
 *   node --test --test-reporter=spec web/tests/account-cascade.test.mjs
 *
 * Without `DATABASE_URL` it skips rather than fails, because `npm run lint`
 * and `tsc` are what CI runs and CI has no database.
 */
function databaseUrl() {
  if (process.env.DATABASE_URL) return process.env.DATABASE_URL;
  // The dev server reads `.env.local`; `node --test` does not.
  try {
    const contents = readFileSync(new URL("../.env.local", import.meta.url), "utf8");
    const match = /^\s*DATABASE_URL\s*=\s*(.+?)\s*$/m.exec(contents);
    return match ? match[1].replace(/^["']|["']$/g, "") : null;
  } catch {
    return null;
  }
}

const url = databaseUrl();

// One prefix per run, so a failed run cannot collide with the next one and
// nothing it leaves behind can be mistaken for a real account.
const tag = `cascade-test-${Date.now()}`;

test("deleting a user takes its sessions and provider links with it", async (t) => {
  if (!url) {
    t.skip("DATABASE_URL is not set; start Postgres and run the migrations");
    return;
  }

  const client = new Client({ connectionString: url });
  await client.connect();

  try {
    // The constraint, not just its effect: a later table that forgets the
    // cascade is the failure mode this guards against.
    const constraints = await client.query(
      `SELECT tc.table_name, rc.delete_rule
         FROM information_schema.table_constraints tc
         JOIN information_schema.referential_constraints rc
           ON rc.constraint_name = tc.constraint_name
         WHERE tc.constraint_type = 'FOREIGN KEY'
           AND tc.table_schema = 'public'
         ORDER BY tc.table_name`,
    );
    const rules = new Map(
      constraints.rows.map((row) => [row.table_name, row.delete_rule]),
    );
    assert.equal(rules.get("session"), "CASCADE");
    assert.equal(rules.get("account"), "CASCADE");
    for (const [table, rule] of rules) {
      assert.equal(rule, "CASCADE", `${table} must cascade from the user it keys on`);
    }

    await client.query(
      `INSERT INTO "user" (id, name, email, email_verified, created_at, updated_at)
       VALUES ($1, 'Cascade test', $2, false, now(), now())`,
      [`${tag}-user`, `${tag}@example.invalid`],
    );
    await client.query(
      `INSERT INTO "session" (id, expires_at, token, created_at, updated_at, user_id)
       VALUES ($1, now() + interval '1 day', $2, now(), now(), $3)`,
      [`${tag}-session`, `${tag}-token`, `${tag}-user`],
    );
    await client.query(
      `INSERT INTO "account" (id, account_id, provider_id, user_id, created_at, updated_at)
       VALUES ($1, $2, 'credential', $3, now(), now())`,
      [`${tag}-account`, `${tag}-user`, `${tag}-user`],
    );

    await client.query(`DELETE FROM "user" WHERE id = $1`, [`${tag}-user`]);

    for (const table of ["user", "session", "account"]) {
      const left = await client.query(
        `SELECT count(*)::int AS n FROM "${table}" WHERE id LIKE $1`,
        [`${tag}-%`],
      );
      assert.equal(left.rows[0].n, 0, `${table} still holds a row for the deleted user`);
    }
  } finally {
    // Only reached if an assertion failed before the delete.
    await client.query(`DELETE FROM "user" WHERE id LIKE $1`, [`${tag}-%`]);
    await client.end();
  }
});
