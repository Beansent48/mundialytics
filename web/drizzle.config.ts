import { readFileSync } from "node:fs";

import { defineConfig } from "drizzle-kit";

/**
 * drizzle-kit runs outside Next, and Next is what normally loads `.env.local`,
 * so the connection string has to be read here. Six lines of parsing rather
 * than a dependency whose only job is this.
 */
function loadEnvLocal(): void {
  let contents: string;
  try {
    contents = readFileSync(new URL(".env.local", import.meta.url), "utf8");
  } catch {
    // Already in the environment, or genuinely absent — the error below says so.
    return;
  }
  for (const line of contents.split("\n")) {
    const match = /^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/.exec(line);
    if (match && !process.env[match[1]]) {
      process.env[match[1]] = match[2].replace(/^["']|["']$/g, "");
    }
  }
}

loadEnvLocal();

const url = process.env.DATABASE_URL;
if (!url) {
  throw new Error(
    "DATABASE_URL is not set. Copy web/.env.example to web/.env.local and " +
      "start Postgres with `docker compose up -d postgres`.",
  );
}

export default defineConfig({
  dialect: "postgresql",
  schema: "./lib/db/schema.ts",
  out: "./lib/db/migrations",
  dbCredentials: { url },
  // Table names are Better Auth's (`user`, `session`, …) and are quoted in the
  // generated SQL, so `user` being a reserved word is not a problem.
  casing: "snake_case",
});
