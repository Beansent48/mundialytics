import { engineFingerprint } from "@/lib/engine-fingerprint";
import { listRuns, saveRun, type RunStatus } from "@/lib/user-data";
import { BadRequest, readJson, withUser } from "@/lib/user-route";

/**
 * A person's SquadLab runs.
 *
 * `GET` is the list for the panel and deliberately leaves `result` out: a
 * played tournament carries every match, the bracket and the scorers, and a
 * handful of them would be megabytes to render four lines. Opening one fetches
 * it by id.
 *
 * `POST` saves, by an id the browser minted, so that saving a draft and later
 * saving the tournament it became are the same row and not two.
 */
export const dynamic = "force-dynamic";

const MODES = ["draft", "sandbox", "historic"];
const STATUSES = ["in_progress", "finished"];

function text(payload: Record<string, unknown>, field: string, allowed?: string[]) {
  const value = payload[field];
  if (typeof value !== "string" || !value) {
    throw new BadRequest(`${field} is required`);
  }
  if (allowed && !allowed.includes(value)) {
    throw new BadRequest(`${field} must be one of ${allowed.join(", ")}`);
  }
  if (value.length > 128) throw new BadRequest(`${field} is too long`);
  return value;
}

function record(payload: Record<string, unknown>, field: string) {
  const value = payload[field];
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new BadRequest(`${field} must be an object`);
  }
  return value as Record<string, never>;
}

export const GET = withUser((userId) => listRuns(userId));

export const POST = withUser(async (userId, request) => {
  const payload = await readJson(request);

  const seed = payload.seed;
  // A uint32 from crypto.getRandomValues. Checked rather than trusted: the
  // column is a bigint and a float would be silently truncated by Postgres.
  if (typeof seed !== "number" || !Number.isSafeInteger(seed) || seed < 0) {
    throw new BadRequest("seed must be a non-negative integer");
  }

  const status = text(payload, "status", STATUSES) as RunStatus;
  const result = payload.result ?? null;
  if (status === "finished" && result === null) {
    // The same rule the schema enforces, answered as a 400 instead of a
    // constraint violation, so a client bug reads as a client bug.
    throw new BadRequest("a finished run must carry its result");
  }

  const run = await saveRun(userId, {
    id: text(payload, "id"),
    mode: text(payload, "mode", MODES),
    seed,
    squad: record(payload, "squad"),
    rerolls: record(payload, "rerolls") as Record<string, number>,
    status,
    result,
    // Stamped here rather than sent by the browser: the browser may have no
    // route to the API at all once this is deployed, and a fingerprint the
    // client could choose would be worth nothing anyway.
    engineFingerprint: await engineFingerprint(),
  });

  return { id: run?.id ?? null, status: run?.status ?? null };
});
