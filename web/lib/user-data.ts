import { and, desc, eq, sql } from "drizzle-orm";

import {
  squadlabRun,
  userFavorites,
  userSettings,
} from "@/lib/db/app-schema";
import { getDb } from "@/lib/db/client";

/**
 * Everything a signed-in person has saved, read and written per request.
 *
 * **Server only, and never through `lib/api.ts`.** That client caches every
 * call under a shared `tags: ["data"]`, which is correct for public football
 * numbers and catastrophic for a user's own rows: one person's favourites
 * would be cached and served to the next. Nothing in this file touches it, the
 * route handlers that call it are dynamic, and the browser fetches them with
 * `cache: "no-store"`.
 *
 * Every function takes a `userId` the caller has already proved. Resolving the
 * session is the route handler's job (`lib/user-route.ts`); this layer trusts
 * its argument and does nothing else, so there is exactly one place where a
 * missing check could let one account read another.
 */

export type Favorites = { teams: string[]; competitions: string[] };

export type Settings = {
  locale: string | null;
  theme: string | null;
  oddsFormat: string | null;
};

export type RunStatus = "in_progress" | "finished";

export type RunSummary = {
  id: string;
  mode: string;
  seed: number;
  status: RunStatus;
  /** Set only for a finished run: what the squad achieved, e.g. "Campeón". */
  stage: string | null;
  teamName: string | null;
  players: number;
  engineFingerprint: string | null;
  createdAt: string;
  updatedAt: string;
};

const EMPTY: Favorites = { teams: [], competitions: [] };

// --- favourites ---------------------------------------------------------

export async function readFavorites(userId: string): Promise<Favorites> {
  const rows = await getDb()
    .select({ kind: userFavorites.kind, value: userFavorites.value })
    .from(userFavorites)
    .where(eq(userFavorites.userId, userId));

  const out: Favorites = { teams: [], competitions: [] };
  for (const row of rows) {
    (row.kind === "team" ? out.teams : out.competitions).push(row.value);
  }
  // Stable order, so a round trip through the server never reshuffles a list
  // the reader is looking at.
  out.teams.sort();
  out.competitions.sort();
  return out;
}

function pairs(favorites: Favorites, userId: string) {
  return [
    ...favorites.teams.map((value) => ({ userId, kind: "team", value })),
    ...favorites.competitions.map((value) => ({
      userId,
      kind: "competition",
      value,
    })),
  ].filter((row) => row.value.length > 0 && row.value.length <= 200);
}

/** Replace the lot. What a toggle sends: the client knows the whole set. */
export async function writeFavorites(
  userId: string,
  favorites: Favorites,
): Promise<Favorites> {
  const rows = pairs(favorites, userId);
  await getDb().transaction(async (tx) => {
    await tx.delete(userFavorites).where(eq(userFavorites.userId, userId));
    if (rows.length) {
      await tx.insert(userFavorites).values(rows);
    }
  });
  return readFavorites(userId);
}

/**
 * Union, not replace — §5 of the design.
 *
 * What first login runs. Someone who followed Valencia on their phone and
 * Betis on their laptop ends with both, and the one that happened to sign in
 * second does not win.
 */
export async function mergeFavorites(
  userId: string,
  incoming: Favorites,
): Promise<Favorites> {
  const rows = pairs(incoming, userId);
  if (rows.length) {
    await getDb().insert(userFavorites).values(rows).onConflictDoNothing();
  }
  return readFavorites(userId);
}

// --- settings -----------------------------------------------------------

const NO_SETTINGS: Settings = { locale: null, theme: null, oddsFormat: null };

export async function readSettings(userId: string): Promise<Settings> {
  const [row] = await getDb()
    .select({
      locale: userSettings.locale,
      theme: userSettings.theme,
      oddsFormat: userSettings.oddsFormat,
    })
    .from(userSettings)
    .where(eq(userSettings.userId, userId))
    .limit(1);
  return row ?? NO_SETTINGS;
}

/**
 * Write the settings that were named, leave the rest alone.
 *
 * A partial update rather than a replace: the theme switch and the language
 * switch are different controls in the same menu, and each should be able to
 * save without knowing the other's current value.
 */
export async function writeSettings(
  userId: string,
  patch: Partial<Settings>,
): Promise<Settings> {
  const next = { ...(await readSettings(userId)), ...patch };
  await getDb()
    .insert(userSettings)
    .values({ userId, ...next })
    .onConflictDoUpdate({ target: userSettings.userId, set: next });
  return next;
}

// --- SquadLab runs ------------------------------------------------------

type SeasonResult = {
  stage?: string;
  teamName?: string;
} | null;

function summarise(row: {
  id: string;
  mode: string;
  seed: number;
  status: string;
  squad: unknown;
  result: unknown;
  engineFingerprint: string | null;
  createdAt: Date;
  updatedAt: Date;
}): RunSummary {
  const result = row.result as SeasonResult;
  return {
    id: row.id,
    mode: row.mode,
    seed: row.seed,
    status: row.status as RunStatus,
    stage: result?.stage ?? null,
    teamName: result?.teamName ?? null,
    players: Object.keys((row.squad as Record<string, unknown>) ?? {}).length,
    engineFingerprint: row.engineFingerprint,
    createdAt: row.createdAt.toISOString(),
    updatedAt: row.updatedAt.toISOString(),
  };
}

/**
 * The list for the "your runs" panel, newest first.
 *
 * Deliberately without `result`, which is a whole played tournament — every
 * match, the bracket, the scorers — and is megabytes once a few runs exist.
 * The list needs a line each; `readRun` fetches the tournament when someone
 * actually opens one.
 */
export async function listRuns(
  userId: string,
  limit = 50,
): Promise<RunSummary[]> {
  const rows = await getDb()
    .select({
      id: squadlabRun.id,
      mode: squadlabRun.mode,
      seed: squadlabRun.seed,
      status: squadlabRun.status,
      squad: squadlabRun.squad,
      // Two fields out of the result rather than the result: Postgres reads
      // them from the jsonb without sending the rest over the wire.
      result: sql<SeasonResult>`jsonb_build_object(
        'stage', ${squadlabRun.result} -> 'stage',
        'teamName', ${squadlabRun.result} -> 'teamName'
      )`,
      engineFingerprint: squadlabRun.engineFingerprint,
      createdAt: squadlabRun.createdAt,
      updatedAt: squadlabRun.updatedAt,
    })
    .from(squadlabRun)
    .where(eq(squadlabRun.userId, userId))
    .orderBy(desc(squadlabRun.createdAt))
    .limit(Math.min(Math.max(limit, 1), 200));
  return rows.map(summarise);
}

export async function readRun(userId: string, id: string) {
  const [row] = await getDb()
    .select()
    .from(squadlabRun)
    // The user id is in the WHERE, not checked afterwards: a run belonging to
    // someone else must be indistinguishable from one that does not exist.
    .where(and(eq(squadlabRun.userId, userId), eq(squadlabRun.id, id)))
    .limit(1);
  return row ?? null;
}

export type RunInput = {
  id: string;
  mode: string;
  seed: number;
  /** The chosen cards by slot; see the column's note in app-schema.ts. */
  squad: Record<string, unknown>;
  rerolls: Record<string, number>;
  status: RunStatus;
  result: unknown | null;
  engineFingerprint: string | null;
};

/**
 * Create or update one run, by its client-minted id.
 *
 * The id comes from the browser so that saving a draft, and later saving the
 * tournament it became, are the same row rather than two. A finished run is
 * final: once `status` is `finished` the result it recorded is never
 * overwritten, because the run cannot be replayed into the same tournament
 * after the data moves under it.
 */
export async function saveRun(userId: string, input: RunInput) {
  const existing = await readRun(userId, input.id);
  if (existing?.status === "finished") return existing;

  const values = {
    id: input.id,
    userId,
    mode: input.mode,
    seed: input.seed,
    squad: input.squad,
    rerolls: input.rerolls,
    status: input.status,
    result: input.result,
    engineFingerprint: input.engineFingerprint,
  };
  await getDb()
    .insert(squadlabRun)
    .values(values)
    .onConflictDoUpdate({
      target: squadlabRun.id,
      set: {
        squad: values.squad,
        rerolls: values.rerolls,
        status: values.status,
        result: values.result,
        engineFingerprint: values.engineFingerprint,
      },
      // Belt and braces next to the ownership check above: a conflict on a
      // primary key someone else owns must not update their row.
      where: eq(squadlabRun.userId, userId),
    });
  return readRun(userId, input.id);
}

export async function deleteRun(userId: string, id: string): Promise<boolean> {
  const deleted = await getDb()
    .delete(squadlabRun)
    .where(and(eq(squadlabRun.userId, userId), eq(squadlabRun.id, id)))
    .returning({ id: squadlabRun.id });
  return deleted.length > 0;
}

export { EMPTY as NO_FAVORITES };
