import { deleteRun, readRun } from "@/lib/user-data";
import { NotFound, withUser } from "@/lib/user-route";

/**
 * One run, with the tournament it produced.
 *
 * The id alone is not authority: both handlers scope the query by user id, so
 * a run belonging to somebody else is a 404 and not a 403 — indistinguishable
 * from one that never existed, which is the only answer that leaks nothing.
 */
export const dynamic = "force-dynamic";

function idFrom(request: Request): string {
  // Taken from the URL rather than the route's `params`, because `withUser`
  // wraps a plain `(userId, request)` handler and one shape for every route is
  // worth more here than the typed second argument.
  const { pathname } = new URL(request.url);
  return decodeURIComponent(pathname.slice(pathname.lastIndexOf("/") + 1));
}

export const GET = withUser(async (userId, request) => {
  const run = await readRun(userId, idFrom(request));
  if (!run) throw new NotFound();
  return {
    id: run.id,
    mode: run.mode,
    seed: run.seed,
    squad: run.squad,
    rerolls: run.rerolls,
    status: run.status,
    result: run.result,
    engineFingerprint: run.engineFingerprint,
    createdAt: run.createdAt.toISOString(),
    updatedAt: run.updatedAt.toISOString(),
  };
});

export const DELETE = withUser(async (userId, request) => {
  if (!(await deleteRun(userId, idFrom(request)))) throw new NotFound();
  return { deleted: true };
});
