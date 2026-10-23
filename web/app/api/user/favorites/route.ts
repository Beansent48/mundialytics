import {
  mergeFavorites,
  readFavorites,
  writeFavorites,
  type Favorites,
} from "@/lib/user-data";
import { readJson, stringList, withUser } from "@/lib/user-route";

/**
 * The followed teams and competitions of whoever is asking.
 *
 * `PUT` replaces, because a toggle in the UI already knows the whole set.
 * `POST` merges, and exists for exactly one moment: the first sign-in on a
 * browser that has been collecting favourites in `localStorage`. §5 of the
 * design insists that is a union — follow Valencia on a phone and Betis on a
 * laptop and you keep both — so it cannot reuse `PUT`.
 */
export const dynamic = "force-dynamic";

function body(payload: Record<string, unknown>): Favorites {
  return {
    teams: stringList(payload.teams, "teams"),
    competitions: stringList(payload.competitions, "competitions"),
  };
}

export const GET = withUser((userId) => readFavorites(userId));

export const PUT = withUser(async (userId, request) =>
  writeFavorites(userId, body(await readJson(request))),
);

export const POST = withUser(async (userId, request) =>
  mergeFavorites(userId, body(await readJson(request))),
);
