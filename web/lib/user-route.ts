import { headers } from "next/headers";
import { NextResponse } from "next/server";

import { auth } from "@/lib/auth";

/**
 * The one place a request becomes a user id.
 *
 * Every `/api/user/*` handler goes through `withUser`, so there is a single
 * line to read when asking "can this request see that row?" rather than one
 * session check per route to keep in step. Signed out is 401 and not a
 * redirect: these are fetches from a page that stays perfectly usable signed
 * out, and a redirect would arrive at the caller as an HTML login page where
 * it expected JSON.
 */
export type UserHandler<T> = (
  userId: string,
  request: Request,
) => Promise<T> | T;

export function withUser<T>(handler: UserHandler<T>) {
  return async (request: Request): Promise<NextResponse> => {
    const session = await auth.api.getSession({ headers: await headers() });
    if (!session) {
      return NextResponse.json({ error: "not signed in" }, { status: 401 });
    }
    try {
      const body = await handler(session.user.id, request);
      return json(body);
    } catch (error) {
      if (error instanceof BadRequest) {
        return NextResponse.json({ error: error.message }, { status: 400 });
      }
      if (error instanceof NotFound) {
        return NextResponse.json({ error: "not found" }, { status: 404 });
      }
      throw error;
    }
  };
}

/** Never cached, by anyone, at any layer: this is one person's own data. */
function json(body: unknown): NextResponse {
  return NextResponse.json(body, {
    headers: { "cache-control": "no-store, private" },
  });
}

/** A body the client got wrong — answered as 400 rather than a 500 stack. */
export class BadRequest extends Error {}

/**
 * A row this user does not have.
 *
 * Thrown for a row that belongs to somebody else as well as for one that never
 * existed, on purpose: 404 for both is the only answer that does not confirm
 * the existence of another account's data.
 */
export class NotFound extends Error {}

export async function readJson(request: Request): Promise<Record<string, unknown>> {
  let parsed: unknown;
  try {
    parsed = await request.json();
  } catch {
    throw new BadRequest("body is not JSON");
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new BadRequest("body must be a JSON object");
  }
  return parsed as Record<string, unknown>;
}

/** A list of names, validated rather than trusted: this goes into the database. */
export function stringList(value: unknown, field: string): string[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new BadRequest(`${field} must be an array`);
  if (value.length > 500) throw new BadRequest(`${field} has too many entries`);
  const out: string[] = [];
  for (const item of value) {
    if (typeof item !== "string") {
      throw new BadRequest(`${field} must contain only strings`);
    }
    const trimmed = item.trim();
    if (trimmed && trimmed.length <= 200) out.push(trimmed);
  }
  return Array.from(new Set(out));
}
