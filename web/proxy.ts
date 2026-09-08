import createMiddleware from "next-intl/middleware";
import { NextResponse, type NextRequest } from "next/server";

import { routing } from "./i18n/routing";

const intl = createMiddleware(routing);

/**
 * Locale routing for every page request.
 *
 * The exclusions are done here, in code, rather than in `config.matcher`. The
 * documented negative-lookahead pattern silently fails to match multi-segment
 * paths — Next compiles the matcher with path-to-regexp semantics, where `.`
 * does not cross a `/` — so `/foo/bar` never reached this proxy, was never
 * rewritten into a locale, and fell through to Next's own bare 404 instead of
 * ours. Matching `/:path*` catches every depth; the filtering below keeps
 * assets and API routes out of it.
 */
export default function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  if (
    pathname.startsWith("/_next") ||
    pathname.startsWith("/_vercel") ||
    pathname.startsWith("/api") ||
    // anything with an extension is a file: favicon.ico, robots.txt, og images
    pathname.includes(".")
  ) {
    return NextResponse.next();
  }

  return intl(request);
}

export const config = {
  matcher: ["/", "/:path*"],
};
