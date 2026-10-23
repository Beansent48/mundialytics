import { toNextJsHandler } from "better-auth/next-js";

import { auth } from "@/lib/auth";

/**
 * Every authentication endpoint: sign-in, sign-out, the Google callback, the
 * session read the browser makes.
 *
 * `/api/...` is already excluded from locale routing in `web/proxy.ts`, so
 * these URLs are locale-free and the Google redirect URI registered in the
 * Google Cloud console is a single, stable path.
 */
export const { GET, POST } = toNextJsHandler(auth);
