/**
 * What the account layer is configured to offer.
 *
 * Separate from `lib/auth.ts` because a page that only needs to know whether
 * to draw the Google button should not pull in the Better Auth instance and,
 * through it, the database connection. Server-only: these are secrets'
 * neighbours, not `NEXT_PUBLIC_` values.
 */
export function googleConfigured(): boolean {
  return Boolean(process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET);
}
