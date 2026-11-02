import { betterAuth } from "better-auth";
import { drizzleAdapter } from "better-auth/adapters/drizzle";
import { nextCookies } from "better-auth/next-js";

import { googleConfigured } from "@/lib/auth-config";
import { getDb } from "@/lib/db/client";
import * as schema from "@/lib/db/schema";
import { send } from "@/lib/mail";

/**
 * Authentication, server side.
 *
 * Better Auth over our own Postgres, inside the Next.js app. The Python API in
 * `api/` has no notion of a user and is not meant to grow one — the reasoning
 * is in docs/USER_ACCOUNTS_DESIGN.md §2.
 *
 * Why Better Auth and not Auth.js v5: Auth.js v5 is still published only under
 * the `beta` tag (5.0.0-beta.32 as of 2026-10-21, with `latest` still on v4),
 * its Credentials provider cannot use database sessions, and it leaves password
 * hashing to the application. Better Auth ships 1.7.12 as `latest`, has the
 * Drizzle adapter and email+password in core, and generates the schema the
 * adapter expects — so the schema in `lib/db/schema.ts` is a reviewable diff
 * rather than a guess.
 *
 * Nothing here gates a page. Everything on the site stays readable signed out;
 * the account unlocks saving, which is why there is no auth check in
 * `web/proxy.ts` (Next's own docs say Proxy "should not be used as a full
 * session management or authorization solution").
 */

// Declared only when configured. Better Auth would otherwise advertise a
// provider whose client id is the empty string, and the Google button would
// lead to Google's own error page instead of simply not being offered.
const google = googleConfigured()
  ? {
      google: {
        clientId: process.env.GOOGLE_CLIENT_ID as string,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET as string,
      },
    }
  : {};

export const auth = betterAuth({
  database: drizzleAdapter(getDb(), { provider: "pg", schema }),
  emailAndPassword: {
    enabled: true,
    // Off by default because the only way to reach a verification link in
    // development is the server log (see lib/mail.ts). Set
    // AUTH_REQUIRE_EMAIL_VERIFICATION=1 once a mail transport exists.
    requireEmailVerification: process.env.AUTH_REQUIRE_EMAIL_VERIFICATION === "1",
  },
  emailVerification: {
    sendOnSignUp: true,
    autoSignInAfterVerification: true,
    sendVerificationEmail: async ({ user, url }) => {
      await send({ to: user.email, subject: "Verify your email", url });
    },
  },
  socialProviders: google,
  user: {
    // The account has to be deletable, and deletable for real: every table
    // keyed on a user id cascades, and tests/account-cascade.test.mjs proves
    // it against a live database.
    deleteUser: { enabled: true },
  },
  account: {
    accountLinking: {
      // Signing up with an address and later using Google on the same address
      // is one person, not two accounts. Google verifies its addresses, which
      // is what makes this safe to do automatically.
      enabled: true,
      trustedProviders: ["google"],
    },
  },
  plugins: [nextCookies()],
});
