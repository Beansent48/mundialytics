import { createAuthClient } from "better-auth/react";

/**
 * Authentication, browser side.
 *
 * No `baseURL`: the client defaults to the origin it was served from, which is
 * what we want in development, in a production build and behind a domain
 * without three places to keep in step.
 *
 * `useSession` is a client-side read on purpose. The header lives in the locale
 * layout, and every football page under it is prerendered through
 * `generateStaticParams`; a server-side session read there would make the whole
 * site dynamic to show one name in a menu. So the header renders its
 * signed-out shell statically and fills in after hydration — the same trick
 * the theme checkmark already uses.
 */
export const authClient = createAuthClient();

export const { signIn, signUp, signOut, useSession, deleteUser } = authClient;
