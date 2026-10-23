/**
 * Better Auth error codes, translated.
 *
 * The library's own `error.message` is English prose written for a developer.
 * The site is bilingual and every other string in it lives in
 * `messages/{en,es}.json`, so a failed sign-in is no different: the code
 * becomes a message key and anything unrecognised falls back to one honest
 * sentence rather than leaking library internals into the UI.
 */
const KEYS: Record<string, string> = {
  INVALID_EMAIL_OR_PASSWORD: "invalidCredentials",
  INVALID_PASSWORD: "invalidCredentials",
  USER_NOT_FOUND: "invalidCredentials",
  USER_ALREADY_EXISTS: "emailTaken",
  USER_ALREADY_EXISTS_USE_ANOTHER_EMAIL: "emailTaken",
  INVALID_EMAIL_FORMAT: "invalidEmail",
  PASSWORD_TOO_SHORT: "passwordTooShort",
  PASSWORD_TOO_LONG: "passwordTooLong",
  EMAIL_NOT_VERIFIED: "emailNotVerified",
  SESSION_NOT_FRESH: "sessionNotFresh",
};

/** A key under the `auth.errors` namespace. */
export function errorKey(code: string | undefined): string {
  return (code && KEYS[code]) || "unknown";
}
