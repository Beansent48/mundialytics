import { routing } from "@/i18n/routing";
import { readSettings, writeSettings, type Settings } from "@/lib/user-data";
import { BadRequest, readJson, withUser } from "@/lib/user-route";

/**
 * Preferences that follow the account across devices.
 *
 * `PUT` is a partial update: the theme switch and the language switch are
 * separate controls in the same menu, and neither should have to know the
 * other's current value to save.
 *
 * `null` is a real value here and means "no choice recorded" — follow the
 * browser for language, the system for theme — which is what a signed-out
 * visitor gets. Storing the absence of a choice as a choice would let the
 * account override a preference the person never expressed.
 */
export const dynamic = "force-dynamic";

const THEMES = ["light", "dark", "system"];

function optional(
  payload: Record<string, unknown>,
  field: string,
  allowed: readonly string[] | null,
): string | null | undefined {
  if (!(field in payload)) return undefined;
  const value = payload[field];
  if (value === null) return null;
  if (typeof value !== "string") {
    throw new BadRequest(`${field} must be a string or null`);
  }
  if (allowed && !allowed.includes(value)) {
    throw new BadRequest(`${field} must be one of ${allowed.join(", ")}`);
  }
  if (!allowed && value.length > 32) {
    throw new BadRequest(`${field} is too long`);
  }
  return value;
}

export const GET = withUser((userId) => readSettings(userId));

export const PUT = withUser(async (userId, request) => {
  const payload = await readJson(request);
  const patch: Partial<Settings> = {};
  // Only the fields that were named: `undefined` means "not mentioned", which
  // is different from `null`, "cleared".
  const locale = optional(payload, "locale", routing.locales);
  if (locale !== undefined) patch.locale = locale;
  const theme = optional(payload, "theme", THEMES);
  if (theme !== undefined) patch.theme = theme;
  const oddsFormat = optional(payload, "oddsFormat", null);
  if (oddsFormat !== undefined) patch.oddsFormat = oddsFormat;
  return writeSettings(userId, patch);
});
