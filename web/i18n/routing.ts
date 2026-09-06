import { defineRouting } from "next-intl/routing";

/**
 * Locales the product ships in.
 *
 * English is the default because the app is meant to be shown and sold beyond
 * Spain; Spanish is first-class, not a translation afterthought. Adding a
 * locale is this array plus a messages file — nothing in the components knows
 * how many there are.
 */
export const routing = defineRouting({
  locales: ["en", "es"],
  defaultLocale: "en",
  // "/matchday" for English, "/es/matchday" for Spanish: the default locale
  // keeps clean URLs, which is what a marketing page wants.
  localePrefix: "as-needed",
});

export type Locale = (typeof routing.locales)[number];

export const LOCALE_NAMES: Record<Locale, string> = {
  en: "English",
  es: "Español",
};
