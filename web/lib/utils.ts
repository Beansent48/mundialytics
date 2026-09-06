import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge conditional class names, letting later Tailwind classes win. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Locale-aware number formatting.
 *
 * Spanish groups thousands with a dot and English with a comma; hard-coding
 * either is how "15,371" ends up on a Spanish page meaning fifteen point three.
 */
export function formatNumber(
  value: number,
  locale: string,
  options?: Intl.NumberFormatOptions,
) {
  return new Intl.NumberFormat(locale, options).format(value);
}

/**
 * `1st`, `2nd`, `3rd`, `4th` — and `1º` in Spanish.
 *
 * Appending "th" to the number gives "1th", which is what shipped until this
 * existed. Intl knows the rule per locale; hard-coding a suffix cannot.
 */
export function ordinal(n: number, locale: string): string {
  if (!locale.startsWith("en")) {
    return `${n}º`;
  }
  const suffixes: Record<string, string> = {
    one: "st",
    two: "nd",
    few: "rd",
    other: "th",
  };
  const rule = new Intl.PluralRules("en", { type: "ordinal" }).select(n);
  return `${n}${suffixes[rule] ?? "th"}`;
}
