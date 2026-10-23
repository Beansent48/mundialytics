"use client";

import { useLocale } from "next-intl";
import { useTheme } from "next-themes";
import { useEffect, useRef } from "react";

import { usePathname, useRouter } from "@/i18n/navigation";
import { routing, type Locale } from "@/i18n/routing";
import { useSession } from "@/lib/auth-client";
import { fetchSettings } from "@/lib/settings-client";

/**
 * Bring the account's preferences down to this device, once per sign-in.
 *
 * The two are not symmetrical, and the difference is deliberate:
 *
 * - **Theme** is applied silently. It is invisible plumbing; nobody is
 *   surprised that their dark mode followed them.
 * - **Language** is applied only when this device has expressed no opinion.
 *   Re-rendering the page someone is reading into another language is the kind
 *   of "helpful" that feels like a bug, and next-intl resolves the locale from
 *   the URL and the cookie, so the cookie stays authoritative for the request
 *   in flight — which is what §6 of the design asks for. A visitor who picked a
 *   language on this device keeps it; their account still records their latest
 *   choice, because picking one while signed in writes it.
 *
 * Mounted once in the locale layout. Renders nothing.
 */
export function SettingsSync() {
  const { data: session } = useSession();
  const { setTheme } = useTheme();
  const locale = useLocale() as Locale;
  const router = useRouter();
  const pathname = usePathname();

  const userId = session?.user.id ?? null;
  // One sync per account per page load. `locale` and `pathname` are in the
  // dependencies below because the effect reads them, and they change as the
  // person browses — but this guard means a re-run does nothing, so the sync
  // still happens exactly once.
  const syncedFor = useRef<string | null>(null);

  useEffect(() => {
    if (!userId || syncedFor.current === userId) return;
    syncedFor.current = userId;

    let cancelled = false;
    (async () => {
      const settings = await fetchSettings();
      if (!settings || cancelled) {
        if (!settings) syncedFor.current = null;
        return;
      }

      if (settings.theme) setTheme(settings.theme);

      const wanted = settings.locale;
      if (
        wanted &&
        wanted !== locale &&
        routing.locales.includes(wanted as Locale) &&
        // "No opinion on this device" is the absence of the cookie next-intl
        // writes when somebody uses the language switcher.
        !document.cookie.includes("NEXT_LOCALE=")
      ) {
        router.replace(pathname, { locale: wanted as Locale });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [userId, router, setTheme, locale, pathname]);

  return null;
}
