"use client";

import { Check, Monitor, Moon, Sun } from "lucide-react";
import { useLocale, useTranslations } from "next-intl";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

import { usePathname, useRouter } from "@/i18n/navigation";
import { LOCALE_NAMES, routing, type Locale } from "@/i18n/routing";
import { useSession } from "@/lib/auth-client";
import { fetchSettings, saveSettings } from "@/lib/settings-client";
import { useMounted } from "@/lib/use-mounted";
import { cn } from "@/lib/utils";

const ODDS_FORMATS = [
  { value: "decimal", key: "oddsDecimal" },
  { value: "fractional", key: "oddsFractional" },
  { value: "american", key: "oddsAmerican" },
  { value: "probability", key: "oddsProbability" },
] as const;

const optionClass =
  "flex cursor-pointer items-center gap-2.5 rounded-[10px] border px-3.5 py-2.5 " +
  "text-[0.85rem] transition-colors duration-150";

const activeClass = "border-brand bg-brand-ghost text-text";
const inactiveClass = "border-border bg-surface text-muted hover:border-border-strong hover:text-text";

export function PreferencesForm() {
  const t = useTranslations("settings");
  const { theme, setTheme } = useTheme();
  const locale = useLocale() as Locale;
  const router = useRouter();
  const pathname = usePathname();
  const { data: session } = useSession();
  const mounted = useMounted();

  const [oddsFormat, setOddsFormat] = useState<string>("decimal");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (!session) return;
    fetchSettings().then((s) => {
      if (s?.oddsFormat) setOddsFormat(s.oddsFormat);
    });
  }, [session]);

  const flash = () => {
    setSaved(true);
    const id = window.setTimeout(() => setSaved(false), 1200);
    return () => window.clearTimeout(id);
  };

  const themes = [
    { value: "light", label: t("themeLabel") + ": " + t("themeLabel"), icon: Sun },
    { value: "dark", label: t("themeLabel"), icon: Moon },
    { value: "system", label: t("themeLabel"), icon: Monitor },
  ];

  const themeItems = [
    { value: "light", icon: Sun },
    { value: "dark", icon: Moon },
    { value: "system", icon: Monitor },
  ];

  const tTheme = useTranslations("theme");

  return (
    <div className="space-y-8">
      {/* Theme */}
      <div>
        <h3 className="text-[0.7rem] font-semibold uppercase tracking-[0.11em] text-dim">
          {t("themeLabel")}
        </h3>
        <div className="mt-2.5 flex flex-wrap gap-2">
          {themeItems.map(({ value, icon: Icon }) => {
            const active = mounted && theme === value;
            return (
              <button
                key={value}
                type="button"
                onClick={() => {
                  setTheme(value);
                  if (session) {
                    saveSettings({ theme: value });
                    flash();
                  }
                }}
                className={cn(optionClass, active ? activeClass : inactiveClass)}
              >
                <Icon className="size-4" />
                {tTheme(value as "light" | "dark" | "system")}
                {active ? <Check className="ml-auto size-3.5 text-brand" /> : null}
              </button>
            );
          })}
        </div>
      </div>

      {/* Language */}
      <div>
        <h3 className="text-[0.7rem] font-semibold uppercase tracking-[0.11em] text-dim">
          {t("languageLabel")}
        </h3>
        <div className="mt-2.5 flex flex-wrap gap-2">
          {routing.locales.map((code) => {
            const active = locale === code;
            return (
              <button
                key={code}
                type="button"
                onClick={() => {
                  router.replace(pathname, { locale: code });
                  if (session) {
                    saveSettings({ locale: code });
                    flash();
                  }
                }}
                className={cn(optionClass, active ? activeClass : inactiveClass)}
              >
                <span className="w-4 text-center text-[0.7rem] font-semibold uppercase text-dim">
                  {code}
                </span>
                {LOCALE_NAMES[code]}
                {active ? <Check className="ml-auto size-3.5 text-brand" /> : null}
              </button>
            );
          })}
        </div>
      </div>

      {/* Odds format */}
      <div>
        <h3 className="text-[0.7rem] font-semibold uppercase tracking-[0.11em] text-dim">
          {t("oddsFormatLabel")}
        </h3>
        <div className="mt-2.5 flex flex-wrap gap-2">
          {ODDS_FORMATS.map(({ value, key }) => {
            const active = oddsFormat === value;
            return (
              <button
                key={value}
                type="button"
                onClick={() => {
                  setOddsFormat(value);
                  if (session) {
                    saveSettings({ oddsFormat: value });
                    flash();
                  }
                }}
                className={cn(optionClass, active ? activeClass : inactiveClass)}
              >
                {t(key)}
                {active ? <Check className="ml-auto size-3.5 text-brand" /> : null}
              </button>
            );
          })}
        </div>
      </div>

      {/* Saved indicator */}
      <div
        className={cn(
          "text-[0.78rem] font-medium text-positive transition-opacity duration-300",
          saved ? "opacity-100" : "opacity-0",
        )}
      >
        <Check className="mb-0.5 mr-1 inline size-3.5" />
        {t("saved")}
      </div>
    </div>
  );
}
