"use client";

import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import {
  Check,
  Heart,
  LogIn,
  Monitor,
  Moon,
  Settings,
  Sun,
  UserPlus,
  UserRound,
} from "lucide-react";
import { useLocale, useTranslations } from "next-intl";
import { useTheme } from "next-themes";

import { usePathname, useRouter } from "@/i18n/navigation";
import { LOCALE_NAMES, routing, type Locale } from "@/i18n/routing";
import { useMounted } from "@/lib/use-mounted";
import { cn } from "@/lib/utils";

const itemClass =
  "flex cursor-pointer select-none items-center gap-2.5 rounded-lg px-2.5 py-2 " +
  "text-[0.83rem] text-muted outline-none transition-colors " +
  "data-[highlighted]:bg-surface-2 data-[highlighted]:text-text " +
  "data-[disabled]:pointer-events-none data-[disabled]:opacity-45";

const labelClass =
  "px-2.5 pb-1 pt-2 text-[0.63rem] font-semibold uppercase tracking-[0.11em] text-dim";

/**
 * Account, preferences and appearance in one place.
 *
 * Theme and language work today; the account items are shown as pending rather
 * than hidden, so the shape of the product is honest about what exists.
 */
export function AccountMenu() {
  const t = useTranslations("account");
  const tTheme = useTranslations("theme");
  const tLang = useTranslations("language");
  const { theme, setTheme } = useTheme();
  const locale = useLocale() as Locale;
  const router = useRouter();
  const pathname = usePathname();

  // The server cannot know the stored theme, so the checkmark is only rendered
  // once mounted; otherwise the first paint disagrees with the DOM.
  const mounted = useMounted();

  const themes = [
    { value: "light", label: tTheme("light"), icon: Sun },
    { value: "dark", label: tTheme("dark"), icon: Moon },
    { value: "system", label: tTheme("system"), icon: Monitor },
  ];

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button
          aria-label={t("label")}
          className={cn(
            "flex size-9 items-center justify-center rounded-full border border-border",
            "bg-surface text-muted transition-colors duration-200",
            "hover:border-border-strong hover:text-text",
            "data-[state=open]:border-brand data-[state=open]:text-brand",
          )}
        >
          <UserRound className="size-[1.05rem]" />
        </button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="end"
          sideOffset={10}
          className={cn(
            "z-50 min-w-56 rounded-[14px] border border-border bg-bg-elevated p-1.5",
            "shadow-[var(--shadow-lg)]",
            "data-[state=open]:animate-in data-[state=closed]:animate-out",
            "origin-[var(--radix-dropdown-menu-content-transform-origin)]",
            "transition-[opacity,transform] duration-150",
            "data-[state=closed]:scale-95 data-[state=closed]:opacity-0",
          )}
        >
          <div className={labelClass}>{t("label")}</div>
          <DropdownMenu.Item className={itemClass} disabled>
            <LogIn className="size-4" />
            {t("signIn")}
            <span className="ml-auto rounded-full bg-surface-2 px-1.5 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide text-dim">
              {t("soon")}
            </span>
          </DropdownMenu.Item>
          <DropdownMenu.Item className={itemClass} disabled>
            <UserPlus className="size-4" />
            {t("signUp")}
          </DropdownMenu.Item>
          <DropdownMenu.Item className={itemClass} disabled>
            <Heart className="size-4" />
            {t("favorites")}
          </DropdownMenu.Item>
          <DropdownMenu.Item className={itemClass} disabled>
            <Settings className="size-4" />
            {t("settings")}
          </DropdownMenu.Item>

          <DropdownMenu.Separator className="my-1.5 h-px bg-border" />

          <div className={labelClass}>{tTheme("label")}</div>
          {themes.map(({ value, label, icon: Icon }) => (
            <DropdownMenu.Item
              key={value}
              className={itemClass}
              onSelect={() => setTheme(value)}
            >
              <Icon className="size-4" />
              {label}
              {mounted && theme === value ? (
                <Check className="ml-auto size-3.5 text-brand" />
              ) : null}
            </DropdownMenu.Item>
          ))}

          <DropdownMenu.Separator className="my-1.5 h-px bg-border" />

          <div className={labelClass}>{tLang("label")}</div>
          {routing.locales.map((code) => (
            <DropdownMenu.Item
              key={code}
              className={itemClass}
              onSelect={() => router.replace(pathname, { locale: code })}
            >
              <span className="w-4 text-center text-[0.7rem] font-semibold uppercase text-dim">
                {code}
              </span>
              {LOCALE_NAMES[code]}
              {locale === code ? (
                <Check className="ml-auto size-3.5 text-brand" />
              ) : null}
            </DropdownMenu.Item>
          ))}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
