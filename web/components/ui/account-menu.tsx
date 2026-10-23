"use client";

import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import {
  Check,
  Heart,
  LogIn,
  LogOut,
  Monitor,
  Moon,
  Settings,
  Sun,
  UserPlus,
  UserRound,
} from "lucide-react";
import { useLocale, useTranslations } from "next-intl";
import { useTheme } from "next-themes";

import { Link, usePathname, useRouter } from "@/i18n/navigation";
import { LOCALE_NAMES, routing, type Locale } from "@/i18n/routing";
import { signOut, useSession } from "@/lib/auth-client";
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
 * The session is read in the browser, not on the server. This menu sits in the
 * locale layout above every football page, and those pages are prerendered
 * over the locales; a server-side session read here would make the whole site
 * render per request to put one name in a dropdown. So the menu ships its
 * signed-out shell in the static HTML and fills in after hydration — the same
 * trick the theme checkmark below already uses.
 *
 * Favourites and settings stay marked as pending: they are task B2, and
 * showing them greyed rather than hiding them keeps the shape of the product
 * honest about what exists.
 */
export function AccountMenu() {
  const t = useTranslations("account");
  const tAuth = useTranslations("auth");
  const tTheme = useTranslations("theme");
  const tLang = useTranslations("language");
  const { theme, setTheme } = useTheme();
  const locale = useLocale() as Locale;
  const router = useRouter();
  const { data: session } = useSession();
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
          {session ? (
            <>
              {/* The address, not the display name: it is what identifies the
                  account, and the name is already on the account page. */}
              <div className="truncate px-2.5 pb-1 pt-2 text-[0.72rem] text-dim">
                {session.user.email}
              </div>
              <DropdownMenu.Item className={itemClass} asChild>
                <Link href="/account">
                  <UserRound className="size-4" />
                  {tAuth("accountTitle")}
                </Link>
              </DropdownMenu.Item>
              <DropdownMenu.Item
                className={itemClass}
                onSelect={async () => {
                  await signOut();
                  router.push("/");
                  router.refresh();
                }}
              >
                <LogOut className="size-4" />
                {tAuth("signOut")}
              </DropdownMenu.Item>
            </>
          ) : (
            <>
              <div className={labelClass}>{t("label")}</div>
              <DropdownMenu.Item className={itemClass} asChild>
                <Link href="/sign-in">
                  <LogIn className="size-4" />
                  {t("signIn")}
                </Link>
              </DropdownMenu.Item>
              <DropdownMenu.Item className={itemClass} asChild>
                <Link href="/sign-up">
                  <UserPlus className="size-4" />
                  {t("signUp")}
                </Link>
              </DropdownMenu.Item>
            </>
          )}
          <DropdownMenu.Item className={itemClass} disabled>
            <Heart className="size-4" />
            {t("favorites")}
            <span className="ml-auto rounded-full bg-surface-2 px-1.5 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide text-dim">
              {t("soon")}
            </span>
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
