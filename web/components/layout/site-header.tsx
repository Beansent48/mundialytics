"use client";

import { Menu, X } from "lucide-react";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { AccountMenu } from "@/components/ui/account-menu";
import { buttonStyles } from "@/components/ui/button";
import { Wordmark } from "@/components/ui/logo";
import { Link, usePathname } from "@/i18n/navigation";
import { cn } from "@/lib/utils";

// Track record sits last on purpose: it is the reference section, the one you
// open to check the model rather than to look at football. SquadLab is play,
// so it goes next to the things you do.
const NAV = [
  { href: "/matchday", key: "matchday" },
  { href: "/leagues", key: "leagues" },
  { href: "/competitions", key: "competitions" },
  { href: "/awards", key: "awards" },
  { href: "/squadlab", key: "lab" },
  { href: "/results", key: "results" },
] as const;

export function SiteHeader() {
  const t = useTranslations("nav");
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  // Transparent over the hero, solid once you leave it — the header should not
  // draw a line across a full-bleed opening image.
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => setOpen(false), [pathname]);

  return (
    <header
      className={cn(
        "sticky top-0 z-40 transition-[background-color,border-color,backdrop-filter] duration-300",
        scrolled
          ? "border-b border-border bg-bg/80 backdrop-blur-xl"
          : "border-b border-transparent bg-transparent",
      )}
    >
      <div className="mx-auto flex h-16 max-w-[1240px] items-center gap-6 px-5 sm:px-7">
        <Link href="/" className="shrink-0" aria-label="Mundialytics">
          <Wordmark />
        </Link>

        <nav className="hidden flex-1 items-center gap-1 lg:flex">
          {NAV.map((item) => {
            const active = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "relative rounded-lg px-3 py-2 text-[0.855rem] font-medium transition-colors duration-200",
                  active ? "text-text" : "text-muted hover:text-text",
                )}
              >
                {t(item.key)}
                {active ? (
                  <span className="absolute inset-x-3 -bottom-px h-px bg-brand" />
                ) : null}
              </Link>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          <Link
            href="/matchday"
            className={cn(buttonStyles("primary", "sm"), "hidden sm:inline-flex")}
          >
            {t("openApp")}
          </Link>
          <AccountMenu />
          <button
            type="button"
            aria-label={open ? t("close") : t("menu")}
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
            className="flex size-9 items-center justify-center rounded-full border border-border bg-surface text-muted transition-colors hover:text-text lg:hidden"
          >
            {open ? <X className="size-4" /> : <Menu className="size-4" />}
          </button>
        </div>
      </div>

      {/* Mobile sheet: height transition rather than display toggling, so it
          opens and closes instead of blinking. */}
      <div
        className={cn(
          "overflow-hidden border-border bg-bg-elevated transition-[max-height,opacity] duration-300 ease-[var(--ease-out-quint)] lg:hidden",
          open ? "max-h-96 border-b opacity-100" : "max-h-0 opacity-0",
        )}
      >
        <nav className="mx-auto flex max-w-[1240px] flex-col gap-0.5 px-4 py-3">
          {NAV.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "rounded-lg px-3 py-2.5 text-[0.9rem] font-medium transition-colors",
                pathname.startsWith(item.href)
                  ? "bg-brand-ghost text-brand"
                  : "text-muted hover:bg-surface-2 hover:text-text",
              )}
            >
              {t(item.key)}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
