import { getTranslations } from "next-intl/server";

import { Wordmark } from "@/components/ui/logo";
import { Link } from "@/i18n/navigation";

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

export async function SiteFooter() {
  const t = await getTranslations("landing.footer");
  const tNav = await getTranslations("nav");
  const tBrand = await getTranslations("brand");
  const year = new Date().getFullYear();

  return (
    <footer className="border-t border-border bg-bg-elevated">
      <div className="mx-auto max-w-[1240px] px-5 py-12 sm:px-7">
        <div className="flex flex-col gap-10 sm:flex-row sm:justify-between">
          <div className="max-w-xs">
            <Wordmark tagline={tBrand("tagline")} />
            <p className="mt-4 text-[0.8rem] leading-relaxed text-dim">
              {t("built")}
            </p>
          </div>

          <nav className="flex flex-wrap gap-x-8 gap-y-2">
            {NAV.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="text-[0.83rem] text-muted transition-colors hover:text-text"
              >
                {tNav(item.key)}
              </Link>
            ))}
          </nav>
        </div>

        <div className="mt-10 flex flex-col gap-2 border-t border-border pt-6 text-[0.73rem] text-dim sm:flex-row sm:items-center sm:justify-between">
          <span>
            © {year} {tBrand("name")}. {t("rights")}
          </span>
          {/* Stated plainly and kept in the footer of every page: the product is
              analysis, and saying so is not optional for this category. */}
          <span>{t("disclaimer")}</span>
        </div>
      </div>
    </footer>
  );
}
