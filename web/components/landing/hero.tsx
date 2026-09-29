import { ArrowRight } from "lucide-react";
import { getLocale, getTranslations } from "next-intl/server";

import { buttonStyles } from "@/components/ui/button";
import { Reveal } from "@/components/ui/reveal";
import { Link } from "@/i18n/navigation";
import { api } from "@/lib/api";
import { STATS } from "@/lib/stats";
import { cn, formatNumber } from "@/lib/utils";

export async function Hero() {
  const t = await getTranslations("landing");
  const locale = await getLocale();

  // The live count from the API; the constant is only for when it is down.
  const matches = await api
    .health()
    .then((h) => h.matches)
    .catch(() => STATS.matches);

  const stats = [
    { value: formatNumber(matches, locale), label: t("stats.matches") },
    { value: String(STATS.seasons), label: t("stats.seasons") },
    { value: `${STATS.markets}+`, label: t("stats.markets") },
    { value: String(STATS.leagues), label: t("stats.leagues") },
  ];

  return (
    <section className="relative overflow-hidden">
      {/* Structure, not decoration: a faint pitch grid that fades out before it
          reaches the text, so nothing has to fight it for contrast. */}
      <div
        aria-hidden
        className="bg-grid pointer-events-none absolute inset-0 [mask-image:radial-gradient(120%_80%_at_50%_0%,black,transparent_72%)]"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute left-1/2 top-[-18rem] size-[36rem] -translate-x-1/2 rounded-full bg-brand opacity-[0.16] blur-[140px]"
      />

      <div className="relative mx-auto max-w-[1240px] px-5 pb-20 pt-16 sm:px-7 sm:pb-28 sm:pt-24">
        <Reveal>
          <p className="text-[0.68rem] font-semibold uppercase tracking-[0.2em] text-brand">
            {t("eyebrow")}
          </p>
        </Reveal>

        <Reveal delay={0.06}>
          <h1 className="font-display mt-5 max-w-4xl text-[2.8rem] leading-[1.03] sm:text-[4.4rem]">
            {t("headline")}
            <br />
            <span className="text-gradient">{t("headlineAccent")}</span>
          </h1>
        </Reveal>

        <Reveal delay={0.12}>
          <p className="mt-7 max-w-2xl text-[1.02rem] leading-relaxed text-muted sm:text-[1.08rem]">
            {t("sub")}
          </p>
        </Reveal>

        <Reveal delay={0.18}>
          <div className="mt-9 flex flex-wrap items-center gap-3">
            <Link href="/matchday" className={buttonStyles("primary", "lg")}>
              {t("ctaPrimary")}
              <ArrowRight className="size-4" />
            </Link>
            <a href="#how" className={buttonStyles("secondary", "lg")}>
              {t("ctaSecondary")}
            </a>
          </div>
        </Reveal>

        <Reveal delay={0.24}>
          <dl className="mt-16 grid grid-cols-2 gap-px overflow-hidden rounded-[var(--radius-card)] border border-border bg-border sm:grid-cols-4">
            {stats.map((s) => (
              <div key={s.label} className="bg-bg-elevated px-5 py-6">
                <dt className="text-[0.66rem] font-semibold uppercase tracking-[0.11em] text-dim">
                  {s.label}
                </dt>
                <dd className="mt-2 text-[1.7rem] font-semibold leading-none tracking-[-0.03em]">
                  {s.value}
                </dd>
              </div>
            ))}
          </dl>
        </Reveal>

        <Reveal delay={0.3}>
          <p className={cn("mt-5 text-[0.78rem] text-dim")}>
            {t("heroNote", {
              matches: formatNumber(matches, locale),
              seasons: STATS.seasons,
            })}
          </p>
        </Reveal>
      </div>
    </section>
  );
}
