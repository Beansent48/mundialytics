import { ArrowRight } from "lucide-react";
import { getLocale, getTranslations } from "next-intl/server";

import { buttonStyles } from "@/components/ui/button";
import { Reveal } from "@/components/ui/reveal";
import { Section, SectionHead } from "@/components/ui/section";
import { Link } from "@/i18n/navigation";
import { BENCHMARK, COVERAGE, STATS } from "@/lib/stats";
import { cn, formatNumber } from "@/lib/utils";

export async function Benchmark() {
  const t = await getTranslations("landing.benchmark");
  const locale = await getLocale();

  return (
    <Section id="how" className="border-t border-border bg-bg-elevated">
      <SectionHead
        eyebrow={t("eyebrow")}
        title={t("title")}
        lead={t("lead", {
          sample: formatNumber(STATS.benchmarkSample, locale),
        })}
      />

      <Reveal delay={0.08} className="mt-12">
        <div className="rounded-[var(--radius-card)] border border-border bg-surface p-6 sm:p-9">
          <div className="flex flex-col gap-5">
            {BENCHMARK.map((row, i) => {
              const highlight = "highlight" in row && row.highlight;
              const market = "market" in row && row.market;

              return (
                <div key={row.key} className="flex items-center gap-4 sm:gap-6">
                  <span
                    className={cn(
                      "w-32 shrink-0 text-[0.8rem] leading-snug sm:w-52 sm:text-[0.88rem]",
                      highlight ? "font-semibold text-text" : "text-muted",
                    )}
                  >
                    {t(row.key)}
                  </span>

                  <div className="relative h-9 flex-1 overflow-hidden rounded-lg bg-surface-2">
                    <div
                      className={cn(
                        "mv-bar-grow h-full rounded-lg",
                        highlight
                          ? "bg-brand"
                          : market
                            ? "bg-text"
                            : "bg-border-strong",
                      )}
                      style={
                        {
                          // a floor of 1.5% so "no information at all" is still
                          // a visible row rather than an empty track
                          "--w": `${Math.max(row.share * 100, 1.5)}%`,
                          transitionDelay: `${0.12 + i * 0.11}s`,
                        } as React.CSSProperties
                      }
                    />
                  </div>

                  <span
                    className={cn(
                      "w-[4.5rem] shrink-0 text-right text-[0.85rem] sm:text-[0.95rem]",
                      highlight ? "font-semibold text-brand" : "text-muted",
                    )}
                  >
                    {row.rps.toFixed(4)}
                  </span>
                </div>
              );
            })}
          </div>

          <p className="mt-5 text-[0.72rem] text-dim">{t("axis")}</p>

          <div className="rule-fade my-8" />

          <p className="text-[0.98rem] leading-relaxed">
            {t("conclusion", {
              gap: COVERAGE.span.toFixed(4),
              coverage: `${Math.round(COVERAGE.covered * 100)}%`,
            })}
          </p>
          <p className="mt-4 text-[0.83rem] leading-relaxed text-dim">
            {t("caveat")}
          </p>

          <Link
            href="/results"
            className={cn(buttonStyles("secondary", "md"), "mt-7")}
          >
            {t("cta")}
            <ArrowRight className="size-4" />
          </Link>
        </div>
      </Reveal>
    </Section>
  );
}
