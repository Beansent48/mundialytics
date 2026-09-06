import { getLocale, getTranslations } from "next-intl/server";

import { Reveal } from "@/components/ui/reveal";
import type { Benchmark } from "@/lib/api";
import { cn, formatNumber } from "@/lib/utils";

/**
 * The RPS ladder, ordered best first and encoded as distance covered.
 *
 * Lower RPS is better, so a bar whose length *is* the RPS makes the best
 * predictor the shortest — and a reader counting bar lengths puts the engine
 * third. The bar is therefore the share of the achievable distance covered,
 * from no information at all to the closing price, and the raw RPS sits on the
 * right so nothing is hidden by the choice.
 */
export async function RpsLadder({ benchmark }: { benchmark: Benchmark }) {
  const t = await getTranslations("results.benchmark");
  const locale = await getLocale();

  const rps = Object.fromEntries(benchmark.rows.map((r) => [r.key, r.rps]));
  const span = rps.uniform - rps.market;
  const share = (v: number) => (rps.uniform - v) / span;
  const covered = share(rps.engine);

  return (
    <Reveal as="section">
      <h2 className="font-display text-[1.6rem] leading-tight sm:text-[2rem]">
        {t("title")}
      </h2>
      <p className="mt-3 max-w-2xl text-[0.92rem] leading-relaxed text-muted">
        {t("lead", {
          sample: formatNumber(benchmark.sample, locale),
          window: benchmark.window,
        })}
      </p>

      <div className="mt-8 rounded-[var(--radius-card)] border border-border bg-surface p-6 sm:p-8">
        <div className="flex flex-col gap-5">
          {benchmark.rows.map((row, i) => {
            const isEngine = row.key === "engine";
            const isMarket = row.key === "market";
            return (
              <div key={row.key} className="flex items-center gap-4 sm:gap-6">
                <span
                  className={cn(
                    "w-32 shrink-0 text-[0.8rem] leading-snug sm:w-52 sm:text-[0.88rem]",
                    isEngine ? "font-semibold text-text" : "text-muted",
                  )}
                >
                  {t(row.key)}
                </span>
                <div className="relative h-9 flex-1 overflow-hidden rounded-lg bg-surface-2">
                  <div
                    className={cn(
                      "mv-bar-grow h-full rounded-lg",
                      isEngine ? "bg-brand" : isMarket ? "bg-text" : "bg-border-strong",
                    )}
                    style={
                      {
                        "--w": `${Math.max(share(row.rps) * 100, 1.5)}%`,
                        transitionDelay: `${0.12 + i * 0.1}s`,
                      } as React.CSSProperties
                    }
                  />
                </div>
                <span
                  className={cn(
                    "w-[4.5rem] shrink-0 text-right text-[0.85rem] tabular-nums sm:text-[0.95rem]",
                    isEngine ? "font-semibold text-brand" : "text-muted",
                  )}
                >
                  {row.rps.toFixed(4)}
                </span>
              </div>
            );
          })}
        </div>
        <p className="mt-5 text-[0.72rem] text-dim">{t("axis")}</p>

        <div className="rule-fade my-7" />

        <p className="text-[0.95rem] leading-relaxed">
          {t("conclusion", {
            gap: span.toFixed(4),
            coverage: `${Math.round(covered * 100)}%`,
          })}
        </p>

        <h3 className="mt-8 text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
          {t("byLeague")}
        </h3>
        <div className="mt-3 flex flex-col gap-2">
          {benchmark.byLeague.map((l) => (
            <div key={l.league} className="flex items-center gap-3">
              <span className="w-32 shrink-0 text-[0.82rem] text-muted sm:w-40">
                {l.league}
              </span>
              <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                <span
                  className="block h-full rounded-full bg-brand"
                  style={{ width: `${(l.gap / 0.012) * 100}%` }}
                />
              </span>
              <span className="w-14 text-right text-[0.8rem] tabular-nums text-muted">
                {l.gap.toFixed(4)}
              </span>
            </div>
          ))}
        </div>
        <p className="mt-3 text-[0.72rem] leading-relaxed text-dim">
          {t("byLeagueNote")}
        </p>
      </div>
    </Reveal>
  );
}
