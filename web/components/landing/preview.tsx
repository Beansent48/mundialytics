import { ArrowRight } from "lucide-react";
import { getTranslations } from "next-intl/server";

import { buttonStyles } from "@/components/ui/button";
import { Reveal } from "@/components/ui/reveal";
import { Section, SectionHead } from "@/components/ui/section";
import { Link } from "@/i18n/navigation";
import { SAMPLE_FIXTURE as F } from "@/lib/stats";
import { cn } from "@/lib/utils";

const pct = (v: number) => `${Math.round(v * 100)}%`;

/**
 * The product itself, on the marketing page.
 *
 * Every number here is a real output of the deployed engine for a real fixture,
 * not an illustration — a landing page that mocks up its own product with
 * invented figures is the one thing this project cannot do and still call its
 * track record honest.
 */
export async function Preview() {
  const t = await getTranslations("landing.preview");

  const outcomes = [
    { label: F.home, p: F.pHome, color: "var(--home)" },
    { label: t("draw"), p: F.pDraw, color: "var(--draw)" },
    { label: F.away, p: F.pAway, color: "var(--away)" },
  ];

  return (
    <Section className="border-t border-border">
      <div className="grid items-center gap-12 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        <div>
          <SectionHead
            eyebrow={t("eyebrow")}
            title={t("title")}
            lead={t("lead")}
          />
          <Reveal delay={0.12}>
            <ul className="mt-7 flex flex-col gap-3">
              {(["b1", "b2", "b3"] as const).map((k) => (
                <li
                  key={k}
                  className="flex gap-3 text-[0.9rem] leading-relaxed text-muted"
                >
                  <span
                    aria-hidden
                    className="mt-[0.55rem] size-1.5 shrink-0 rounded-full bg-brand"
                  />
                  {t(k)}
                </li>
              ))}
            </ul>
            <Link
              href="/matchday"
              className={cn(buttonStyles("primary", "lg"), "mt-8")}
            >
              {t("cta")}
              <ArrowRight className="size-4" />
            </Link>
          </Reveal>
        </div>

        <Reveal delay={0.1}>
          <figure className="relative">
            <div
              aria-hidden
              className="pointer-events-none absolute -inset-6 rounded-[28px] bg-brand opacity-[0.07] blur-[60px]"
            />
            <div className="relative overflow-hidden rounded-[18px] border border-border bg-surface shadow-[var(--shadow-lg)]">
              <div className="flex items-center justify-between border-b border-border px-6 py-3.5">
                <span className="text-[0.68rem] font-semibold uppercase tracking-[0.13em] text-dim">
                  {F.competition}
                </span>
                <span className="flex items-center gap-1.5 text-[0.68rem] font-semibold uppercase tracking-[0.11em] text-brand">
                  <span className="size-1.5 rounded-full bg-brand" />
                  {t("badge")}
                </span>
              </div>

              <div className="px-6 py-6">
                <div className="flex items-baseline justify-between gap-4">
                  <span className="text-[1.15rem] font-semibold tracking-[-0.02em]">
                    {F.home}
                  </span>
                  <span className="text-[0.7rem] font-semibold uppercase tracking-[0.14em] text-dim">
                    vs
                  </span>
                  <span className="text-[1.15rem] font-semibold tracking-[-0.02em]">
                    {F.away}
                  </span>
                </div>

                {/* one stacked bar, three outcomes, widths straight from the model */}
                <div className="mt-5 flex h-11 overflow-hidden rounded-[10px]">
                  {outcomes.map((o) => (
                    <div
                      key={o.label}
                      className="flex items-center justify-center text-[0.8rem] font-semibold text-white"
                      style={{
                        width: `${o.p * 100}%`,
                        background: o.color,
                      }}
                    >
                      {pct(o.p)}
                    </div>
                  ))}
                </div>
                <div className="mt-2 flex justify-between text-[0.7rem] text-dim">
                  {outcomes.map((o) => (
                    <span key={o.label}>{o.label}</span>
                  ))}
                </div>

                <dl className="mt-6 grid grid-cols-3 gap-px overflow-hidden rounded-[10px] border border-border bg-border">
                  {[
                    {
                      k: t("xg"),
                      v: `${F.lambdaHome.toFixed(2)} – ${F.lambdaAway.toFixed(2)}`,
                    },
                    { k: t("over25"), v: pct(F.pOver25) },
                    { k: t("btts"), v: pct(F.pBtts) },
                  ].map((m) => (
                    <div key={m.k} className="bg-bg-elevated px-3 py-3.5">
                      <dt className="text-[0.6rem] font-semibold uppercase tracking-[0.1em] text-dim">
                        {m.k}
                      </dt>
                      <dd className="mt-1.5 text-[0.95rem] font-semibold">
                        {m.v}
                      </dd>
                    </div>
                  ))}
                </dl>

                <p className="mt-6 text-[0.63rem] font-semibold uppercase tracking-[0.12em] text-dim">
                  {t("scorelines")}
                </p>
                <div className="mt-2.5 flex flex-col gap-1.5">
                  {F.scorelines.map((s) => (
                    <div key={s.score} className="flex items-center gap-3">
                      <span className="w-9 text-[0.82rem] font-semibold">
                        {s.score}
                      </span>
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                        <div
                          className="h-full rounded-full bg-brand"
                          style={{
                            width: `${(s.p / F.scorelines[0].p) * 100}%`,
                          }}
                        />
                      </div>
                      <span className="w-11 text-right text-[0.78rem] text-muted">
                        {(s.p * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <figcaption className="mt-3 text-center text-[0.72rem] text-dim">
              {t("caption")}
            </figcaption>
          </figure>
        </Reveal>
      </div>
    </Section>
  );
}
