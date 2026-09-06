import { ArrowRight, ClipboardCheck, Database, Lock } from "lucide-react";
import { getTranslations, setRequestLocale } from "next-intl/server";

import { Benchmark } from "@/components/landing/benchmark";
import { Hero } from "@/components/landing/hero";
import { Preview } from "@/components/landing/preview";
import { buttonStyles } from "@/components/ui/button";
import { Reveal } from "@/components/ui/reveal";
import { Section, SectionHead } from "@/components/ui/section";
import { Link } from "@/i18n/navigation";

const MARKETS = [
  { key: "match", accent: "var(--home)" },
  { key: "goals", accent: "var(--positive)" },
  { key: "team", accent: "var(--warning)" },
  { key: "player", accent: "var(--brand)" },
] as const;

const PROOF = [
  { key: "point1", icon: ClipboardCheck },
  { key: "point2", icon: Lock },
  { key: "point3", icon: Database },
] as const;

export default async function LandingPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("landing");

  return (
    <>
      <Hero />
      <Benchmark />
      {/* Show the product before explaining it: a real fixture, priced, sits
          between the claim and the feature list. */}
      <Preview />

      {/* ── What it prices ────────────────────────────────────────────────── */}
      <Section className="border-t border-border bg-bg-elevated">
        <SectionHead
          eyebrow={t("markets.eyebrow")}
          title={t("markets.title")}
          lead={t("markets.lead")}
        />
        <div className="mt-12 grid gap-px overflow-hidden rounded-[var(--radius-card)] border border-border bg-border sm:grid-cols-2">
          {MARKETS.map((m, i) => (
            <Reveal key={m.key} delay={i * 0.07}>
              <article className="group h-full bg-surface p-7 transition-colors duration-300 hover:bg-surface-2">
                <span
                  aria-hidden
                  className="block h-1 w-9 rounded-full transition-[width] duration-500 ease-[var(--ease-out-quint)] group-hover:w-16"
                  style={{ background: m.accent }}
                />
                <h3 className="mt-5 text-[1.05rem] font-semibold tracking-[-0.015em]">
                  {t(`markets.${m.key}`)}
                </h3>
                <p className="mt-2 text-[0.88rem] leading-relaxed text-muted">
                  {t(`markets.${m.key}Desc`)}
                </p>
              </article>
            </Reveal>
          ))}
        </div>
        <Reveal delay={0.1}>
          <Link
            href="/matchday"
            className={`${buttonStyles("secondary", "md")} mt-8`}
          >
            {t("markets.cta")}
            <ArrowRight className="size-4" />
          </Link>
        </Reveal>
      </Section>

      {/* ── How it is verified ────────────────────────────────────────────── */}
      <Section className="border-t border-border">
        <SectionHead
          eyebrow={t("proof.eyebrow")}
          title={t("proof.title")}
          lead={t("proof.lead")}
        />
        <div className="mt-12 grid gap-8 sm:grid-cols-3">
          {PROOF.map((p, i) => (
            <Reveal key={p.key} delay={i * 0.08}>
              <div className="flex h-full flex-col">
                <span className="flex size-10 items-center justify-center rounded-[11px] bg-brand-ghost text-brand">
                  <p.icon className="size-[1.15rem]" />
                </span>
                <h3 className="mt-5 text-[0.98rem] font-semibold">
                  {t(`proof.${p.key}Title`)}
                </h3>
                <p className="mt-2 text-[0.87rem] leading-relaxed text-muted">
                  {t(`proof.${p.key}`)}
                </p>
              </div>
            </Reveal>
          ))}
        </div>
        <Reveal delay={0.12}>
          <Link
            href="/results"
            className={`${buttonStyles("secondary", "md")} mt-10`}
          >
            {t("proof.cta")}
            <ArrowRight className="size-4" />
          </Link>
        </Reveal>
      </Section>

      {/* ── Closing call to action ────────────────────────────────────────── */}
      <Section className="border-t border-border bg-bg-elevated">
        <Reveal>
          <div className="relative overflow-hidden rounded-[20px] border border-border bg-surface px-7 py-14 text-center sm:px-12 sm:py-20">
            <div
              aria-hidden
              className="pointer-events-none absolute left-1/2 top-full size-[28rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-brand opacity-[0.13] blur-[120px]"
            />
            <h2 className="font-display relative text-[2rem] leading-tight sm:text-[2.7rem]">
              {t("cta.title")}
            </h2>
            <p className="relative mx-auto mt-4 max-w-lg text-[0.95rem] text-muted">
              {t("cta.lead")}
            </p>
            <div className="relative mt-8 flex flex-wrap items-center justify-center gap-3">
              <Link href="/matchday" className={buttonStyles("primary", "lg")}>
                {t("cta.button")}
                <ArrowRight className="size-4" />
              </Link>
              <Link href="/results" className={buttonStyles("ghost", "lg")}>
                {t("cta.secondary")}
              </Link>
            </div>
            <p className="relative mt-6 text-[0.75rem] text-dim">
              {t("cta.note")}
            </p>
          </div>
        </Reveal>
      </Section>
    </>
  );
}
