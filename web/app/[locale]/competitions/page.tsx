import { ArrowRight } from "lucide-react";
import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";

import { Reveal } from "@/components/ui/reveal";
import { Link } from "@/i18n/navigation";
import { api, type UefaCard } from "@/lib/api";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: "competitions" });
  return { title: t("title"), description: t("lead") };
}

export default async function CompetitionsPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("competitions");

  let cards: UefaCard[] = [];
  let offline = false;
  try {
    cards = await api.uefa();
  } catch {
    offline = true;
  }

  return (
    <div className="mx-auto max-w-[1000px] px-5 py-12 sm:px-7 sm:py-16">
      <header className="max-w-2xl">
        <p className="text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-brand">
          {t("eyebrow")}
        </p>
        <h1 className="font-display mt-3 text-[2.2rem] leading-[1.1] sm:text-[2.9rem]">
          {t("title")}
        </h1>
        <p className="mt-4 text-[0.97rem] leading-relaxed text-muted">{t("lead")}</p>
      </header>

      {offline ? (
        <div className="mt-12 rounded-[14px] border border-border bg-surface px-6 py-12 text-center">
          <p className="text-[1rem] font-semibold">{t("offlineTitle")}</p>
          <p className="mx-auto mt-2 max-w-md text-[0.87rem] text-muted">
            {t("offlineBody")}
          </p>
        </div>
      ) : (
        <>
          <div className="mt-12 grid gap-4 sm:grid-cols-3">
            {cards.map((c, i) => (
              <Reveal key={c.slug} delay={i * 0.06}>
                <Link
                  href={`/competitions/${c.slug}`}
                  className="group flex h-full flex-col rounded-[var(--radius-card)] border border-border bg-surface p-6 transition-[border-color,background-color] duration-200 hover:border-border-strong hover:bg-surface-2"
                >
                  <span className="text-[0.66rem] font-semibold uppercase tracking-[0.1em] text-dim">
                    {c.season}
                  </span>
                  <h2 className="mt-2 text-[1.1rem] font-semibold leading-snug tracking-[-0.02em]">
                    {c.name}
                  </h2>

                  <div className="mt-5 flex flex-col gap-1">
                    <span className="text-[0.66rem] font-semibold uppercase tracking-[0.1em] text-dim">
                      {t("favourite")}
                    </span>
                    {c.favourite && c.pChampion != null ? (
                      <span className="text-[1rem] font-semibold">
                        {c.favourite}{" "}
                        <span className="text-brand">
                          {Math.round(c.pChampion * 100)}%
                        </span>
                      </span>
                    ) : (
                      <span className="text-[0.88rem] text-dim">{t("notYet")}</span>
                    )}
                  </div>

                  <span className="mt-auto flex items-center gap-1.5 pt-6 text-[0.82rem] font-medium text-brand">
                    {t("open")}
                    <ArrowRight className="size-3.5 transition-transform duration-200 group-hover:translate-x-0.5" />
                  </span>
                </Link>
              </Reveal>
            ))}
          </div>

          <Reveal delay={0.15}>
            <p className="mt-10 max-w-2xl text-[0.85rem] leading-relaxed text-dim">
              {t("methodNote")}
            </p>
          </Reveal>
        </>
      )}
    </div>
  );
}
