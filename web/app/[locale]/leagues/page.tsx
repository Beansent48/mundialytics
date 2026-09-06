import { ArrowRight } from "lucide-react";
import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";

import { Reveal } from "@/components/ui/reveal";
import { Link } from "@/i18n/navigation";
import { api, type LeagueCard } from "@/lib/api";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: "leagues" });
  return { title: t("title"), description: t("lead") };
}

export default async function LeaguesPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("leagues");

  let leagues: LeagueCard[] = [];
  let offline = false;
  try {
    leagues = await api.leagues();
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
        <div className="mt-12 grid gap-4 sm:grid-cols-2">
          {leagues.map((l, i) => (
            <Reveal key={l.slug} delay={i * 0.06}>
              <Link
                href={`/leagues/${l.slug}`}
                className="group flex h-full flex-col rounded-[var(--radius-card)] border border-border bg-surface p-6 transition-[border-color,background-color] duration-200 hover:border-border-strong hover:bg-surface-2"
              >
                <div className="flex items-baseline justify-between gap-3">
                  <h2 className="text-[1.15rem] font-semibold tracking-[-0.02em]">
                    {l.name}
                  </h2>
                  <span className="text-[0.68rem] font-semibold uppercase tracking-[0.1em] text-dim">
                    {l.season}
                  </span>
                </div>

                <dl className="mt-5 flex flex-col gap-3 text-[0.86rem]">
                  <div className="flex justify-between gap-3">
                    <dt className="text-dim">{t("leader")}</dt>
                    <dd className="font-medium">
                      {l.leader
                        ? `${l.leader.team} · ${t("points", { points: l.leader.points })}`
                        : "—"}
                    </dd>
                  </div>
                  <div className="flex justify-between gap-3">
                    <dt className="text-dim">{t("favourite")}</dt>
                    <dd className="font-medium">
                      {l.favourite && l.pTitle != null ? (
                        <>
                          {l.favourite}{" "}
                          <span className="text-brand">
                            {Math.round(l.pTitle * 100)}%
                          </span>
                        </>
                      ) : (
                        // Honest placeholder: the forecast has not been computed
                        // for this league yet, and inventing a favourite here
                        // would be the one thing this product cannot do.
                        <span className="text-dim">{t("notYet")}</span>
                      )}
                    </dd>
                  </div>
                </dl>

                <span className="mt-auto flex items-center gap-1.5 pt-6 text-[0.82rem] font-medium text-brand">
                  {t("open")}
                  <ArrowRight className="size-3.5 transition-transform duration-200 group-hover:translate-x-0.5" />
                </span>
              </Link>
            </Reveal>
          ))}
        </div>
      )}
    </div>
  );
}
