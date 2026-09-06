import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";

import { MatchdayClient } from "@/components/matchday/matchday-client";
import { api, type Competition } from "@/lib/api";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: "matchday" });
  return { title: t("title"), description: t("lead") };
}

export default async function MatchdayPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("matchday");

  // Today is resolved on the server so the page and the API agree on which day
  // "today" is, rather than trusting whatever clock the visitor's device keeps.
  const today = new Date().toISOString().slice(0, 10);

  let competitions: Competition[] = [];
  let offline = false;
  try {
    competitions = await api.competitions();
  } catch {
    offline = true;
  }

  return (
    <div className="mx-auto max-w-[1240px] px-5 py-12 sm:px-7 sm:py-16">
      <header className="max-w-2xl">
        <p className="text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-brand">
          {t("eyebrow")}
        </p>
        <h1 className="font-display mt-3 text-[2.2rem] leading-[1.1] sm:text-[2.8rem]">
          {t("title")}
        </h1>
        <p className="mt-4 text-[0.97rem] leading-relaxed text-muted">{t("lead")}</p>
      </header>

      {offline ? (
        <div className="mt-12 rounded-[14px] border border-border bg-surface px-6 py-12 text-center">
          <p className="text-[1rem] font-semibold">{t("offlineTitle")}</p>
          <p className="mx-auto mt-2 max-w-md text-[0.87rem] leading-relaxed text-muted">
            {t("offlineBody")}
          </p>
        </div>
      ) : (
        <div className="mt-10">
          <MatchdayClient today={today} competitions={competitions} />
        </div>
      )}
    </div>
  );
}
