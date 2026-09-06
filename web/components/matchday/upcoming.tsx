"use client";

import { useLocale, useTranslations } from "next-intl";
import { useMemo } from "react";

import { FixtureRow } from "@/components/matchday/fixture-row";
import type { UpcomingDay } from "@/lib/api";
import { useFavorites } from "@/lib/favorites";

function dayLabel(iso: string, locale: string, t: (k: string) => string) {
  const d = new Date(`${iso}T12:00:00`);
  const today = new Date();
  const diff = Math.round(
    (d.setHours(12, 0, 0, 0) - today.setHours(12, 0, 0, 0)) / 86_400_000,
  );
  if (diff === 0) return t("today");
  if (diff === 1) return t("tomorrow");
  if (diff === -1) return t("yesterday");
  return new Intl.DateTimeFormat(locale, {
    weekday: "long",
    day: "numeric",
    month: "short",
  }).format(new Date(`${iso}T12:00:00`));
}

/**
 * What is on now, plus a band at the top for the teams you follow.
 *
 * The followed band is a filter over the same fixtures rather than a second
 * request: a match involving your team should appear in both places, and
 * fetching it twice would be a second chance to disagree with itself.
 */
export function Upcoming({ days }: { days: UpcomingDay[] }) {
  const t = useTranslations("matchday");
  const locale = useLocale();
  const { has, ready } = useFavorites();

  const followed = useMemo(() => {
    if (!ready) return [];
    return days
      .flatMap((d) => d.fixtures)
      .filter((f) => has("teams", f.homeSlug) || has("teams", f.awaySlug));
  }, [days, has, ready]);

  return (
    <div className="flex flex-col gap-10">
      {followed.length > 0 ? (
        <section>
          <h2 className="text-[0.68rem] font-semibold uppercase tracking-[0.16em] text-warning">
            {t("yourTeams")}
          </h2>
          <div className="mt-3 flex flex-col gap-1.5">
            {followed.map((f) => (
              <FixtureRow key={`fav-${f.competition}-${f.slug}`} fixture={f} />
            ))}
          </div>
        </section>
      ) : null}

      {days.map((day) => (
        <section key={day.date}>
          <h2 className="flex items-center gap-3 text-[0.68rem] font-semibold uppercase tracking-[0.14em] text-muted">
            {dayLabel(day.date, locale, t)}
            <span className="h-px flex-1 bg-border" />
            <span className="text-dim">
              {t("matchCount", { count: day.fixtures.length })}
            </span>
          </h2>
          <div className="mt-3 flex flex-col gap-1.5">
            {day.fixtures.map((f) => (
              <FixtureRow key={`${f.competition}-${f.slug}`} fixture={f} />
            ))}
          </div>
        </section>
      ))}

      {days.length === 0 ? (
        <p className="rounded-[12px] border border-border bg-surface px-5 py-10 text-center text-[0.88rem] text-muted">
          {t("noUpcoming")}
        </p>
      ) : null}
    </div>
  );
}
