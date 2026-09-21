"use client";

import { Star } from "lucide-react";
import { useTranslations } from "next-intl";

import { Link } from "@/i18n/navigation";
import type { Fixture } from "@/lib/api";
import { useFavorites } from "@/lib/favorites";
import { cn } from "@/lib/utils";

/**
 * One fixture in a list.
 *
 * The whole row is the link — a row you can only enter through a small button
 * on its right makes people aim. The star sits outside that link so following a
 * team never navigates by accident.
 */
export function FixtureRow({
  fixture,
  hideCompetition = false,
}: {
  fixture: Fixture;
  /** The day view groups by competition, so repeating it on every row is noise. */
  hideCompetition?: boolean;
}) {
  const t = useTranslations("matchday");
  const { has, toggle, ready } = useFavorites();
  const followed =
    ready && (has("teams", fixture.homeSlug) || has("teams", fixture.awaySlug));

  const p = fixture.probabilities;
  // European ties have no match page of their own -- they are priced off the
  // Elo scale, not the big-five engine, so there is no analysis to open. The
  // row still leads somewhere: its competition, on the round it belongs to.
  const analysed = fixture.analysis !== false;
  const href = analysed
    ? `/matchday/${fixture.competition}/${fixture.slug}`
    : `/competitions/${fixture.competition}${
        fixture.matchday != null ? `?matchday=${fixture.matchday}` : ""
      }`;

  return (
    <div className="group relative flex items-stretch">
      <Link
        href={href}
        className={cn(
          "flex flex-1 items-center gap-3 rounded-[12px] border border-border bg-surface",
          "px-4 py-3.5 transition-[border-color,background-color,transform] duration-200",
          "hover:border-border-strong hover:bg-surface-2",
          "sm:gap-5 sm:px-5",
        )}
      >
        {hideCompetition ? null : (
          <span className="hidden w-[4.5rem] shrink-0 text-[0.66rem] font-semibold uppercase tracking-[0.09em] text-dim sm:block">
            {fixture.competitionName}
          </span>
        )}

        <span className="flex-1 text-right text-[0.9rem] font-medium sm:text-[0.95rem]">
          {fixture.home}
        </span>

        <span className="w-[5.5rem] shrink-0 sm:w-[9rem]">
          {fixture.played && fixture.score ? (
            <span className="mx-auto flex w-fit items-center gap-1.5 rounded-[7px] border border-border bg-bg px-2.5 py-1 text-[0.88rem] font-semibold">
              {fixture.score.home}
              <span className="text-dim">–</span>
              {fixture.score.away}
            </span>
          ) : p ? (
            <span className="flex items-center justify-center gap-1 text-[0.76rem] sm:gap-1.5 sm:text-[0.82rem]">
              <b className="font-semibold text-text">{Math.round(p.home * 100)}%</b>
              <span className="text-dim">·</span>
              <span className="text-muted">{Math.round(p.draw * 100)}%</span>
              <span className="text-dim">·</span>
              <b className="font-semibold text-text">{Math.round(p.away * 100)}%</b>
            </span>
          ) : (
            <span className="block text-center text-[0.7rem] font-semibold uppercase tracking-[0.1em] text-dim">
              {t("vs")}
            </span>
          )}
        </span>

        <span className="flex-1 text-left text-[0.9rem] font-medium sm:text-[0.95rem]">
          {fixture.away}
        </span>
      </Link>

      <button
        type="button"
        aria-label={t("follow")}
        aria-pressed={followed}
        onClick={() => toggle("teams", fixture.homeSlug)}
        className={cn(
          "ml-1.5 flex w-10 items-center justify-center rounded-[12px] border border-transparent",
          "text-dim transition-colors duration-200 hover:border-border hover:text-warning",
          followed && "text-warning",
        )}
      >
        <Star
          className="size-4"
          fill={followed ? "currentColor" : "none"}
          strokeWidth={followed ? 0 : 2}
        />
      </button>
    </div>
  );
}
