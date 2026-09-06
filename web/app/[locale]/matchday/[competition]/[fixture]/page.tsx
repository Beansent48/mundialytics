import { ArrowLeft } from "lucide-react";
import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { notFound } from "next/navigation";

import { PlayedMatch } from "@/components/match/played";
import {
  GoalMarkets,
  Scorelines,
  Scorers,
  TeamMarkets,
} from "@/components/match/sections";
import { Verdict } from "@/components/match/verdict";
import { Link } from "@/i18n/navigation";
import { api, type Match } from "@/lib/api";

type Params = Promise<{ locale: string; competition: string; fixture: string }>;

async function load(competition: string, fixture: string): Promise<Match | null> {
  try {
    return await api.match(competition, fixture);
  } catch {
    return null;
  }
}

export async function generateMetadata({
  params,
}: {
  params: Params;
}): Promise<Metadata> {
  const { locale, competition, fixture } = await params;
  const match = await load(competition, fixture);
  if (!match) return {};
  const t = await getTranslations({ locale, namespace: "match" });

  // The description carries the actual numbers: a search result that already
  // says 27/28/45 earns the click that a generic blurb does not.
  const p = match.prediction.probabilities;
  const description = match.played
    ? t("metaPlayed", {
        home: match.home,
        away: match.away,
        score: `${match.score?.home}–${match.score?.away}`,
      })
    : t("metaUpcoming", {
        home: match.home,
        away: match.away,
        h: Math.round(p.home * 100),
        d: Math.round(p.draw * 100),
        a: Math.round(p.away * 100),
      });

  return {
    title: `${match.home} vs ${match.away}`,
    description,
    openGraph: { title: `${match.home} vs ${match.away}`, description },
  };
}

export default async function MatchPage({ params }: { params: Params }) {
  const { locale, competition, fixture } = await params;
  setRequestLocale(locale);
  const match = await load(competition, fixture);
  if (!match) notFound();
  const t = await getTranslations("match");

  return (
    <div className="mx-auto max-w-[1000px] px-5 py-10 sm:px-7 sm:py-14">
      <div className="flex items-center gap-3 text-[0.76rem] text-dim">
        <Link
          href="/matchday"
          className="flex items-center gap-1.5 transition-colors hover:text-text"
        >
          <ArrowLeft className="size-3.5" />
          {t("back")}
        </Link>
        <span>·</span>
        <span className="font-semibold uppercase tracking-[0.09em]">
          {match.competitionName}
        </span>
        {match.matchday ? (
          <>
            <span>·</span>
            <span>{t("round", { round: match.matchday })}</span>
          </>
        ) : null}
        {match.kickoff ? (
          <>
            <span>·</span>
            <time dateTime={match.kickoff}>
              {new Intl.DateTimeFormat(locale, {
                weekday: "short",
                day: "numeric",
                month: "short",
              }).format(new Date(`${match.kickoff}T12:00:00`))}
            </time>
          </>
        ) : null}
      </div>

      <div className="mt-7 flex flex-col gap-14">
        {match.played ? (
          <PlayedMatch match={match} />
        ) : (
          <>
            <Verdict match={match} />
            {/* Ordered by what a reader repeats, not by what took longest to
                build: who scores travels, a corners distribution does not. */}
            <Scorers match={match} />
            <GoalMarkets match={match} />
            <Scorelines match={match} />
            <TeamMarkets match={match} />
          </>
        )}
      </div>
    </div>
  );
}
