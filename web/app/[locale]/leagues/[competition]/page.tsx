import { ArrowLeft } from "lucide-react";
import type { Metadata } from "next";
import { getLocale, getTranslations, setRequestLocale } from "next-intl/server";
import { notFound } from "next/navigation";

import { PositionMatrix } from "@/components/leagues/position-matrix";
import { Reveal } from "@/components/ui/reveal";
import { Link } from "@/i18n/navigation";
import { api, type LeagueForecast } from "@/lib/api";
import { cn } from "@/lib/utils";

type Params = Promise<{ locale: string; competition: string }>;

const pct = (v: number | null, d = 0) =>
  v == null ? "—" : `${(v * 100).toFixed(d)}%`;

async function load(slug: string): Promise<LeagueForecast | null> {
  try {
    return await api.league(slug);
  } catch {
    return null;
  }
}

export async function generateMetadata({
  params,
}: {
  params: Params;
}): Promise<Metadata> {
  const { locale, competition } = await params;
  const data = await load(competition);
  if (!data) return {};
  const t = await getTranslations({ locale, namespace: "leagues" });
  const fav = data.standings.find((r) => r.pTitle != null);
  return {
    title: t("metaTitle", { league: data.competitionName }),
    description: fav
      ? t("metaDesc", {
          league: data.competitionName,
          team: fav.team,
          p: Math.round((fav.pTitle ?? 0) * 100),
        })
      : t("lead"),
  };
}

export default async function LeaguePage({ params }: { params: Params }) {
  const { locale: paramLocale, competition } = await params;
  setRequestLocale(paramLocale);
  const data = await load(competition);
  if (!data) notFound();
  const t = await getTranslations("leagues");
  const locale = await getLocale();

  const byTitle = [...data.standings]
    .filter((r) => r.pTitle != null)
    .sort((a, b) => (b.pTitle ?? 0) - (a.pTitle ?? 0));
  const favourite = byTitle[0];
  const leader = data.standings[0];
  const raceTop = byTitle.slice(0, 6);
  const raceLead = raceTop[0]?.pTitle ?? 1;

  return (
    <div className="mx-auto max-w-[1080px] px-5 py-10 sm:px-7 sm:py-14">
      <div className="flex items-center gap-3 text-[0.76rem] text-dim">
        <Link
          href="/leagues"
          className="flex items-center gap-1.5 transition-colors hover:text-text"
        >
          <ArrowLeft className="size-3.5" />
          {t("back")}
        </Link>
        <span>·</span>
        <span>{data.season}</span>
        {data.matchday ? (
          <>
            <span>·</span>
            <span>{t("matchday", { n: data.matchday })}</span>
          </>
        ) : null}
      </div>

      <h1 className="font-display mt-5 text-[2.2rem] leading-[1.08] sm:text-[2.9rem]">
        {t("h1", { league: data.competitionName })}
      </h1>

      {/* The headline first: one team, one number. Everything under it is the
          working that supports it. */}
      {favourite ? (
        <p className="mt-4 max-w-2xl text-[1.02rem] leading-relaxed text-muted">
          {t.rich("headline", {
            team: favourite.team,
            p: Math.round((favourite.pTitle ?? 0) * 100),
            remaining: data.remaining,
            b: (chunks) => (
              <span className="font-semibold text-text">{chunks}</span>
            ),
          })}
        </p>
      ) : null}

      <div className="mt-12 flex flex-col gap-16">
        {raceTop.length ? (
          <Reveal as="section">
            <h2 className="font-display text-[1.5rem] leading-tight sm:text-[1.9rem]">
              {t("raceTitle")}
            </h2>
            <div className="mt-6 flex flex-col gap-3">
              {raceTop.map((r) => (
                <div key={r.team} className="flex items-center gap-3 sm:gap-4">
                  <span className="w-28 shrink-0 truncate text-[0.88rem] font-medium sm:w-40">
                    {r.team}
                  </span>
                  <span className="h-2.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                    <span
                      className="block h-full rounded-full bg-brand"
                      style={{ width: `${((r.pTitle ?? 0) / raceLead) * 100}%` }}
                    />
                  </span>
                  <span className="w-12 text-right text-[0.85rem] font-semibold tabular-nums">
                    {pct(r.pTitle)}
                  </span>
                </div>
              ))}
            </div>
            <p className="mt-4 text-[0.78rem] text-dim">{t("raceNote")}</p>
          </Reveal>
        ) : null}

        <Reveal as="section">
          <h2 className="font-display text-[1.5rem] leading-tight sm:text-[1.9rem]">
            {t("tableTitle")}
          </h2>
          <div className="mt-5 overflow-x-auto rounded-[var(--radius-card)] border border-border">
            <table className="w-full min-w-[640px] text-[0.85rem]">
              <thead>
                <tr className="border-b border-border bg-surface-2 text-[0.64rem] uppercase tracking-[0.08em] text-dim">
                  <th className="px-3 py-2.5 text-left font-semibold">#</th>
                  <th className="px-3 py-2.5 text-left font-semibold">{t("team")}</th>
                  <th className="px-3 py-2.5 text-right font-semibold">{t("pj")}</th>
                  <th className="px-3 py-2.5 text-right font-semibold">{t("pts")}</th>
                  <th className="px-3 py-2.5 text-right font-semibold">{t("xPts")}</th>
                  <th className="px-3 py-2.5 text-right font-semibold">{t("colTitle")}</th>
                  <th className="px-3 py-2.5 text-right font-semibold">{t("colTop4")}</th>
                  <th className="px-3 py-2.5 text-right font-semibold">{t("colRel")}</th>
                </tr>
              </thead>
              <tbody>
                {data.standings.map((r) => (
                  <tr
                    key={r.team}
                    className="border-b border-border bg-surface last:border-0"
                  >
                    <td className="px-3 py-2.5 tabular-nums text-dim">{r.rank}</td>
                    <td className="px-3 py-2.5 font-medium">{r.team}</td>
                    <td className="px-3 py-2.5 text-right tabular-nums text-muted">
                      {r.played}
                    </td>
                    <td className="px-3 py-2.5 text-right font-semibold tabular-nums">
                      {r.points}
                    </td>
                    <td className="px-3 py-2.5 text-right tabular-nums text-muted">
                      {r.expectedPoints?.toFixed(0) ?? "—"}
                    </td>
                    <td
                      className={cn(
                        "px-3 py-2.5 text-right tabular-nums",
                        (r.pTitle ?? 0) > 0.05 ? "font-semibold text-brand" : "text-muted",
                      )}
                    >
                      {pct(r.pTitle)}
                    </td>
                    <td className="px-3 py-2.5 text-right tabular-nums text-muted">
                      {pct(r.pTop4)}
                    </td>
                    <td
                      className={cn(
                        "px-3 py-2.5 text-right tabular-nums",
                        (r.pRelegation ?? 0) > 0.15 ? "text-negative" : "text-muted",
                      )}
                    >
                      {pct(r.pRelegation)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[0.78rem] text-dim">
            {t("tableNote", { leader: leader.team, remaining: data.remaining })}
          </p>
        </Reveal>

        <PositionMatrix matrix={data.positionMatrix} />

        {data.upcoming.length ? (
          <Reveal as="section">
            <h2 className="font-display text-[1.5rem] leading-tight sm:text-[1.9rem]">
              {t("nextTitle")}
            </h2>
            <div className="mt-5 flex flex-col gap-1.5">
              {data.upcoming.map((f) => (
                <Link
                  key={f.slug}
                  href={`/matchday/${data.competition}/${f.slug}`}
                  className="flex items-center gap-3 rounded-[12px] border border-border bg-surface px-4 py-3.5 transition-colors duration-200 hover:border-border-strong hover:bg-surface-2 sm:gap-5 sm:px-5"
                >
                  <span className="hidden w-[4.5rem] shrink-0 text-[0.68rem] font-semibold uppercase tracking-[0.06em] text-dim sm:block">
                    {new Intl.DateTimeFormat(locale, {
                      day: "numeric",
                      month: "short",
                    }).format(new Date(`${f.date}T12:00:00`))}
                  </span>
                  <span className="flex-1 text-right text-[0.9rem] font-medium">
                    {f.home}
                  </span>
                  <span className="flex w-[5.5rem] shrink-0 items-center justify-center gap-1 text-[0.76rem] sm:w-[9rem] sm:gap-1.5 sm:text-[0.82rem]">
                    <b className="font-semibold">{Math.round(f.pHome * 100)}%</b>
                    <span className="text-dim">·</span>
                    <span className="text-muted">{Math.round(f.pDraw * 100)}%</span>
                    <span className="text-dim">·</span>
                    <b className="font-semibold">{Math.round(f.pAway * 100)}%</b>
                  </span>
                  <span className="flex-1 text-left text-[0.9rem] font-medium">
                    {f.away}
                  </span>
                </Link>
              ))}
            </div>
          </Reveal>
        ) : null}
      </div>
    </div>
  );
}
