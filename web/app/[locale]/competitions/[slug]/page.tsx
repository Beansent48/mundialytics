import { ArrowLeft } from "lucide-react";
import type { Metadata } from "next";
import { getLocale, getTranslations, setRequestLocale } from "next-intl/server";
import { notFound } from "next/navigation";

import { Reveal } from "@/components/ui/reveal";
import { Link } from "@/i18n/navigation";
import { api, type UefaCompetition } from "@/lib/api";
import { cn } from "@/lib/utils";

type Params = Promise<{ locale: string; slug: string }>;

const pct = (v: number | null, d = 0) =>
  v == null ? "—" : v < 0.005 ? "—" : `${(v * 100).toFixed(d)}%`;

async function load(slug: string): Promise<UefaCompetition | null> {
  try {
    return await api.uefaCompetition(slug);
  } catch {
    return null;
  }
}

export async function generateMetadata({
  params,
}: {
  params: Params;
}): Promise<Metadata> {
  const { locale, slug } = await params;
  const data = await load(slug);
  if (!data) return {};
  const t = await getTranslations({ locale, namespace: "competitions" });
  const fav = data.standings[0];
  return {
    title: t("metaTitle", { competition: data.name }),
    description: fav?.champion
      ? t("metaDesc", {
          competition: data.name,
          team: fav.team,
          p: Math.round(fav.champion * 100),
        })
      : t("lead"),
  };
}

const ROUND_COLUMNS = [
  "leaguePhase",
  "topEight",
  "roundOf16",
  "quarterFinal",
  "semiFinal",
  "final",
  "champion",
] as const;

export default async function CompetitionPage({ params }: { params: Params }) {
  const { locale: paramLocale, slug } = await params;
  setRequestLocale(paramLocale);
  const data = await load(slug);
  if (!data) notFound();
  const t = await getTranslations("competitions");
  const locale = await getLocale();

  const ranked = [...data.standings].sort(
    (a, b) => (b.champion ?? 0) - (a.champion ?? 0),
  );
  const favourite = ranked[0];
  const race = ranked.slice(0, 8);
  const raceLead = race[0]?.champion ?? 1;

  return (
    <div className="mx-auto max-w-[1120px] px-5 py-10 sm:px-7 sm:py-14">
      <div className="flex items-center gap-3 text-[0.76rem] text-dim">
        <Link
          href="/competitions"
          className="flex items-center gap-1.5 transition-colors hover:text-text"
        >
          <ArrowLeft className="size-3.5" />
          {t("back")}
        </Link>
        <span>·</span>
        <span>{data.season}</span>
        <span>·</span>
        <span>{t("teamCount", { n: data.teams })}</span>
      </div>

      <h1 className="font-display mt-5 text-[2.2rem] leading-[1.08] sm:text-[2.9rem]">
        {data.name}
      </h1>

      {data.preDraw ? (
        // Said plainly rather than quietly: before the draw the field is an Elo
        // guess, and a page that let that pass for the real thing would be
        // inventing participants.
        <p className="mt-5 rounded-[12px] border border-warning/30 bg-warning/10 px-4 py-3 text-[0.85rem] leading-relaxed text-warning">
          {t("preDraw")}
        </p>
      ) : (
        <p className="mt-4 max-w-2xl text-[1.02rem] leading-relaxed text-muted">
          {t.rich("headline", {
            team: favourite?.team ?? "—",
            p: Math.round((favourite?.champion ?? 0) * 100),
            played: data.played,
            total: data.leaguePhaseTotal,
            b: (chunks) => <span className="font-semibold text-text">{chunks}</span>,
          })}
        </p>
      )}

      <div className="mt-12 flex flex-col gap-16">
        <Reveal as="section">
          <h2 className="font-display text-[1.5rem] leading-tight sm:text-[1.9rem]">
            {t("raceTitle")}
          </h2>
          <div className="mt-6 flex flex-col gap-3">
            {race.map((r) => (
              <div key={r.team} className="flex items-center gap-3 sm:gap-4">
                <span className="w-32 shrink-0 truncate text-[0.88rem] font-medium sm:w-48">
                  {r.team}
                </span>
                <span className="h-2.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                  <span
                    className="block h-full rounded-full bg-brand"
                    style={{ width: `${((r.champion ?? 0) / raceLead) * 100}%` }}
                  />
                </span>
                <span className="w-12 text-right text-[0.85rem] font-semibold tabular-nums">
                  {pct(r.champion, 1)}
                </span>
              </div>
            ))}
          </div>
          <p className="mt-4 text-[0.78rem] leading-relaxed text-dim">
            {t("raceNote", { sims: data.simulations.toLocaleString(locale) })}
          </p>
        </Reveal>

        <Reveal as="section">
          <h2 className="font-display text-[1.5rem] leading-tight sm:text-[1.9rem]">
            {t("roundsTitle")}
          </h2>
          <div className="mt-5 overflow-x-auto rounded-[var(--radius-card)] border border-border">
            <table className="w-full min-w-[820px] text-[0.83rem]">
              <thead>
                <tr className="border-b border-border bg-surface-2 text-[0.62rem] uppercase tracking-[0.07em] text-dim">
                  <th className="px-3 py-2.5 text-left font-semibold">{t("team")}</th>
                  <th className="px-3 py-2.5 text-right font-semibold">Elo</th>
                  {ROUND_COLUMNS.map((c) => (
                    <th key={c} className="px-3 py-2.5 text-right font-semibold">
                      {t(`rounds.${c}`)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ranked.map((r) => (
                  <tr
                    key={r.team}
                    className="border-b border-border bg-surface last:border-0"
                  >
                    <td className="px-3 py-2.5 font-medium">{r.team}</td>
                    <td className="px-3 py-2.5 text-right tabular-nums text-dim">
                      {Math.round(r.elo)}
                    </td>
                    {ROUND_COLUMNS.map((c) => (
                      <td
                        key={c}
                        className={cn(
                          "px-3 py-2.5 text-right tabular-nums",
                          c === "champion" && (r.champion ?? 0) > 0.03
                            ? "font-semibold text-brand"
                            : "text-muted",
                        )}
                      >
                        {pct(r[c], c === "champion" ? 1 : 0)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Reveal>

        {data.phase.fixtures.length ? (
          <Reveal as="section">
            <h2 className="font-display text-[1.5rem] leading-tight sm:text-[1.9rem]">
              {t("phaseTitle", { round: data.phase.round ?? 1 })}
            </h2>
            <div className="mt-5 flex flex-col gap-1.5">
              {data.phase.fixtures.map((f) => (
                <div
                  key={f.slug}
                  className="flex items-center gap-3 rounded-[12px] border border-border bg-surface px-4 py-3.5 sm:gap-5 sm:px-5"
                >
                  <span className="hidden w-[4.5rem] shrink-0 text-[0.66rem] font-semibold uppercase tracking-[0.06em] text-dim sm:block">
                    {f.date
                      ? new Intl.DateTimeFormat(locale, {
                          day: "numeric",
                          month: "short",
                        }).format(new Date(`${f.date}T12:00:00`))
                      : ""}
                  </span>
                  <span className="flex-1 text-right text-[0.9rem] font-medium">
                    {f.home}
                  </span>
                  <span className="w-[6rem] shrink-0 text-center sm:w-[10rem]">
                    {f.played && f.result ? (
                      <span className="inline-block rounded-[7px] border border-border bg-bg px-2.5 py-1 text-[0.85rem] font-semibold">
                        {f.result}
                      </span>
                    ) : f.probabilities ? (
                      <span className="flex flex-col items-center">
                        <span className="flex items-center gap-1 text-[0.76rem] sm:gap-1.5 sm:text-[0.82rem]">
                          <b>{Math.round(f.probabilities.home * 100)}%</b>
                          <span className="text-dim">·</span>
                          <span className="text-muted">
                            {Math.round(f.probabilities.draw * 100)}%
                          </span>
                          <span className="text-dim">·</span>
                          <b>{Math.round(f.probabilities.away * 100)}%</b>
                        </span>
                        {f.over25 != null ? (
                          <span className="mt-0.5 text-[0.6rem] font-semibold uppercase tracking-[0.06em] text-dim">
                            {t("over25", { p: Math.round(f.over25 * 100) })}
                          </span>
                        ) : null}
                      </span>
                    ) : (
                      <span className="text-[0.7rem] uppercase tracking-[0.08em] text-dim">
                        {t("noElo")}
                      </span>
                    )}
                  </span>
                  <span className="flex-1 text-left text-[0.9rem] font-medium">
                    {f.away}
                  </span>
                </div>
              ))}
            </div>
          </Reveal>
        ) : null}

        <Reveal>
          <p className="max-w-2xl text-[0.82rem] leading-relaxed text-dim">
            {t("methodNote")}
          </p>
        </Reveal>
      </div>
    </div>
  );
}
