"use client";

import { Trophy } from "lucide-react";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { api, type Competition, type ScorerRace } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * The Golden Boot race.
 *
 * Client-side because the first computation for a league takes about half a
 * minute — the API walks every remaining fixture through the player model — and
 * a server-rendered page would simply hang. Switching leagues shows skeletons
 * and the result is cached on the server from then on.
 */
export function AwardsBrowser({ competitions }: { competitions: Competition[] }) {
  const t = useTranslations("awards");
  const [slug, setSlug] = useState(competitions[0]?.slug ?? "");
  const [race, setRace] = useState<ScorerRace | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!slug) return;
    let cancelled = false;
    setLoading(true);
    setFailed(false);
    setRace(null);
    api
      .awards(slug)
      .then((r) => !cancelled && setRace(r))
      .catch(() => !cancelled && setFailed(true))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [slug]);

  const lead = race?.players[0]?.p ?? 1;
  const favourite = race?.players[0];

  return (
    <div>
      <div className="flex flex-wrap gap-1.5">
        {competitions.map((c) => (
          <button
            key={c.slug}
            type="button"
            onClick={() => setSlug(c.slug)}
            className={cn(
              "rounded-[10px] border px-3 py-1.5 text-[0.83rem] font-medium transition-colors duration-200",
              slug === c.slug
                ? "border-brand bg-brand-ghost text-text"
                : "border-border bg-surface text-muted hover:border-border-strong hover:text-text",
            )}
          >
            {c.name}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="mt-8">
          <p className="text-[0.85rem] text-muted">{t("computing")}</p>
          <div className="mt-5 flex flex-col gap-2">
            {Array.from({ length: 8 }).map((_, i) => (
              <div
                key={i}
                className="h-[3.1rem] animate-pulse rounded-[12px] border border-border bg-surface"
                style={{ animationDelay: `${i * 70}ms` }}
              />
            ))}
          </div>
        </div>
      ) : failed || !race ? (
        <p className="mt-8 rounded-[var(--radius-card)] border border-border bg-surface px-5 py-10 text-center text-[0.88rem] text-muted">
          {t("failed")}
        </p>
      ) : (
        <>
          <dl className="mt-8 grid grid-cols-1 gap-px overflow-hidden rounded-[var(--radius-card)] border border-border bg-border sm:grid-cols-3">
            <div className="bg-surface px-5 py-5">
              <dt className="flex items-center gap-1.5 text-[0.63rem] font-semibold uppercase tracking-[0.1em] text-dim">
                <Trophy className="size-3 text-warning" />
                {t("favourite")}
              </dt>
              <dd className="mt-2 text-[1.25rem] font-semibold leading-tight">
                {favourite?.player ?? "—"}
              </dd>
              <dd className="mt-1 text-[0.78rem] text-brand">
                {favourite ? `${Math.round(favourite.p * 100)}%` : ""}
              </dd>
            </div>
            <div className="bg-surface px-5 py-5">
              <dt className="text-[0.63rem] font-semibold uppercase tracking-[0.1em] text-dim">
                {t("leader")}
              </dt>
              <dd className="mt-2 text-[1.25rem] font-semibold leading-tight">
                {race.leader?.player ?? "—"}
              </dd>
              <dd className="mt-1 text-[0.78rem] text-muted">
                {race.leader ? t("goalsNow", { n: race.leader.goals }) : ""}
              </dd>
            </div>
            <div className="bg-surface px-5 py-5">
              <dt className="text-[0.63rem] font-semibold uppercase tracking-[0.1em] text-dim">
                {t("inTheRace")}
              </dt>
              <dd className="mt-2 text-[1.25rem] font-semibold leading-tight">
                {race.inTheRace}
              </dd>
              <dd className="mt-1 text-[0.78rem] text-muted">{t("inTheRaceNote")}</dd>
            </div>
          </dl>

          <h2 className="font-display mt-12 text-[1.5rem] leading-tight sm:text-[1.9rem]">
            {t("rankingTitle")}
          </h2>
          <div className="mt-5 flex flex-col gap-2">
            {race.players.slice(0, 15).map((p, i) => (
              <div
                key={`${p.player}-${i}`}
                className="flex items-center gap-3 rounded-[12px] border border-border bg-surface px-4 py-3 sm:gap-5 sm:px-5"
              >
                <span className="w-5 shrink-0 text-[0.75rem] tabular-nums text-dim">
                  {i + 1}
                </span>
                <span className="w-[9rem] shrink-0 sm:w-[13rem]">
                  <span className="block truncate text-[0.9rem] font-medium">
                    {p.player}
                  </span>
                  <span className="block truncate text-[0.72rem] text-dim">
                    {p.team} · {t("goalsNow", { n: p.goals })} ·{" "}
                    {t("expected", { n: p.expected.toFixed(1) })}
                  </span>
                </span>
                <span className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
                  <span
                    className="block h-full rounded-full bg-warning"
                    style={{ width: `${(p.p / lead) * 100}%` }}
                  />
                </span>
                <span className="w-12 shrink-0 text-right text-[0.85rem] font-semibold tabular-nums">
                  {p.p < 0.001 ? "<0.1%" : `${(p.p * 100).toFixed(1)}%`}
                </span>
              </div>
            ))}
          </div>

          <p className="mt-5 max-w-2xl text-[0.79rem] leading-relaxed text-dim">
            {t("methodNote")}
          </p>
        </>
      )}
    </div>
  );
}
