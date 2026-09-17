"use client";

import { ChevronLeft, ChevronRight, Loader2, Star } from "lucide-react";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { FixtureRow } from "@/components/matchday/fixture-row";
import { api, type Competition, type Fixture } from "@/lib/api";
import { useFavorites } from "@/lib/favorites";
import { cn } from "@/lib/utils";

type Round = {
  competitionName: string;
  season: string;
  matchday: number;
  rounds: number[];
  fixtures: Fixture[];
};

/**
 * Browse any round of any competition.
 *
 * A peer of the day view rather than a footnote below it: "show me this
 * weekend's La Liga" is as common an intent as "show me today", and burying it
 * under the fold made a whole way of reading the season hard to find.
 * Followed competitions lead the strip.
 */
export function CompetitionBrowser({
  competitions,
}: {
  competitions: Competition[];
}) {
  const t = useTranslations("matchday");
  const { has, toggle, ready } = useFavorites();

  const [slug, setSlug] = useState(competitions[0]?.slug ?? "");
  const [round, setRound] = useState<number | null>(null);
  // The last answer, keyed on the competition and round it answers. The
  // previous round stays on screen while the next one loads.
  const [result, setResult] = useState<{
    key: string;
    data: Round | null;
    failed: boolean;
  } | null>(null);
  const key = `${slug}|${round ?? ""}`;

  useEffect(() => {
    if (!slug) return;
    let cancelled = false;
    const requested = `${slug}|${round ?? ""}`;
    api
      .round(slug, undefined, round ?? undefined)
      .then((r) => {
        if (cancelled) return;
        // Filed under the round it turned out to be, so letting the API pick
        // the upcoming round does not flash a second loading state.
        setResult({ key: `${slug}|${r.matchday}`, data: r, failed: false });
        setRound(r.matchday);
      })
      .catch(() => {
        if (cancelled) return;
        setResult((prev) => ({ key: requested, data: prev?.data ?? null, failed: true }));
      });
    return () => {
      cancelled = true;
    };
  }, [slug, round]);

  const loading = result?.key !== key;
  const failed = !loading && (result?.failed ?? false);
  const data = result?.data ?? null;

  // Followed competitions first, order otherwise untouched.
  const ordered = ready
    ? [...competitions].sort(
        (a, b) =>
          Number(has("competitions", b.slug)) - Number(has("competitions", a.slug)),
      )
    : competitions;

  const rounds = data?.rounds ?? [];
  const idx = data ? rounds.indexOf(data.matchday) : -1;

  return (
    <div>
      <div className="flex flex-wrap items-center gap-1.5">
        {ordered.map((c) => (
          <button
            key={c.slug}
            type="button"
            onClick={() => {
              setSlug(c.slug);
              setRound(null); // let the API pick the upcoming round again
            }}
            className={cn(
              "flex items-center gap-1.5 rounded-[10px] border px-3 py-1.5 text-[0.83rem] font-medium",
              "transition-colors duration-200",
              slug === c.slug
                ? "border-brand bg-brand-ghost text-text"
                : "border-border bg-surface text-muted hover:border-border-strong hover:text-text",
            )}
          >
            {ready && has("competitions", c.slug) ? (
              <Star className="size-3 text-warning" fill="currentColor" strokeWidth={0} />
            ) : null}
            {c.name}
          </button>
        ))}
      </div>

      <div className="mt-5 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <button
            type="button"
            aria-label={t("previousRound")}
            disabled={idx <= 0}
            onClick={() => setRound(rounds[idx - 1])}
            className="flex size-8 items-center justify-center rounded-lg border border-border text-muted transition-colors hover:border-border-strong hover:text-text disabled:opacity-40"
          >
            <ChevronLeft className="size-4" />
          </button>
          <span className="min-w-[8.5rem] text-center text-[0.86rem] font-semibold">
            {data
              ? t("roundOf", { round: data.matchday, total: rounds.length })
              : "—"}
          </span>
          <button
            type="button"
            aria-label={t("nextRound")}
            disabled={idx < 0 || idx >= rounds.length - 1}
            onClick={() => setRound(rounds[idx + 1])}
            className="flex size-8 items-center justify-center rounded-lg border border-border text-muted transition-colors hover:border-border-strong hover:text-text disabled:opacity-40"
          >
            <ChevronRight className="size-4" />
          </button>
        </div>

        <button
          type="button"
          onClick={() => toggle("competitions", slug)}
          className="flex items-center gap-1.5 rounded-lg border border-border px-2.5 py-1.5 text-[0.76rem] font-medium text-muted transition-colors hover:border-border-strong hover:text-text"
        >
          <Star
            className={cn("size-3.5", ready && has("competitions", slug) && "text-warning")}
            fill={ready && has("competitions", slug) ? "currentColor" : "none"}
            strokeWidth={ready && has("competitions", slug) ? 0 : 2}
          />
          {ready && has("competitions", slug) ? t("following") : t("follow")}
        </button>
      </div>

      <div className="mt-4 flex flex-col gap-1.5">
        {loading ? (
          // Skeletons rather than a spinner: the list keeps its shape, so the
          // page does not jump when the real rows land.
          Array.from({ length: 6 }).map((_, i) => (
            <div
              key={i}
              className="h-[3.4rem] animate-pulse rounded-[12px] border border-border bg-surface"
              style={{ animationDelay: `${i * 60}ms` }}
            />
          ))
        ) : failed ? (
          <p className="rounded-[12px] border border-border bg-surface px-5 py-8 text-center text-[0.87rem] text-muted">
            {t("failed")}
          </p>
        ) : data && data.fixtures.length ? (
          data.fixtures.map((f) => <FixtureRow key={f.slug} fixture={f} />)
        ) : (
          <p className="rounded-[12px] border border-border bg-surface px-5 py-8 text-center text-[0.87rem] text-muted">
            {t("empty")}
          </p>
        )}
      </div>

      {loading && data ? (
        <p className="mt-3 flex items-center gap-2 text-[0.75rem] text-dim">
          <Loader2 className="size-3 animate-spin" />
          {t("loading")}
        </p>
      ) : null}
    </div>
  );
}
