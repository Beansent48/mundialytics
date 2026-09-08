"use client";

import { CalendarDays, ChevronLeft, ChevronRight, ListOrdered, Star } from "lucide-react";
import { useLocale, useTranslations } from "next-intl";
import { useEffect, useMemo, useRef, useState } from "react";

import { CompetitionBrowser } from "@/components/matchday/competition-browser";
import { FixtureRow } from "@/components/matchday/fixture-row";
import {
  api,
  type Competition,
  type DayFixtures,
  type Fixture,
  type UpcomingDay,
} from "@/lib/api";
import { useFavorites } from "@/lib/favorites";
import { cn } from "@/lib/utils";

const iso = (d: Date) => d.toISOString().slice(0, 10);

function addDays(isoDate: string, n: number) {
  const d = new Date(`${isoDate}T12:00:00`);
  d.setDate(d.getDate() + n);
  return iso(d);
}

const TAB_ICON = {
  day: CalendarDays,
  round: ListOrdered,
  favorites: Star,
} as const;

export function MatchdayClient({
  today,
  competitions,
}: {
  today: string;
  competitions: Competition[];
}) {
  const t = useTranslations("matchday");
  const [tab, setTab] = useState<"day" | "round" | "favorites">("day");
  const { favorites, ready } = useFavorites();

  return (
    <div>
      <div className="inline-flex max-w-full gap-1 overflow-x-auto rounded-[11px] border border-border bg-surface p-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        {(["day", "round", "favorites"] as const).map((v) => {
          const Icon = TAB_ICON[v];
          return (
            <button
              key={v}
              type="button"
              onClick={() => setTab(v)}
              className={cn(
                "flex shrink-0 items-center gap-1.5 rounded-[8px] px-3.5 py-1.5 text-[0.83rem] font-medium transition-colors duration-200",
                tab === v ? "bg-brand-ghost text-text" : "text-muted hover:text-text",
              )}
            >
              <Icon className="size-3.5" />
              {t(`tab.${v}`)}
              {v === "favorites" && ready && favorites.teams.length ? (
                <span className="rounded-full bg-warning/15 px-1.5 text-[0.65rem] font-semibold text-warning">
                  {favorites.teams.length}
                </span>
              ) : null}
            </button>
          );
        })}
      </div>

      <div className="mt-8">
        {tab === "day" ? (
          <DayView today={today} />
        ) : tab === "round" ? (
          <div>
            <p className="mb-5 max-w-xl text-[0.87rem] text-muted">{t("browseLead")}</p>
            <CompetitionBrowser competitions={competitions} />
          </div>
        ) : (
          <FavoritesView />
        )}
      </div>
    </div>
  );
}

/* ── The date strip ─────────────────────────────────────────────────────── */

function DateStrip({
  today,
  selected,
  counts,
  onSelect,
}: {
  today: string;
  selected: string;
  counts: Record<string, number>;
  onSelect: (d: string) => void;
}) {
  const locale = useLocale();
  const scroller = useRef<HTMLDivElement>(null);

  // Two weeks back and three forward: far enough to reach last weekend and the
  // next two, which is the whole range anybody actually scrubs through.
  const days = useMemo(() => {
    const out: string[] = [];
    for (let i = -14; i <= 21; i++) out.push(addDays(today, i));
    return out;
  }, [today]);

  useEffect(() => {
    const el = scroller.current?.querySelector<HTMLElement>('[data-selected="true"]');
    el?.scrollIntoView({ inline: "center", block: "nearest", behavior: "smooth" });
  }, [selected]);

  const shift = (n: number) => onSelect(addDays(selected, n));

  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        onClick={() => shift(-1)}
        aria-label="−1"
        className="flex size-9 shrink-0 items-center justify-center rounded-[10px] border border-border text-muted transition-colors hover:border-border-strong hover:text-text"
      >
        <ChevronLeft className="size-4" />
      </button>

      <div
        ref={scroller}
        className="flex flex-1 gap-1.5 overflow-x-auto scroll-smooth pb-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
      >
        {days.map((d) => {
          const date = new Date(`${d}T12:00:00`);
          const isToday = d === today;
          const isSelected = d === selected;
          const n = counts[d] ?? 0;
          return (
            <button
              key={d}
              type="button"
              data-selected={isSelected}
              onClick={() => onSelect(d)}
              className={cn(
                "flex w-[3.6rem] shrink-0 flex-col items-center rounded-[11px] border px-1 py-2 transition-colors duration-200",
                isSelected
                  ? "border-brand bg-brand-ghost"
                  : "border-border bg-surface hover:border-border-strong",
              )}
            >
              <span
                className={cn(
                  "text-[0.6rem] font-semibold uppercase tracking-[0.06em]",
                  isToday ? "text-brand" : "text-dim",
                )}
              >
                {new Intl.DateTimeFormat(locale, { weekday: "short" }).format(date)}
              </span>
              <span
                className={cn(
                  "mt-0.5 text-[1.05rem] font-semibold leading-none tabular-nums",
                  isSelected ? "text-text" : "text-muted",
                )}
              >
                {date.getDate()}
              </span>
              {/* a dot, not a count: the strip answers "is there football", and
                  the number belongs next to the fixtures themselves */}
              <span
                className={cn(
                  "mt-1.5 size-1 rounded-full",
                  n ? (isSelected ? "bg-brand" : "bg-border-strong") : "bg-transparent",
                )}
              />
            </button>
          );
        })}
      </div>

      <button
        type="button"
        onClick={() => shift(1)}
        aria-label="+1"
        className="flex size-9 shrink-0 items-center justify-center rounded-[10px] border border-border text-muted transition-colors hover:border-border-strong hover:text-text"
      >
        <ChevronRight className="size-4" />
      </button>
    </div>
  );
}

/* ── One day, grouped by competition ────────────────────────────────────── */

function DayView({ today }: { today: string }) {
  const t = useTranslations("matchday");
  const locale = useLocale();
  const [selected, setSelected] = useState(today);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [day, setDay] = useState<DayFixtures | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    api
      .fixturesCalendar()
      .then((c) => setCounts(c.counts))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setFailed(false);
    api
      .fixturesDay(selected)
      .then((d) => !cancelled && setDay(d))
      .catch(() => !cancelled && setFailed(true))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const label =
    selected === today
      ? t("today")
      : new Intl.DateTimeFormat(locale, {
          weekday: "long",
          day: "numeric",
          month: "long",
        }).format(new Date(`${selected}T12:00:00`));

  return (
    <div>
      <DateStrip
        today={today}
        selected={selected}
        counts={counts}
        onSelect={setSelected}
      />

      <div className="mt-8 flex items-baseline gap-3">
        <h2 className="font-display text-[1.4rem] leading-tight capitalize sm:text-[1.7rem]">
          {label}
        </h2>
        {day && !loading ? (
          <span className="text-[0.78rem] text-dim">
            {t("matchCount", { count: day.count })}
          </span>
        ) : null}
      </div>

      {loading ? (
        <div className="mt-6 flex flex-col gap-1.5">
          {Array.from({ length: 6 }).map((_, i) => (
            <div
              key={i}
              className="h-[3.4rem] animate-pulse rounded-[12px] border border-border bg-surface"
              style={{ animationDelay: `${i * 60}ms` }}
            />
          ))}
        </div>
      ) : failed ? (
        <p className="mt-6 rounded-[12px] border border-border bg-surface px-5 py-10 text-center text-[0.88rem] text-muted">
          {t("failed")}
        </p>
      ) : !day?.competitions.length ? (
        <p className="mt-6 rounded-[12px] border border-border bg-surface px-5 py-12 text-center text-[0.88rem] text-muted">
          {t("noneThisDay")}
        </p>
      ) : (
        <div className="mt-6 flex flex-col gap-8">
          {day.competitions.map((c) => (
            <section key={c.slug}>
              <h3 className="flex items-center gap-3 text-[0.68rem] font-semibold uppercase tracking-[0.14em] text-muted">
                {c.name}
                <span className="h-px flex-1 bg-border" />
                <span className="text-dim">{c.fixtures.length}</span>
              </h3>
              <div className="mt-3 flex flex-col gap-1.5">
                {c.fixtures.map((f) => (
                  <FixtureRow key={f.slug} fixture={f} hideCompetition />
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── The teams you follow ───────────────────────────────────────────────── */

function FavoritesView() {
  const t = useTranslations("matchday");
  const locale = useLocale();
  const { favorites, ready } = useFavorites();
  const [days, setDays] = useState<UpcomingDay[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .upcoming(14)
      .then((r) => setDays(r.days))
      .catch(() => undefined)
      .finally(() => setLoading(false));
  }, []);

  const mine: { date: string; fixtures: Fixture[] }[] = useMemo(() => {
    if (!ready || !favorites.teams.length) return [];
    return days
      .map((d) => ({
        date: d.date,
        fixtures: d.fixtures.filter(
          (f) =>
            favorites.teams.includes(f.homeSlug) ||
            favorites.teams.includes(f.awaySlug),
        ),
      }))
      .filter((d) => d.fixtures.length > 0);
  }, [days, favorites.teams, ready]);

  if (!ready || loading) {
    return (
      <div className="flex flex-col gap-1.5">
        {Array.from({ length: 4 }).map((_, i) => (
          <div
            key={i}
            className="h-[3.4rem] animate-pulse rounded-[12px] border border-border bg-surface"
          />
        ))}
      </div>
    );
  }

  if (!favorites.teams.length) {
    return (
      <div className="rounded-[14px] border border-dashed border-border-strong bg-surface/50 px-6 py-14 text-center">
        <Star className="mx-auto size-6 text-dim" />
        <p className="mt-4 text-[1rem] font-semibold">{t("noFavoritesTitle")}</p>
        <p className="mx-auto mt-2 max-w-sm text-[0.87rem] leading-relaxed text-muted">
          {t("noFavoritesBody")}
        </p>
      </div>
    );
  }

  if (!mine.length) {
    return (
      <p className="rounded-[12px] border border-border bg-surface px-5 py-12 text-center text-[0.88rem] text-muted">
        {t("noFavoriteFixtures")}
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-8">
      {mine.map((d) => (
        <section key={d.date}>
          <h3 className="flex items-center gap-3 text-[0.68rem] font-semibold uppercase tracking-[0.14em] text-muted">
            {new Intl.DateTimeFormat(locale, {
              weekday: "long",
              day: "numeric",
              month: "short",
            }).format(new Date(`${d.date}T12:00:00`))}
            <span className="h-px flex-1 bg-border" />
          </h3>
          <div className="mt-3 flex flex-col gap-1.5">
            {d.fixtures.map((f) => (
              <FixtureRow key={`${f.competition}-${f.slug}`} fixture={f} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
