"use client";

import { FastForward, Square, Star } from "lucide-react";
import { useTranslations } from "next-intl";
import { useEffect, useMemo, useState } from "react";

import type { SquadFixture, SquadMatchEvent } from "@/lib/api";
import { cn } from "@/lib/utils";

const FULL_TIME = 90;
/** A tick of the clock. 90 of them is roughly five seconds of match. */
const TICK_MS = 55;
/** How long the clock holds *on* a goal, so the celebration has room to play. */
const GOAL_PAUSE_MS = 1200;

export function LiveMatch({
  fixture,
  matchday,
  teamName,
  squadLabel,
  running,
  onFinish,
}: {
  fixture: SquadFixture;
  matchday: number;
  /** The name the API uses for the squad, matched against the fixture. */
  teamName: string;
  /** The name the reader sees, in their language. */
  squadLabel: string;
  running: boolean;
  onFinish: () => void;
}) {
  const t = useTranslations("squadlab");
  const [tick, setTick] = useState(0);

  // Memoised because the clock's effect depends on it: a fresh `[]` every
  // render would tear down the pending timeout before it ever fired.
  const events = useMemo(() => fixture.events ?? [], [fixture.events]);
  const minute = running ? Math.min(tick, FULL_TIME) : FULL_TIME;
  const done = minute >= FULL_TIME;

  // The clock. A timeout rather than an interval because one minute is not like
  // the others: the hold belongs to the minute a goal went in, so the bar runs
  // at a steady rate and then stops dead on the goal while the flash plays.
  useEffect(() => {
    if (!running || tick >= FULL_TIME) return;
    const scored = events.some((e) => e.type === "goal" && e.minute === tick);
    const id = setTimeout(() => setTick(tick + 1), scored ? GOAL_PAUSE_MS : TICK_MS);
    return () => clearTimeout(id);
  }, [running, tick, events]);

  useEffect(() => {
    if (running && tick >= FULL_TIME) onFinish();
  }, [running, tick, onFinish]);

  const shown = events.filter((e) => e.minute <= minute);
  const goals = shown.filter((e) => e.type === "goal");
  const homeGoals = done
    ? fixture.homeGoals
    : goals.filter((e) => e.side === "home").length;
  const awayGoals = done
    ? fixture.awayGoals
    : goals.filter((e) => e.side === "away").length;

  const justScored =
    running && !done
      ? events.find((e) => e.type === "goal" && e.minute === minute)
      : undefined;

  const homeIsSquad = fixture.home === teamName;

  return (
    <div className="overflow-hidden rounded-[18px] border border-border bg-surface">
      <div className="flex items-center justify-between gap-3 border-b border-border px-5 py-3">
        <span className="text-[0.62rem] font-semibold uppercase tracking-[0.13em] text-dim">
          {t("matchdayLabel", { n: matchday })}
        </span>
        {done ? (
          <span className="flex items-center gap-1.5 text-[0.62rem] font-semibold uppercase tracking-[0.13em] text-dim">
            <Square className="size-2.5" fill="currentColor" strokeWidth={0} />
            {t("fullTime")}
          </span>
        ) : (
          <span className="flex items-center gap-2 text-[0.62rem] font-semibold uppercase tracking-[0.13em] text-negative">
            <span className="animate-live-dot size-1.5 rounded-full bg-negative" />
            {t("live")}
          </span>
        )}
      </div>

      <div className="relative px-5 py-7">
        {justScored ? (
          // Keyed on the minute so the flash replays on every goal rather than
          // animating once and staying put.
          <span
            key={justScored.minute}
            className="animate-goal pointer-events-none absolute inset-x-0 top-2 z-10 text-center font-display text-[2.6rem] leading-none text-brand"
          >
            {t("goalShout")}
          </span>
        ) : null}

        <div className="flex items-center gap-3">
          <span
            className={cn(
              "flex-1 truncate text-right text-[0.98rem]",
              homeIsSquad ? "font-semibold" : "text-muted",
            )}
          >
            {homeIsSquad ? squadLabel : fixture.home}
          </span>
          <span className="flex items-center gap-2 rounded-[11px] border border-border bg-bg px-4 py-2 text-[1.5rem] font-semibold tabular-nums">
            {homeGoals}
            <span className="text-dim">–</span>
            {awayGoals}
          </span>
          <span
            className={cn(
              "flex-1 truncate text-[0.98rem]",
              !homeIsSquad ? "font-semibold" : "text-muted",
            )}
          >
            {homeIsSquad ? fixture.away : squadLabel}
          </span>
        </div>

        <div className="mt-4 flex items-center gap-3">
          <span className="w-9 shrink-0 text-[0.72rem] font-semibold tabular-nums text-muted">
            {minute}&rsquo;
          </span>
          <span className="h-1 flex-1 overflow-hidden rounded-full bg-surface-3">
            <span
              className="block h-full rounded-full bg-brand transition-[width] duration-100 ease-linear"
              style={{ width: `${(minute / FULL_TIME) * 100}%` }}
            />
          </span>
          {!done ? (
            <button
              type="button"
              onClick={onFinish}
              className="flex shrink-0 items-center gap-1.5 rounded-lg border border-border px-2.5 py-1.5 text-[0.74rem] font-medium text-muted transition-colors hover:border-border-strong hover:text-text"
            >
              <FastForward className="size-3.5" />
              {t("skipMatch")}
            </button>
          ) : null}
        </div>

        {shown.length ? (
          <div className="mt-5 flex flex-col gap-1">
            {shown
              .slice()
              .reverse()
              .map((e, i) => (
                <EventRow
                  key={`${e.minute}-${e.player}-${i}`}
                  event={e}
                  onLeft={e.side === "home"}
                />
              ))}
          </div>
        ) : null}
      </div>

      {done ? <FullTime fixture={fixture} /> : null}
    </div>
  );
}

function EventRow({ event, onLeft }: { event: SquadMatchEvent; onLeft: boolean }) {
  const t = useTranslations("squadlab");
  return (
    <div
      className={cn(
        "animate-event flex items-center gap-2 text-[0.82rem]",
        onLeft ? "" : "flex-row-reverse text-right",
      )}
    >
      <span className="w-8 shrink-0 text-[0.68rem] tabular-nums text-dim">
        {event.minute}&rsquo;
      </span>
      <span
        aria-hidden
        className={cn(
          "size-2.5 shrink-0 rounded-[2px]",
          event.type === "goal" ? "rounded-full bg-text" : "bg-warning",
        )}
      />
      <span className="truncate">
        <span className="font-medium">{event.player}</span>
        {event.assist ? (
          <span className="ml-1.5 text-[0.72rem] text-dim">
            {t("assistBy", { player: event.assist })}
          </span>
        ) : null}
      </span>
    </div>
  );
}

/* ── What the engine measured ───────────────────────────────────────────── */

function FullTime({ fixture }: { fixture: SquadFixture }) {
  const t = useTranslations("squadlab");
  const stats = fixture.stats ?? [];
  const ratings = fixture.ratings ?? [];
  const motm = ratings[0];

  return (
    <div className="border-t border-border">
      {stats.length ? (
        <div className="px-5 py-5">
          <p className="text-[0.6rem] font-semibold uppercase tracking-[0.13em] text-muted">
            {t("matchStats")}
          </p>
          <div className="mt-3 flex flex-col gap-2.5">
            {stats.map((s) => (
              <StatRow
                key={s.key}
                label={t(`stat.${s.key}`)}
                home={s.home}
                away={s.away}
                decimals={s.key === "xg" ? 2 : 0}
              />
            ))}
          </div>
        </div>
      ) : null}

      {ratings.length ? (
        <div className="border-t border-border px-5 py-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-[0.6rem] font-semibold uppercase tracking-[0.13em] text-muted">
              {t("playerRatings")}
            </p>
            {motm ? (
              <p className="flex items-center gap-1.5 text-[0.75rem] text-warning">
                <Star className="size-3.5" fill="currentColor" strokeWidth={0} />
                {t("motm", { player: motm.player })}
              </p>
            ) : null}
          </div>
          <div className="mt-3 grid gap-1.5 sm:grid-cols-2">
            {ratings.map((r) => (
              <div
                key={r.player}
                className={cn(
                  "flex items-center gap-2 rounded-[10px] border px-3 py-1.5",
                  r === motm
                    ? "border-warning/40 bg-warning/5"
                    : "border-border bg-bg-elevated",
                )}
              >
                <span className="flex-1 truncate text-[0.83rem]">{r.player}</span>
                {r.goals ? (
                  <span className="text-[0.68rem] text-dim">
                    {"⚽".repeat(Math.min(r.goals, 3))}
                  </span>
                ) : null}
                {r.assists ? (
                  <span className="text-[0.62rem] font-semibold text-dim">
                    {t("assistShort", { n: r.assists })}
                  </span>
                ) : null}
                {r.cards ? (
                  <span aria-hidden className="size-2.5 rounded-[2px] bg-warning" />
                ) : null}
                <span
                  className={cn(
                    "rounded-[6px] px-1.5 py-0.5 text-[0.78rem] font-semibold tabular-nums",
                    r.rating >= 7.5
                      ? "bg-positive/15 text-positive"
                      : r.rating >= 6
                        ? "bg-surface-3 text-text"
                        : "bg-negative/12 text-negative",
                  )}
                >
                  {r.rating.toFixed(1)}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function StatRow({
  label,
  home,
  away,
  decimals,
}: {
  label: string;
  home: number;
  away: number;
  decimals: number;
}) {
  // Share of the pair, so the two bars always meet in the middle. A goalless,
  // shotless match would divide by zero, hence the even split.
  const total = home + away;
  const homeShare = total > 0 ? (home / total) * 100 : 50;

  return (
    <div>
      <div className="flex items-center justify-between text-[0.78rem]">
        <span className="font-semibold tabular-nums">{home.toFixed(decimals)}</span>
        <span className="text-[0.62rem] uppercase tracking-[0.08em] text-dim">
          {label}
        </span>
        <span className="font-semibold tabular-nums">{away.toFixed(decimals)}</span>
      </div>
      <div className="mt-1 flex h-1 gap-0.5 overflow-hidden rounded-full">
        <span className="flex flex-1 justify-end bg-surface-3">
          <span className="block h-full bg-home" style={{ width: `${homeShare}%` }} />
        </span>
        <span className="flex flex-1 bg-surface-3">
          <span
            className="block h-full bg-away"
            style={{ width: `${100 - homeShare}%` }}
          />
        </span>
      </div>
    </div>
  );
}
