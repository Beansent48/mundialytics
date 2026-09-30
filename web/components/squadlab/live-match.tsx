"use client";

import { ArrowDownUp, Cross, FastForward, Square, Star } from "lucide-react";
import { useTranslations } from "next-intl";
import { useEffect, useMemo, useState } from "react";

import type { SquadFixture, SquadMatchEvent } from "@/lib/api";
import { cn } from "@/lib/utils";

const FULL_TIME = 90;
const EXTRA_TIME = 120;
/** A tick of the clock. 90 of them is roughly five seconds of match. */
const TICK_MS = 55;
/** How long the clock holds *on* a goal, so the celebration has room to play. */
const GOAL_PAUSE_MS = 1200;
/** A shorter hold for the other big moments: a red card, a missed penalty. */
const MOMENT_PAUSE_MS = 650;

const isGoal = (e: SquadMatchEvent) => e.type === "goal";
const isMoment = (e: SquadMatchEvent) => e.type === "red" || e.type === "penMiss";

export function LiveMatch({
  fixture,
  label,
  teamName,
  squadLabel,
  running,
  onFinish,
}: {
  fixture: SquadFixture;
  /** The stage caption for this match, e.g. "Fase liga · J1" or "Octavos · Ida". */
  label: string;
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
  // A knockout that went the distance runs its clock to 120.
  const end = fixture.extraTime ? EXTRA_TIME : FULL_TIME;
  const minute = running ? Math.min(tick, end) : end;
  const done = minute >= end;

  // The clock. A timeout rather than an interval because one minute is not like
  // the others: the hold belongs to the minute a goal went in, so the bar runs
  // at a steady rate and then stops dead on the goal while the flash plays.
  useEffect(() => {
    if (!running || tick >= end) return;
    const now = events.filter((e) => e.minute === tick);
    const wait = now.some(isGoal) ? GOAL_PAUSE_MS : now.some(isMoment) ? MOMENT_PAUSE_MS : TICK_MS;
    const id = setTimeout(() => setTick(tick + 1), wait);
    return () => clearTimeout(id);
  }, [running, tick, events, end]);

  useEffect(() => {
    if (running && tick >= end) onFinish();
  }, [running, tick, end, onFinish]);

  const shown = events.filter((e) => e.minute <= minute);
  const goals = shown.filter(isGoal);
  const homeGoals = done ? fixture.homeGoals : goals.filter((e) => e.side === "home").length;
  const awayGoals = done ? fixture.awayGoals : goals.filter((e) => e.side === "away").length;

  const justScored =
    running && !done ? events.find((e) => isGoal(e) && e.minute === minute) : undefined;

  const homeIsSquad = fixture.home === teamName;
  const absences = fixture.absences ?? [];

  return (
    <div className="overflow-hidden rounded-[18px] border border-border bg-surface">
      <div className="flex items-center justify-between gap-3 border-b border-border px-5 py-3">
        <span className="text-[0.62rem] font-semibold uppercase tracking-[0.13em] text-dim">
          {label}
        </span>
        {done ? (
          <span className="flex items-center gap-1.5 text-[0.62rem] font-semibold uppercase tracking-[0.13em] text-dim">
            <Square className="size-2.5" fill="currentColor" strokeWidth={0} />
            {fixture.shootout ? t("afterPens") : fixture.extraTime ? t("afterExtraTime") : t("fullTime")}
          </span>
        ) : (
          <span className="flex items-center gap-2 text-[0.62rem] font-semibold uppercase tracking-[0.13em] text-negative">
            <span className="animate-live-dot size-1.5 rounded-full bg-negative" />
            {minute > FULL_TIME ? t("extraTime") : t("live")}
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

        {done && fixture.shootout ? (
          <p className="mt-2 text-center text-[0.8rem] text-muted">
            {t("shootoutLine", { home: fixture.shootout.home, away: fixture.shootout.away })}
          </p>
        ) : null}

        {absences.length ? (
          <p className="mt-3 text-center text-[0.74rem] text-dim">
            {t("absences")}:{" "}
            {absences
              .map((a) =>
                a.reason === "lesion"
                  ? t("absenceInjury", { player: a.player, detail: a.detail })
                  : t("absenceBan", { player: a.player }),
              )
              .join(" · ")}
          </p>
        ) : null}

        <div className="mt-4 flex items-center gap-3">
          <span className="w-9 shrink-0 text-[0.72rem] font-semibold tabular-nums text-muted">
            {minute}&rsquo;
          </span>
          <span className="h-1 flex-1 overflow-hidden rounded-full bg-surface-3">
            <span
              className="block h-full rounded-full bg-brand transition-[width] duration-100 ease-linear"
              style={{ width: `${(minute / end) * 100}%` }}
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
                  key={`${e.minuteLabel ?? e.minute}-${e.type}-${e.player}-${i}`}
                  event={e}
                  onLeft={e.side === "home"}
                />
              ))}
          </div>
        ) : null}
      </div>

      {done && fixture.shootout ? <Shootout fixture={fixture} /> : null}
      {done ? <FullTime fixture={fixture} /> : null}
    </div>
  );
}

function EventIcon({ event }: { event: SquadMatchEvent }) {
  switch (event.type) {
    case "goal":
      return <span aria-hidden className="size-2.5 shrink-0 rounded-full bg-text" />;
    case "penMiss":
      return (
        <span
          aria-hidden
          className="size-2.5 shrink-0 rounded-full border-2 border-negative bg-transparent"
        />
      );
    case "red":
      return event.detail === "2y" ? (
        <span aria-hidden className="relative size-2.5 shrink-0">
          <span className="absolute inset-0 -translate-x-0.5 rounded-[2px] bg-card-yellow" />
          <span className="absolute inset-0 translate-x-0.5 rounded-[2px] bg-negative" />
        </span>
      ) : (
        <span aria-hidden className="size-2.5 shrink-0 rounded-[2px] bg-negative" />
      );
    case "sub":
      return <ArrowDownUp aria-hidden className="size-3 shrink-0 text-brand" />;
    case "injury":
      return <Cross aria-hidden className="size-3 shrink-0 text-negative" />;
    default:
      return <span aria-hidden className="size-2.5 shrink-0 rounded-[2px] bg-card-yellow" />;
  }
}

function EventRow({ event, onLeft }: { event: SquadMatchEvent; onLeft: boolean }) {
  const t = useTranslations("squadlab");

  let text: React.ReactNode = <span className="font-medium">{event.player}</span>;
  let note: string | null = null;
  if (event.type === "goal") {
    note =
      event.detail === "pen"
        ? t("penGoal")
        : event.detail === "og"
          ? t("ownGoal")
          : event.assist
            ? t("assistBy", { player: event.assist })
            : null;
  } else if (event.type === "penMiss") {
    note =
      event.detail === "saved" && event.assist
        ? t("penSavedBy", { player: event.assist })
        : t("penMissed");
  } else if (event.type === "red") {
    note = event.detail === "2y" ? t("secondYellow") : t("redCard");
  } else if (event.type === "injury") {
    note = t("injured");
  } else if (event.type === "sub") {
    text = (
      <>
        <span className="font-medium text-positive">{event.playerIn}</span>
        <span className="mx-1 text-dim">↔</span>
        <span className="text-muted">{event.player}</span>
      </>
    );
  }

  return (
    <div
      className={cn(
        "animate-event flex items-center gap-2 text-[0.82rem]",
        onLeft ? "" : "flex-row-reverse text-right",
        event.type === "sub" || event.type === "injury" ? "text-[0.76rem]" : "",
      )}
    >
      <span className="w-9 shrink-0 text-[0.68rem] tabular-nums text-dim">
        {event.minuteLabel ?? event.minute}&rsquo;
      </span>
      <EventIcon event={event} />
      <span className="truncate">
        {text}
        {note ? <span className="ml-1.5 text-[0.72rem] text-dim">{note}</span> : null}
      </span>
    </div>
  );
}

/* ── The shoot-out, kick by kick ───────────────────────────────────────────── */

function Shootout({ fixture }: { fixture: SquadFixture }) {
  const t = useTranslations("squadlab");
  const so = fixture.shootout;
  if (!so) return null;
  const side = (s: "home" | "away") => so.kicks.filter((k) => k.side === s);
  return (
    <div className="border-t border-border px-5 py-5">
      <p className="text-[0.6rem] font-semibold uppercase tracking-[0.13em] text-muted">
        {t("shootout")}
      </p>
      <div className="mt-3 grid grid-cols-2 gap-4">
        {(["home", "away"] as const).map((s) => (
          <div key={s} className={cn("flex flex-col gap-1", s === "away" ? "text-right" : "")}>
            {side(s).map((k, i) => (
              <span
                key={`${k.player}-${i}`}
                className={cn(
                  "flex items-center gap-2 text-[0.8rem]",
                  s === "away" ? "flex-row-reverse" : "",
                )}
              >
                <span
                  aria-hidden
                  className={cn(
                    "size-2.5 shrink-0 rounded-full",
                    k.scored ? "bg-positive" : "border-2 border-negative",
                  )}
                />
                <span className={k.scored ? "" : "text-dim line-through"}>{k.player}</span>
              </span>
            ))}
          </div>
        ))}
      </div>
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
                <span className="flex-1 truncate text-[0.83rem]">
                  {r.player}
                  {r.minutes != null && r.minutes < 90 ? (
                    <span className="ml-1.5 text-[0.66rem] text-dim">
                      {r.started === false ? "↑" : ""}
                      {r.minutes}&rsquo;
                    </span>
                  ) : null}
                </span>
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
                {r.red ? (
                  <span aria-hidden className="size-2.5 rounded-[2px] bg-negative" />
                ) : r.cards ? (
                  <span aria-hidden className="size-2.5 rounded-[2px] bg-card-yellow" />
                ) : null}
                {r.injured ? <Cross aria-hidden className="size-3 text-negative" /> : null}
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
