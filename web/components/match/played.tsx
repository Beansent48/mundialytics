import { Check, X } from "lucide-react";
import { useTranslations } from "next-intl";
import { Fragment } from "react";

import { Reveal } from "@/components/ui/reveal";
import type { Match, StatRange, TimelineEvent, TimelineEventType } from "@/lib/api";
import { cn } from "@/lib/utils";

const pct = (v: number) => `${Math.round(v * 100)}%`;

// "45'+2'" -> 45.02, "90'" -> 90 — a sortable scalar, used only to split the
// timeline at half-time (the API sends the display string, not the number).
const minuteValue = (m: string) => {
  const s = m.replace(/'/g, "").trim();
  if (!s) return -1;
  if (s.includes("+")) {
    const [b, e] = s.split("+");
    return (Number(b) || 0) + (Number(e) || 0) / 100;
  }
  return Number(s) || 0;
};
const isGoal = (t: TimelineEventType) =>
  t === "goal" || t === "penalty" || t === "own_goal";

// One side of the stat comparison: did the real value land in the range we called
// most likely, and with what probability. Green tick when it did, amber cross when
// the match went somewhere we thought less likely.
function RangeHit({ actual, range }: { actual: number; range: StatRange | null }) {
  if (!range) return <span className="text-[0.68rem] text-dim">—</span>;
  const inside = actual >= range.lo && actual <= range.hi;
  const Icon = inside ? Check : X;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-full px-1.5 py-0.5 text-[0.62rem] font-semibold tabular-nums",
        inside ? "bg-positive/15 text-positive" : "bg-warning/15 text-warning",
      )}
    >
      <Icon className="h-3 w-3 shrink-0" strokeWidth={2.75} aria-hidden />
      {range.lo}–{range.hi}
      <span className="font-medium opacity-70">{pct(range.p)}</span>
    </span>
  );
}
// Scoreline strings arrive from the API with a plain hyphen, but the real score
// is rendered with an en dash elsewhere — normalise both before comparing.
const normScore = (s: string) => s.replace(/[–—]/g, "-").replace(/\s/g, "");

// The mark that leads a timeline row: a drawn football for any goal, a crisp
// card for a booking. A real vector, not an emoji, so it renders identically on
// every device and sits at a size the layout controls.
function EventMark({ type }: { type: TimelineEventType }) {
  if (type === "yellow" || type === "red") {
    return (
      <span
        aria-hidden
        className={cn(
          "inline-block h-[15px] w-[11px] shrink-0 rounded-[2.5px] shadow-sm",
          type === "yellow" ? "bg-card-yellow" : "bg-negative",
        )}
      />
    );
  }
  const own = type === "own_goal";
  return (
    <svg
      aria-hidden
      viewBox="0 0 16 16"
      className={cn("h-[15px] w-[15px] shrink-0", own ? "text-dim" : "text-text")}
      fill="none"
      stroke="currentColor"
      strokeWidth="1"
      strokeLinejoin="round"
    >
      <circle cx="8" cy="8" r="6.6" />
      <path
        d="M8 5.6l2.28 1.66-.87 2.68H6.59l-.87-2.68z"
        fill="currentColor"
        stroke="none"
      />
      <path
        d="M8 5.6V1.4M10.28 7.26l3.9-1.27M9.41 9.94l2.4 3.3M6.59 9.94l-2.4 3.3M5.72 7.26l-3.9-1.27"
        strokeWidth="0.9"
      />
    </svg>
  );
}

// A stop on the central spine: kick-off, half-time (with the score at the break)
// and full-time. Gives the column the shape of a real match instead of a loose
// list of minutes.
function SpineMarker({ label }: { label: string }) {
  return (
    <li className="grid grid-cols-[1fr_auto_1fr] items-center">
      <span className="col-start-2 z-[1] inline-flex justify-center whitespace-nowrap rounded-full border border-border bg-surface px-2.5 py-0.5 text-[0.55rem] font-semibold uppercase tracking-[0.12em] text-dim">
        {label}
      </span>
    </li>
  );
}

// One event. The minute badge and the player's name share a single grid row and
// are both vertically centred in it, so they always line up; the assist drops to
// a second row in the same side column and never nudges the badge. A fixed centre
// track keeps the badge on the spine and the two rows' columns aligned.
function TimelineRow({ e, t }: { e: TimelineEvent; t: ReturnType<typeof useTranslations> }) {
  const home = e.side === "home";
  return (
    <li className="grid grid-cols-[1fr_3rem_1fr] items-center gap-x-3">
      <div
        className={cn(
          "row-start-1 flex min-w-0 items-center gap-1.5",
          home ? "col-start-1 flex-row-reverse" : "col-start-3",
        )}
      >
        <EventMark type={e.type} />
        <span
          className={cn(
            "truncate text-[0.9rem] font-medium leading-5",
            e.type === "own_goal" && "text-dim",
          )}
        >
          {e.player}
          {e.type === "own_goal" ? (
            <span className="ml-1 text-[0.7rem] font-normal text-dim">
              {t("ownGoalTag")}
            </span>
          ) : null}
          {e.type === "penalty" ? (
            <span className="ml-1 text-[0.7rem] font-normal text-muted">
              {t("penaltyTag")}
            </span>
          ) : null}
        </span>
      </div>
      <div className="col-start-2 row-start-1 flex justify-center">
        <span className="z-[1] inline-flex min-w-[2.4rem] justify-center whitespace-nowrap rounded-full border border-border bg-surface px-1.5 py-0.5 text-[0.64rem] font-semibold leading-4 tabular-nums text-muted">
          {e.minute}
        </span>
      </div>
      {isGoal(e.type) && e.assist ? (
        <p
          className={cn(
            "row-start-2 truncate text-[0.76rem] text-muted",
            home ? "col-start-1 pr-[1.625rem] text-right" : "col-start-3 pl-[1.625rem]",
          )}
        >
          {t("assistLabel", { player: e.assist })}
        </p>
      ) : null}
    </li>
  );
}

/**
 * A match that has been played, on one scrolling page.
 *
 * What happened, then what we said would happen — the timeline first, the model's
 * own scorecard right under it. It reads top to bottom instead of hiding half the
 * story behind a tab: thousands of these pages, each grading itself in public, is
 * the most credible thing this project can show.
 */
export function PlayedMatch({ match }: { match: Match }) {
  const t = useTranslations("match");

  const actual = match.actual;
  const events = match.events;
  const pred = match.prediction;
  const score = match.score;
  if (!score) return null;

  const outcome = score.home > score.away ? "1" : score.home === score.away ? "X" : "2";
  const total = score.home + score.away;
  const bothScored = score.home > 0 && score.away > 0;

  // 1X2 as a division of one whole, with the outcome that actually happened lit
  // up and the other two dimmed — the single most legible way to show a
  // probabilistic call against a settled result.
  const outcomes = [
    { id: "1", label: match.home, p: pred.probabilities.home, color: "var(--home)" },
    { id: "X", label: t("draw"), p: pred.probabilities.draw, color: "var(--draw)" },
    { id: "2", label: match.away, p: pred.probabilities.away, color: "var(--away)" },
  ];
  const pick = outcomes.reduce((a, b) => (b.p > a.p ? b : a));
  const resultHit = pick.id === outcome;

  // Where the real score fell in our distribution of likeliest scorelines.
  const realScore = `${score.home}-${score.away}`;
  const rank =
    pred.scorelines.findIndex((s) => normScore(s.score) === realScore) + 1;
  const rankProb = rank > 0 ? pred.scorelines[rank - 1].p : 0;
  const xgTotal = pred.expectedGoals.home + pred.expectedGoals.away;

  const calls = [
    {
      market: t("result"),
      said: pick.id,
      conf: pick.p,
      real: outcome,
      hit: resultHit,
    },
    {
      market: t("over25"),
      said: pred.goals.over25 >= 0.5 ? t("over") : t("under"),
      conf: Math.max(pred.goals.over25, 1 - pred.goals.over25),
      real: t("goalCount", { count: total }),
      hit: total > 2.5 === pred.goals.over25 >= 0.5,
    },
    {
      market: t("over15"),
      said: pred.goals.over15 >= 0.5 ? t("over") : t("under"),
      conf: Math.max(pred.goals.over15, 1 - pred.goals.over15),
      real: t("goalCount", { count: total }),
      hit: total > 1.5 === pred.goals.over15 >= 0.5,
    },
    {
      market: t("btts"),
      said: pred.goals.btts >= 0.5 ? t("yes") : t("no"),
      conf: Math.max(pred.goals.btts, 1 - pred.goals.btts),
      real: bothScored ? t("yes") : t("no"),
      hit: bothScored === pred.goals.btts >= 0.5,
    },
  ];

  const predByKey = new Map(pred.expectedStats.map((s) => [s.key, s]));
  const hits = calls.filter((c) => c.hit).length;

  // Timeline: split at half-time, and read the score at the break off the events.
  const timeline = events?.timeline ?? [];
  const htIndex = timeline.findIndex((e) => minuteValue(e.minute) >= 46);
  const htHome = timeline.filter(
    (e) => isGoal(e.type) && e.side === "home" && minuteValue(e.minute) < 46,
  ).length;
  const htAway = timeline.filter(
    (e) => isGoal(e.type) && e.side === "away" && minuteValue(e.minute) < 46,
  ).length;

  // Possession is a share of one whole — a split bar, not a grid cell. Everything
  // else keeps the compare grid with its most-likely range.
  const possession = actual?.stats.find((s) => s.key === "possession") ?? null;
  const gridStats = actual?.stats.filter((s) => s.key !== "possession") ?? [];

  const eyebrow = "text-[0.66rem] font-semibold uppercase tracking-[0.18em] text-brand";
  const heading = "font-display text-[1.5rem] leading-tight sm:text-[1.8rem]";

  return (
    <div className="flex flex-col gap-16">
      <header>
        <h1 className="font-display text-[1.9rem] leading-[1.12] sm:text-[2.7rem]">
          <span className={outcome === "2" ? "text-dim" : undefined}>{match.home}</span>
          <span className="mx-3 tabular-nums">
            {score.home} – {score.away}
          </span>
          <span className={outcome === "1" ? "text-dim" : undefined}>{match.away}</span>
        </h1>
      </header>

      {/* 1 · What happened. A minute-by-minute timeline when ESPN gave us the
             clock; otherwise the grouped per-side summary, which has no minutes. */}
      {events ? (
        <Reveal as="section">
          <p className={eyebrow}>{t("sectionMatch")}</p>
          <h2 className={cn(heading, "mb-6 mt-2")}>{t("summaryTitle")}</h2>
          {timeline.length ? (
            <ol className="relative mx-auto flex max-w-lg flex-col gap-4 py-1 before:absolute before:inset-y-2 before:left-1/2 before:w-px before:-translate-x-1/2 before:bg-border">
              <SpineMarker label={t("kickoff")} />
              {timeline.map((e, i) => (
                <Fragment key={`${e.minute}-${e.player}-${e.type}-${i}`}>
                  {i === htIndex && htIndex > 0 ? (
                    <SpineMarker label={`${t("halfTime")} · ${htHome}–${htAway}`} />
                  ) : null}
                  <TimelineRow e={e} t={t} />
                </Fragment>
              ))}
              <SpineMarker label={t("fullTime")} />
            </ol>
          ) : (
            <div className="grid grid-cols-2 gap-px overflow-hidden rounded-[var(--radius-card)] border border-border bg-border">
              {(
                [
                  { s: events.home, name: match.home, color: "text-home", right: false },
                  { s: events.away, name: match.away, color: "text-away", right: true },
                ] as const
              ).map(({ s, name, color, right }) => {
                const empty =
                  !s.goals.length &&
                  !s.assists.length &&
                  !s.yellows.length &&
                  !s.unattributed;
                return (
                  <div
                    key={name}
                    className={cn("bg-surface px-5 py-5", right && "text-right")}
                  >
                    <p className={cn("text-[0.9rem] font-semibold", color)}>{name}</p>
                    {empty ? (
                      <p className="mt-3 text-[0.8rem] text-dim">—</p>
                    ) : (
                      <ul className="mt-3 flex flex-col gap-2.5">
                        {s.goals.map((g) => (
                          <li
                            key={`g-${g.player}`}
                            className={cn(
                              "flex items-center gap-2 text-[0.88rem]",
                              right && "flex-row-reverse",
                            )}
                          >
                            <EventMark type="goal" />
                            <span className="font-medium">{g.player}</span>
                            {g.count > 1 ? (
                              <span className="rounded-full bg-surface-3 px-1.5 py-0.5 text-[0.62rem] font-semibold tabular-nums text-muted">
                                ×{g.count}
                              </span>
                            ) : null}
                          </li>
                        ))}
                        {s.unattributed ? (
                          <li
                            className={cn(
                              "flex items-center gap-2 text-[0.88rem] text-dim",
                              right && "flex-row-reverse",
                            )}
                          >
                            <EventMark type="goal" />
                            <span>{t("unattributed", { count: s.unattributed })}</span>
                          </li>
                        ) : null}
                        {s.assists.map((a) => (
                          <li
                            key={`a-${a.player}`}
                            className={cn(
                              "text-[0.8rem] text-muted",
                              right && "text-right",
                            )}
                          >
                            {t("assistLabel", { player: a.player })}
                            {a.count > 1 ? ` ×${a.count}` : ""}
                          </li>
                        ))}
                        {s.yellows.map((y) => (
                          <li
                            key={`y-${y}`}
                            className={cn(
                              "flex items-center gap-2 text-[0.88rem]",
                              right && "flex-row-reverse",
                            )}
                          >
                            <span
                              aria-hidden
                              className="inline-block h-3.5 w-2.5 shrink-0 rounded-[2px] bg-card-yellow"
                            />
                            <span>{y}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </Reveal>
      ) : null}

      {/* 2 · What we said. The outcome as a probability bar with reality lit up. */}
      <Reveal as="section">
        <p className={eyebrow}>{t("sectionCall")}</p>
        <div className="mb-6 mt-2 flex items-baseline gap-3">
          <h2 className={heading}>{t("callsTitle")}</h2>
          <span className="text-[0.78rem] text-dim">
            {t("callsScore", { hits, total: calls.length })}
          </span>
        </div>

        <div className="flex h-16 overflow-hidden rounded-[12px] sm:h-20">
          {outcomes.map((o) => {
            const happened = o.id === outcome;
            return (
              <div
                key={o.id}
                className={cn(
                  "relative flex flex-col items-center justify-center transition-[flex-grow] duration-700 ease-[var(--ease-out-quint)]",
                  // The outcome that happened is full colour with white on top; the
                  // others are a pale tint of their colour with legible muted text —
                  // dimming the whole segment with opacity left white text unreadable
                  // over the pale result in the light theme.
                  happened ? "text-white" : "text-muted",
                )}
                style={{
                  flexGrow: o.p,
                  flexBasis: 0,
                  background: happened
                    ? o.color
                    : `color-mix(in srgb, ${o.color} 16%, var(--surface))`,
                }}
              >
                <span className="text-[1.15rem] font-semibold leading-none sm:text-[1.5rem]">
                  {pct(o.p)}
                </span>
                {happened ? (
                  <span className="mt-1 text-[0.55rem] font-semibold uppercase tracking-[0.12em]">
                    {t("happened")}
                  </span>
                ) : null}
              </div>
            );
          })}
        </div>
        <div className="mt-2.5 flex justify-between text-[0.78rem] text-muted">
          {outcomes.map((o) => (
            <span
              key={o.id}
              className={cn(
                "max-w-[33%] truncate",
                o.id === outcome ? "font-semibold text-text" : undefined,
              )}
            >
              {o.label}
            </span>
          ))}
        </div>

        <p className="mt-4 flex items-center gap-2 text-[0.85rem] text-muted">
          <span
            className={cn(
              "flex size-5 shrink-0 items-center justify-center rounded-full",
              resultHit
                ? "bg-positive/15 text-positive"
                : "bg-negative/15 text-negative",
            )}
          >
            {resultHit ? <Check className="size-3" /> : <X className="size-3" />}
          </span>
          {t("favouriteWas", { pick: pick.label, p: pct(pick.p) })}
        </p>

        {/* The scoreline: what we thought was likeliest, versus reality. */}
        <div className="mt-8 overflow-hidden rounded-[var(--radius-card)] border border-border bg-surface">
          <div className="grid grid-cols-2 divide-x divide-border">
            <div className="px-5 py-5">
              <p className="text-[0.62rem] font-semibold uppercase tracking-[0.1em] text-dim">
                {t("predicted")}
              </p>
              <p className="font-display mt-2 text-[1.7rem] leading-none tabular-nums text-muted">
                {pred.headline.likelyScore}
              </p>
            </div>
            <div className="px-5 py-5">
              <p className="text-[0.62rem] font-semibold uppercase tracking-[0.1em] text-dim">
                {t("realLabel")}
              </p>
              <p className="font-display mt-2 text-[1.7rem] leading-none tabular-nums">
                {score.home}–{score.away}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 border-t border-border px-5 py-3 text-[0.78rem] text-dim">
            <span>
              {rank > 0
                ? t("scorelineRank", { rank, p: pct(rankProb) })
                : t("scorelineOutside")}
            </span>
            <span className="tabular-nums">
              {t("goalsExpectedReal", { expected: xgTotal.toFixed(1), real: total })}
            </span>
          </div>
        </div>

        {/* Every headline market, one card each. */}
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          {calls.map((c) => (
            <div
              key={c.market}
              className="flex items-center justify-between gap-4 rounded-[12px] border border-border bg-surface px-5 py-4"
            >
              <div className="min-w-0 flex-1">
                <p className="text-[0.62rem] font-semibold uppercase tracking-[0.1em] text-dim">
                  {c.market}
                </p>
                <div className="mt-1.5 flex items-baseline gap-2">
                  <span className="text-[1.05rem] font-semibold">{c.said}</span>
                  <span className="text-[0.78rem] tabular-nums text-dim">{pct(c.conf)}</span>
                </div>
                <div className="mt-2 h-1 overflow-hidden rounded-full bg-surface-3">
                  <div
                    className={cn(
                      "h-full rounded-full",
                      c.hit ? "bg-positive" : "bg-negative",
                    )}
                    style={{ width: `${Math.round(c.conf * 100)}%` }}
                  />
                </div>
                <p className="mt-2 text-[0.75rem] text-dim">
                  {t("realWas", { value: c.real })}
                </p>
              </div>
              <span
                className={cn(
                  "flex size-6 shrink-0 items-center justify-center rounded-full",
                  c.hit
                    ? "bg-positive/15 text-positive"
                    : "bg-negative/15 text-negative",
                )}
              >
                {c.hit ? <Check className="size-3.5" /> : <X className="size-3.5" />}
              </span>
            </div>
          ))}
        </div>
      </Reveal>

      {/* 3 · Real match stats beside what the model expected. Possession leads as
             a split bar; the rest carry a hit/miss against the range we called. */}
      {actual?.stats.length ? (
        <Reveal as="section">
          <p className={eyebrow}>{t("sectionStats")}</p>
          <h2 className={cn(heading, "mb-6 mt-2")}>{t("statsTitle")}</h2>

          {possession ? (
            <div className="mb-4">
              <div className="mb-1.5 flex items-baseline justify-between">
                <span className="text-[0.95rem] font-semibold tabular-nums text-home">
                  {possession.home}%
                </span>
                <span className="text-[0.58rem] font-semibold uppercase tracking-[0.1em] text-dim">
                  {t("markets.possession")}
                </span>
                <span className="text-[0.95rem] font-semibold tabular-nums text-away">
                  {possession.away}%
                </span>
              </div>
              <div className="flex h-2.5 overflow-hidden rounded-full bg-surface-3">
                <div style={{ width: `${possession.home}%`, background: "var(--home)" }} />
                <div style={{ width: `${possession.away}%`, background: "var(--away)" }} />
              </div>
            </div>
          ) : null}

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {gridStats.map((s) => {
              const p = predByKey.get(s.key);
              return (
                <div
                  key={s.key}
                  className="rounded-[12px] border border-border bg-surface px-4 py-3.5"
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-[1.05rem] font-semibold text-home tabular-nums">
                      {s.home}
                    </span>
                    <span className="text-[0.58rem] font-semibold uppercase tracking-[0.08em] text-dim">
                      {t(`markets.${s.key}`)}
                    </span>
                    <span className="text-[1.05rem] font-semibold text-away tabular-nums">
                      {s.away}
                    </span>
                  </div>
                  {p ? (
                    <div className="mt-3 border-t border-dashed border-border pt-2.5">
                      <div className="flex items-baseline justify-between gap-2 text-dim">
                        <span className="text-[0.82rem] font-medium tabular-nums">
                          {p.home}
                        </span>
                        <span className="text-[0.55rem] font-semibold uppercase tracking-[0.08em]">
                          {t("mostLikely")}
                        </span>
                        <span className="text-[0.82rem] font-medium tabular-nums">
                          {p.away}
                        </span>
                      </div>
                      <div className="mt-2 flex items-center justify-between gap-2">
                        <RangeHit actual={s.home} range={p.homeRange} />
                        <RangeHit actual={s.away} range={p.awayRange} />
                      </div>
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
          <p className="mt-4 text-[0.78rem] text-dim">{t("rangeNote")}</p>
        </Reveal>
      ) : null}
    </div>
  );
}
