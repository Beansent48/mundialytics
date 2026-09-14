"use client";

import { Check, X } from "lucide-react";
import { useTranslations } from "next-intl";
import { useState } from "react";

import type { Match } from "@/lib/api";
import { cn } from "@/lib/utils";

const pct = (v: number) => `${Math.round(v * 100)}%`;
// Scoreline strings arrive from the API with a plain hyphen, but the real score
// is rendered with an en dash elsewhere — normalise both before comparing.
const normScore = (s: string) => s.replace(/[–—]/g, "-").replace(/\s/g, "");

/**
 * A match that has been played.
 *
 * Two views of one screen, not two screens: what happened, and what we said it
 * would be. The comparison is the product's own scorecard — thousands of these
 * pages, each grading itself in public, is the most credible thing this project
 * can show, so it is one click away rather than buried.
 */
export function PlayedMatch({ match }: { match: Match }) {
  const t = useTranslations("match");
  const [compare, setCompare] = useState(false);

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

  return (
    <div className="flex flex-col gap-12">
      <section>
        <h1 className="font-display text-[1.9rem] leading-[1.1] sm:text-[2.7rem]">
          {match.home}
          <span className="mx-3 tabular-nums">
            {score.home} – {score.away}
          </span>
          {match.away}
        </h1>

        <div
          role="tablist"
          aria-label={t("viewLabel")}
          className="mt-6 inline-flex gap-1 rounded-[11px] border border-border bg-surface p-1"
        >
          {[
            { id: false, label: t("viewReal") },
            { id: true, label: t("viewCompare") },
          ].map((v) => (
            <button
              key={String(v.id)}
              role="tab"
              aria-selected={compare === v.id}
              onClick={() => setCompare(v.id)}
              className={cn(
                "rounded-[8px] px-3.5 py-1.5 text-[0.82rem] font-medium transition-colors duration-200",
                compare === v.id
                  ? "bg-brand-ghost text-text"
                  : "text-muted hover:text-text",
              )}
            >
              {v.label}
            </button>
          ))}
        </div>
      </section>

      {compare ? (
        <>
          {/* 1 · The outcome, as a probability bar with reality lit up. */}
          <section>
            <div className="mb-5 flex items-baseline gap-3">
              <h2 className="font-display text-[1.5rem] leading-tight sm:text-[1.8rem]">
                {t("callsTitle")}
              </h2>
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
                      "relative flex flex-col items-center justify-center text-white transition-[flex-grow] duration-700 ease-[var(--ease-out-quint)]",
                      !happened && "opacity-35",
                    )}
                    style={{ flexGrow: o.p, flexBasis: 0, background: o.color }}
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
          </section>

          {/* 2 · The scoreline: what we thought was likeliest, versus reality. */}
          <section>
            <h3 className="mb-4 text-[0.68rem] font-semibold uppercase tracking-[0.14em] text-muted">
              {t("scoreboardTitle")}
            </h3>
            <div className="overflow-hidden rounded-[var(--radius-card)] border border-border bg-surface">
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
                  {t("goalsExpectedReal", {
                    expected: xgTotal.toFixed(1),
                    real: total,
                  })}
                </span>
              </div>
            </div>
          </section>

          {/* 3 · Every headline market, one card each. */}
          <section>
            <div className="grid gap-3 sm:grid-cols-2">
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
                      <span className="text-[0.78rem] tabular-nums text-dim">
                        {pct(c.conf)}
                      </span>
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
          </section>
        </>
      ) : null}

      {/* What happened: goals, assists and cards grouped per side. No minutes —
          the settled source has none — so it reads as a summary, not a clock. */}
      {!compare && events ? (
        <section>
          <h2 className="font-display mb-5 text-[1.5rem] leading-tight sm:text-[1.8rem]">
            {t("summaryTitle")}
          </h2>
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
                          <span aria-hidden className="text-[0.8rem]">
                            ⚽
                          </span>
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
                          <span aria-hidden className="text-[0.8rem]">
                            ⚽
                          </span>
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
                            className="inline-block h-3.5 w-2.5 shrink-0 rounded-[2px] bg-warning"
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
        </section>
      ) : null}

      {actual?.stats.length ? (
        <section>
          <h2 className="font-display mb-5 text-[1.5rem] leading-tight sm:text-[1.8rem]">
            {t("statsTitle")}
          </h2>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
            {actual.stats.map((s) => {
              const p = predByKey.get(s.key);
              const err = p ? Math.abs(p.home - s.home) + Math.abs(p.away - s.away) : 0;
              return (
                <div
                  key={s.key}
                  className="rounded-[12px] border border-border bg-surface px-4 py-3.5"
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-[1.05rem] font-semibold text-home">
                      {s.home}
                    </span>
                    <span className="text-[0.58rem] font-semibold uppercase tracking-[0.08em] text-dim">
                      {t(`markets.${s.key}`)}
                    </span>
                    <span className="text-[1.05rem] font-semibold text-away">
                      {s.away}
                    </span>
                  </div>
                  {compare && p ? (
                    <div className="mt-3 border-t border-dashed border-border pt-2.5">
                      <div className="flex items-baseline justify-between gap-2 text-dim">
                        <span className="text-[0.82rem] font-medium tabular-nums">
                          {p.home}
                        </span>
                        <span className="text-[0.55rem] font-semibold uppercase tracking-[0.08em]">
                          {t("predicted")}
                        </span>
                        <span className="text-[0.82rem] font-medium tabular-nums">
                          {p.away}
                        </span>
                      </div>
                      <p className="mt-2 text-center">
                        <span
                          className={cn(
                            "inline-block rounded-full px-2 py-0.5 text-[0.6rem] font-semibold tabular-nums",
                            err <= 1
                              ? "bg-positive/15 text-positive"
                              : err <= 3
                                ? "bg-warning/15 text-warning"
                                : "bg-negative/15 text-negative",
                          )}
                        >
                          {t("deltaLabel", { value: err.toFixed(1) })}
                        </span>
                      </p>
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
          <p className="mt-4 text-[0.78rem] text-dim">{t("statsNote")}</p>
        </section>
      ) : null}

      {!compare && !events && !actual?.stats.length ? (
        <p className="rounded-[var(--radius-card)] border border-border bg-surface px-5 py-8 text-center text-[0.88rem] text-muted">
          {t("noData")}
        </p>
      ) : null}
    </div>
  );
}
