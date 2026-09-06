"use client";

import { Check, X } from "lucide-react";
import { useTranslations } from "next-intl";
import { useState } from "react";

import type { Match } from "@/lib/api";
import { cn } from "@/lib/utils";

const pct = (v: number) => `${Math.round(v * 100)}%`;

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
  const pred = match.prediction;
  const score = match.score;
  if (!score) return null;

  const outcome = score.home > score.away ? "1" : score.home === score.away ? "X" : "2";
  const trio: Record<string, number> = {
    "1": pred.probabilities.home,
    X: pred.probabilities.draw,
    "2": pred.probabilities.away,
  };
  const pick = Object.keys(trio).reduce((a, b) => (trio[b] > trio[a] ? b : a));
  const total = score.home + score.away;

  const calls = [
    {
      market: t("result"),
      said: `${pick} (${pct(trio[pick])})`,
      real: outcome,
      hit: pick === outcome,
    },
    {
      market: t("over25"),
      said: `${pred.goals.over25 >= 0.5 ? t("over") : t("under")} (${pct(Math.max(pred.goals.over25, 1 - pred.goals.over25))})`,
      real: t("goalCount", { count: total }),
      hit: total > 2.5 === pred.goals.over25 >= 0.5,
    },
    {
      market: t("over15"),
      said: `${pred.goals.over15 >= 0.5 ? t("over") : t("under")} (${pct(Math.max(pred.goals.over15, 1 - pred.goals.over15))})`,
      real: t("goalCount", { count: total }),
      hit: total > 1.5 === pred.goals.over15 >= 0.5,
    },
    {
      market: t("btts"),
      said: `${pred.goals.btts >= 0.5 ? t("yes") : t("no")} (${pct(Math.max(pred.goals.btts, 1 - pred.goals.btts))})`,
      real: score.home > 0 && score.away > 0 ? t("yes") : t("no"),
      hit: (score.home > 0 && score.away > 0) === pred.goals.btts >= 0.5,
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
        <section>
          <div className="mb-5 flex items-baseline gap-3">
            <h2 className="font-display text-[1.5rem] leading-tight sm:text-[1.8rem]">
              {t("callsTitle")}
            </h2>
            <span className="text-[0.78rem] text-dim">
              {t("callsScore", { hits, total: calls.length })}
            </span>
          </div>
          <div className="grid gap-px overflow-hidden rounded-[var(--radius-card)] border border-border bg-border sm:grid-cols-4">
            {calls.map((c) => (
              <div key={c.market} className="bg-surface px-5 py-5">
                <p className="text-[0.63rem] font-semibold uppercase tracking-[0.1em] text-dim">
                  {c.market}
                </p>
                <p className="mt-2 flex items-center gap-2 text-[1.05rem] font-semibold">
                  {c.said}
                  <span
                    className={cn(
                      "flex size-5 items-center justify-center rounded-full",
                      c.hit
                        ? "bg-positive/15 text-positive"
                        : "bg-negative/15 text-negative",
                    )}
                  >
                    {c.hit ? <Check className="size-3" /> : <X className="size-3" />}
                  </span>
                </p>
                <p className="mt-1.5 text-[0.75rem] text-dim">
                  {t("realWas", { value: c.real })}
                </p>
              </div>
            ))}
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
                        <span className="text-[0.82rem] font-medium">{p.home}</span>
                        <span className="text-[0.55rem] font-semibold uppercase tracking-[0.08em]">
                          {t("predicted")}
                        </span>
                        <span className="text-[0.82rem] font-medium">{p.away}</span>
                      </div>
                      <p
                        className={cn(
                          "mt-1.5 text-center text-[0.6rem] font-semibold uppercase tracking-[0.06em]",
                          err <= 1
                            ? "text-positive"
                            : err <= 3
                              ? "text-warning"
                              : "text-negative",
                        )}
                      >
                        {t("error", { value: err.toFixed(1) })}
                      </p>
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
          {!actual ? null : (
            <p className="mt-4 text-[0.78rem] text-dim">{t("statsNote")}</p>
          )}
        </section>
      ) : null}
    </div>
  );
}
