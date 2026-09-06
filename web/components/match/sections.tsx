import { getTranslations } from "next-intl/server";

import { Reveal } from "@/components/ui/reveal";
import type { Match } from "@/lib/api";
import { cn } from "@/lib/utils";

const pct = (v: number, d = 0) => `${(v * 100).toFixed(d)}%`;

function Heading({ children, note }: { children: string; note?: string }) {
  return (
    <div className="mb-5 flex items-baseline gap-3">
      <h2 className="font-display text-[1.5rem] leading-tight sm:text-[1.8rem]">
        {children}
      </h2>
      {note ? <span className="text-[0.75rem] text-dim">{note}</span> : null}
    </div>
  );
}

/**
 * Who scores.
 *
 * Placed directly under the verdict because it is the line people repeat — "the
 * model likes Lewandowski" travels in a way that a corners distribution never
 * will. Bars are scaled to the leader, not to 100%: nobody clears ~35%, so
 * absolute widths would render four identical stubs and destroy the ranking,
 * which is the only thing this model is actually good at.
 */
export async function Scorers({ match }: { match: Match }) {
  const t = await getTranslations("match");
  if (!match.scorers) return null;

  const sides = [
    { key: "home" as const, team: match.home, list: match.scorers.home ?? [] },
    { key: "away" as const, team: match.away, list: match.scorers.away ?? [] },
  ].filter((s) => s.list.length > 0);
  if (!sides.length) return null;

  const lead = Math.max(...sides.flatMap((s) => s.list.map((p) => p.p)), 0.0001);

  return (
    <Reveal as="section">
      <Heading>{t("scorersTitle")}</Heading>
      <div className="grid gap-6 sm:grid-cols-2">
        {sides.map((side) => (
          <div
            key={side.key}
            className="rounded-[var(--radius-card)] border border-border bg-surface p-5"
          >
            <p className="text-[0.68rem] font-semibold uppercase tracking-[0.12em] text-dim">
              {side.team}
            </p>
            <ul className="mt-4 flex flex-col gap-3.5">
              {side.list.map((p) => (
                <li key={p.player} className="flex items-center gap-3">
                  <span className="w-[7.5rem] shrink-0 truncate text-[0.87rem] font-medium sm:w-[9rem]">
                    {p.player}
                  </span>
                  <span className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
                    <span
                      className="block h-full rounded-full bg-brand"
                      style={{ width: `${(p.p / lead) * 100}%` }}
                    />
                  </span>
                  <span className="w-11 text-right text-[0.83rem] font-semibold">
                    {pct(p.p)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <p className="mt-4 text-[0.78rem] leading-relaxed text-dim">
        {t("scorersNote")}
      </p>
    </Reveal>
  );
}

export async function GoalMarkets({ match }: { match: Match }) {
  const t = await getTranslations("match");
  const g = match.prediction.goals;
  const rows = [
    { k: t("over15"), v: g.over15 },
    { k: t("over25"), v: g.over25 },
    { k: t("over35"), v: g.over35 },
    { k: t("btts"), v: g.btts },
  ];

  return (
    <Reveal as="section">
      <Heading>{t("goalsTitle")}</Heading>
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-[var(--radius-card)] border border-border bg-border sm:grid-cols-4">
        {rows.map((r) => (
          <div key={r.k} className="bg-surface px-5 py-5">
            <p className="text-[0.63rem] font-semibold uppercase tracking-[0.1em] text-dim">
              {r.k}
            </p>
            <p
              className={cn(
                "mt-2 text-[1.5rem] font-semibold leading-none",
                r.v >= 0.55 ? "text-positive" : r.v <= 0.45 ? "text-muted" : "text-text",
              )}
            >
              {pct(r.v)}
            </p>
          </div>
        ))}
      </div>
    </Reveal>
  );
}

export async function Scorelines({ match }: { match: Match }) {
  const t = await getTranslations("match");
  const lines = match.prediction.scorelines;
  const lead = lines[0]?.p ?? 1;

  return (
    <Reveal as="section">
      <Heading>{t("scorelinesTitle")}</Heading>
      <div className="rounded-[var(--radius-card)] border border-border bg-surface p-5 sm:p-6">
        <ul className="flex flex-col gap-2.5">
          {lines.map((s) => (
            <li key={s.score} className="flex items-center gap-3">
              <span className="w-10 shrink-0 text-[0.88rem] font-semibold">
                {s.score}
              </span>
              <span className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
                <span
                  className="block h-full rounded-full bg-brand"
                  style={{ width: `${(s.p / lead) * 100}%` }}
                />
              </span>
              <span className="w-12 text-right text-[0.82rem] text-muted">
                {pct(s.p, 1)}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </Reveal>
  );
}

/**
 * Corners, cards, fouls and shots.
 *
 * Each line shows its dominant side. A 26% over is a 74% under, and a column of
 * small percentages hides half of what the model is saying.
 */
export async function TeamMarkets({ match }: { match: Match }) {
  const t = await getTranslations("match");
  if (!match.teamProps?.length) return null;

  return (
    <Reveal as="section">
      <Heading note={t("teamNote")}>{t("teamTitle")}</Heading>
      <div className="grid gap-5 sm:grid-cols-2">
        {match.teamProps.map((m) => (
          <div
            key={m.key}
            className="rounded-[var(--radius-card)] border border-border bg-surface p-5"
          >
            <div className="flex items-baseline justify-between">
              <p className="text-[0.9rem] font-semibold">{t(`markets.${m.key}`)}</p>
              {m.lambdaTotal != null ? (
                <p className="text-[0.72rem] text-dim">
                  λ {Number(m.lambdaTotal).toFixed(2)}
                </p>
              ) : null}
            </div>
            <ul className="mt-4 flex flex-col gap-2.5">
              {m.lines.map((ln) => (
                <li key={ln.line} className="flex items-center gap-3">
                  <span className="w-14 shrink-0 text-[0.78rem] font-semibold uppercase">
                    {ln.side === "over" ? "O" : "U"} {ln.line}
                  </span>
                  <span className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
                    <span
                      className={cn(
                        "block h-full rounded-full",
                        ln.p >= 0.62
                          ? "bg-positive"
                          : ln.side === "over"
                            ? "bg-brand"
                            : "bg-warning",
                      )}
                      style={{ width: `${ln.p * 100}%` }}
                    />
                  </span>
                  <span className="w-11 text-right text-[0.8rem] font-medium">
                    {pct(ln.p)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </Reveal>
  );
}
