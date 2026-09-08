import { getTranslations } from "next-intl/server";

import type { Match } from "@/lib/api";

const pct = (v: number) => `${Math.round(v * 100)}%`;

/**
 * The answer, before any scrolling.
 *
 * One stacked bar across the full width, because three probabilities that sum to
 * one are a division of a whole and should look like one — three separate bars
 * would invite reading them as unrelated quantities.
 */
export async function Verdict({ match }: { match: Match }) {
  const t = await getTranslations("match");
  const p = match.prediction.probabilities;
  const xg = match.prediction.expectedGoals;
  const top = match.prediction.scorelines[0];

  const outcomes = [
    { key: "home", label: match.home, p: p.home, color: "var(--home)" },
    { key: "draw", label: t("draw"), p: p.draw, color: "var(--draw)" },
    { key: "away", label: match.away, p: p.away, color: "var(--away)" },
  ];
  const leader = outcomes.reduce((a, b) => (b.p > a.p ? b : a));

  return (
    <section>
      <div className="flex items-end justify-between gap-4">
        <h1 className="font-display text-[1.9rem] leading-[1.1] sm:text-[2.7rem]">
          {match.home}
          <span className="mx-3 text-[0.5em] align-middle text-dim">vs</span>
          {match.away}
        </h1>
      </div>

      <div className="mt-8 flex h-16 overflow-hidden rounded-[12px] sm:h-20">
        {outcomes.map((o) => (
          <div
            key={o.key}
            className="flex flex-col items-center justify-center text-white transition-[flex-grow] duration-700 ease-[var(--ease-out-quint)]"
            style={{ flexGrow: o.p, flexBasis: 0, background: o.color }}
          >
            <span className="text-[1.15rem] font-semibold leading-none sm:text-[1.5rem]">
              {pct(o.p)}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-2.5 flex justify-between text-[0.78rem] text-muted">
        {outcomes.map((o) => (
          <span
            key={o.key}
            className={o.key === leader.key ? "font-semibold text-text" : undefined}
          >
            {o.label}
          </span>
        ))}
      </div>

      <dl className="mt-8 grid grid-cols-2 gap-px overflow-hidden rounded-[var(--radius-card)] border border-border bg-border sm:grid-cols-4">
        {[
          { k: t("expectedGoals"), v: `${xg.home.toFixed(2)} – ${xg.away.toFixed(2)}` },
          { k: t("likeliestScore"), v: top ? top.score : "—" },
          { k: t("over25"), v: pct(match.prediction.goals.over25) },
          { k: t("btts"), v: pct(match.prediction.goals.btts) },
        ].map((m) => (
          <div key={m.k} className="bg-bg-elevated px-4 py-4">
            <dt className="text-[0.62rem] font-semibold uppercase tracking-[0.1em] text-dim">
              {m.k}
            </dt>
            <dd className="mt-1.5 text-[1.15rem] font-semibold">{m.v}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
