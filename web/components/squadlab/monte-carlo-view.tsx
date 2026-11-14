"use client";

import { BarChart3, ChevronLeft, RotateCcw, Trophy } from "lucide-react";
import { useLocale, useTranslations } from "next-intl";
import { useMemo } from "react";

import type {
  MonteCarloChampions,
  MonteCarloLeague,
  MonteCarloResult,
  PositionProb,
  TeamProb,
} from "@/lib/api";
import { cn, ordinal } from "@/lib/utils";

function isChampions(r: MonteCarloResult): r is MonteCarloChampions {
  return !("mode" in r && r.mode === "league");
}

function ProbBar({ value, max }: { value: number; max: number }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div className="relative h-5 w-full overflow-hidden rounded-[5px] bg-border/40">
      <div
        className="absolute inset-y-0 left-0 rounded-[5px] bg-brand/70 transition-all duration-300"
        style={{ width: `${pct}%` }}
      />
      <span className="absolute inset-y-0 right-1.5 flex items-center text-[0.68rem] font-medium tabular-nums text-text">
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function PositionHistogram({
  positions,
  label,
}: {
  positions: PositionProb[];
  label: string;
}) {
  const maxProb = Math.max(...positions.map((p) => p.probability), 0.01);
  const nonZero = positions.filter((p) => p.count > 0);
  if (!nonZero.length) return null;

  const min = Math.min(...nonZero.map((p) => p.position));
  const max = Math.max(...nonZero.map((p) => p.position));
  const range = positions.slice(Math.max(0, min - 1), max);

  return (
    <div>
      <p className="mb-3 text-[0.72rem] font-semibold uppercase tracking-[0.1em] text-dim">
        {label}
      </p>
      <div className="flex items-end gap-[2px]" style={{ height: 100 }}>
        {range.map((p) => (
          <div
            key={p.position}
            className="group relative flex-1"
            style={{ height: "100%" }}
          >
            <div
              className={cn(
                "absolute inset-x-0 bottom-0 rounded-t-[3px] transition-all",
                p.probability === maxProb ? "bg-brand" : "bg-brand/50",
              )}
              style={{ height: `${(p.probability / maxProb) * 100}%`, minHeight: p.count > 0 ? 2 : 0 }}
            />
            <div className="absolute -top-6 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-md bg-bg-elevated px-1.5 py-0.5 text-[0.62rem] font-medium tabular-nums text-muted opacity-0 shadow-sm transition-opacity group-hover:opacity-100">
              #{p.position}: {(p.probability * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>
      <div className="mt-1 flex justify-between text-[0.62rem] tabular-nums text-dim">
        <span>#{min}</span>
        <span>#{max}</span>
      </div>
    </div>
  );
}

function TopChampionsTable({
  champions,
  t,
}: {
  champions: TeamProb[];
  t: (key: string, values?: Record<string, string | number>) => string;
}) {
  if (!champions.length) return null;
  return (
    <div>
      <p className="mb-3 text-[0.72rem] font-semibold uppercase tracking-[0.1em] text-dim">
        {t("mc.topChampions")}
      </p>
      <div className="overflow-hidden rounded-[10px] border border-border">
        <table className="w-full text-[0.8rem]">
          <tbody>
            {champions.map((c) => (
              <tr
                key={c.team}
                className={cn(
                  "border-b border-border last:border-0",
                  c.isSquad && "bg-brand-ghost",
                )}
              >
                <td className="px-3 py-2 font-medium">{c.team}</td>
                <td className="px-3 py-2 text-right tabular-nums text-muted">
                  {(c.probability * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function MonteCarloView({
  result,
  onBack,
  onReset,
}: {
  result: MonteCarloResult;
  onBack: () => void;
  onReset: () => void;
}) {
  const t = useTranslations("squadlab");
  const locale = useLocale();

  if (isChampions(result)) {
    return <ChampionsMC result={result} t={t} locale={locale} onBack={onBack} onReset={onReset} />;
  }
  return <LeagueMC result={result} t={t} locale={locale} onBack={onBack} onReset={onReset} />;
}

function ChampionsMC({
  result,
  t,
  locale,
  onBack,
  onReset,
}: {
  result: MonteCarloChampions;
  t: ReturnType<typeof useTranslations<"squadlab">>;
  locale: string;
  onBack: () => void;
  onReset: () => void;
}) {
  const reach = useMemo(() => {
    const order = ["Campeón", "Final", "Semifinales", "Cuartos", "Octavos", "Playoff"];
    return order
      .map((r) => {
        const found = result.reachProbabilities.find((x) => x.round === r);
        return found ? { round: r, probability: found.probability } : null;
      })
      .filter(Boolean) as { round: string; probability: number }[];
  }, [result.reachProbabilities]);

  const stages = useMemo(
    () => result.stages.filter((s) => s.count > 0),
    [result.stages],
  );
  const maxStageProb = Math.max(...stages.map((s) => s.probability), 0.01);

  return (
    <div>
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex h-9 items-center gap-2 rounded-[10px] border border-border px-3.5 text-[0.83rem] text-muted transition-colors hover:border-border-strong hover:text-text"
        >
          <ChevronLeft className="size-3.5" />
          {t("mc.backToRun")}
        </button>
        <button
          type="button"
          onClick={onReset}
          className="inline-flex h-9 items-center gap-2 rounded-[10px] border border-border px-3.5 text-[0.83rem] text-muted transition-colors hover:border-border-strong hover:text-text"
        >
          <RotateCcw className="size-3.5" />
          {t("newTeam")}
        </button>
      </div>

      {/* Hero */}
      <div className="mt-8 overflow-hidden rounded-[18px] border border-border bg-surface">
        <div className="relative px-7 py-9 text-center">
          <div
            aria-hidden
            className="pointer-events-none absolute left-1/2 top-0 size-[22rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-brand opacity-[0.12] blur-[110px]"
          />
          <BarChart3 className="relative mx-auto size-7 text-brand" />
          <p className="relative mt-3 text-[0.66rem] font-semibold uppercase tracking-[0.14em] text-dim">
            {t("mc.title")}
          </p>
          <p className="font-display relative mt-2 text-[2rem] leading-tight sm:text-[2.4rem]">
            {t("mc.nSimulations", { n: result.n })}
          </p>
          <p className="relative mt-2 text-[0.88rem] text-muted">
            Elo {Math.round(result.squadElo)}
          </p>

          {/* Headline stats */}
          <dl className="relative mx-auto mt-7 grid max-w-lg grid-cols-3 gap-px overflow-hidden rounded-[12px] border border-border bg-border">
            {[
              { label: t("mc.pChampion"), value: `${(reach.find((r) => r.round === "Campeón")?.probability ?? 0) * 100}%` },
              { label: t("mc.avgPoints"), value: result.averagePoints.toFixed(1) },
              { label: t("mc.avgGF"), value: result.averageGoalsFor.toFixed(1) },
            ].map((s) => (
              <div key={s.label} className="bg-bg-elevated px-3 py-3.5">
                <dt className="text-[0.58rem] font-semibold uppercase tracking-[0.09em] text-dim">
                  {s.label}
                </dt>
                <dd className="mt-1.5 text-[1.05rem] font-semibold tabular-nums">
                  {s.value}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </div>

      <div className="mt-8 grid gap-8 sm:grid-cols-2">
        {/* Stage distribution */}
        <div>
          <p className="mb-3 text-[0.72rem] font-semibold uppercase tracking-[0.1em] text-dim">
            {t("mc.stageDistribution")}
          </p>
          <div className="space-y-1.5">
            {stages.map((s) => (
              <div key={s.stage} className="grid grid-cols-[140px_1fr] items-center gap-2">
                <span className="truncate text-[0.78rem]">{s.stage}</span>
                <ProbBar value={s.probability} max={maxStageProb} />
              </div>
            ))}
          </div>
        </div>

        {/* Cumulative reach */}
        <div>
          <p className="mb-3 text-[0.72rem] font-semibold uppercase tracking-[0.1em] text-dim">
            {t("mc.reachProbability")}
          </p>
          <div className="overflow-hidden rounded-[10px] border border-border">
            <table className="w-full text-[0.8rem]">
              <tbody>
                {reach.map((r) => (
                  <tr key={r.round} className="border-b border-border last:border-0">
                    <td className="px-3 py-2">{r.round}</td>
                    <td className="px-3 py-2 text-right tabular-nums font-medium">
                      {(r.probability * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* League phase position histogram */}
      <div className="mt-8">
        <PositionHistogram
          positions={result.leaguePhasePositions}
          label={t("mc.positionDistribution")}
        />
      </div>

      {/* Top champions */}
      <div className="mt-8">
        <TopChampionsTable champions={result.topChampions} t={t} />
      </div>
    </div>
  );
}

function LeagueMC({
  result,
  t,
  locale,
  onBack,
  onReset,
}: {
  result: MonteCarloLeague;
  t: ReturnType<typeof useTranslations<"squadlab">>;
  locale: string;
  onBack: () => void;
  onReset: () => void;
}) {
  return (
    <div>
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex h-9 items-center gap-2 rounded-[10px] border border-border px-3.5 text-[0.83rem] text-muted transition-colors hover:border-border-strong hover:text-text"
        >
          <ChevronLeft className="size-3.5" />
          {t("mc.backToRun")}
        </button>
        <button
          type="button"
          onClick={onReset}
          className="inline-flex h-9 items-center gap-2 rounded-[10px] border border-border px-3.5 text-[0.83rem] text-muted transition-colors hover:border-border-strong hover:text-text"
        >
          <RotateCcw className="size-3.5" />
          {t("newTeam")}
        </button>
      </div>

      {/* Hero */}
      <div className="mt-8 overflow-hidden rounded-[18px] border border-border bg-surface">
        <div className="relative px-7 py-9 text-center">
          <div
            aria-hidden
            className="pointer-events-none absolute left-1/2 top-0 size-[22rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-brand opacity-[0.12] blur-[110px]"
          />
          <BarChart3 className="relative mx-auto size-7 text-brand" />
          <p className="relative mt-3 text-[0.66rem] font-semibold uppercase tracking-[0.14em] text-dim">
            {t("mc.title")}
          </p>
          <p className="font-display relative mt-2 text-[2rem] leading-tight sm:text-[2.4rem]">
            {t("mc.nSimulations", { n: result.n })}
          </p>

          <dl className="relative mx-auto mt-7 grid max-w-lg grid-cols-3 gap-px overflow-hidden rounded-[12px] border border-border bg-border">
            {[
              { label: t("mc.pChampion"), value: `${(result.pChampion * 100).toFixed(1)}%` },
              { label: t("mc.pTop4"), value: `${(result.pTop4 * 100).toFixed(1)}%` },
              { label: t("mc.avgPoints"), value: result.averagePoints.toFixed(1) },
            ].map((s) => (
              <div key={s.label} className="bg-bg-elevated px-3 py-3.5">
                <dt className="text-[0.58rem] font-semibold uppercase tracking-[0.09em] text-dim">
                  {s.label}
                </dt>
                <dd className="mt-1.5 text-[1.05rem] font-semibold tabular-nums">
                  {s.value}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </div>

      <div className="mt-8 grid gap-8 sm:grid-cols-2">
        {/* Position distribution */}
        <div>
          <PositionHistogram
            positions={result.positions}
            label={t("mc.positionDistribution")}
          />
        </div>

        {/* Top champions */}
        <TopChampionsTable champions={result.topChampions} t={t} />
      </div>

      {/* Relegation risk */}
      {result.pRelegation > 0 && (
        <div className="mt-6 rounded-[10px] border border-negative/30 bg-negative/5 px-4 py-3">
          <p className="text-[0.82rem] font-medium text-negative">
            {t("mc.relegationRisk", { pct: (result.pRelegation * 100).toFixed(1) })}
          </p>
        </div>
      )}
    </div>
  );
}
