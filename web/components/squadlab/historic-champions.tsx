"use client";

import { Dices, Loader2, RotateCcw, Star, Trophy } from "lucide-react";
import { useTranslations } from "next-intl";
import { useState } from "react";

import { api, type BracketTie, type HistoricChampions } from "@/lib/api";
import { cn } from "@/lib/utils";

const POOLS = ["campeones", "variado", "cualquiera"] as const;
const ROUNDS = ["final", "sf", "qf", "r16", "playoff"] as const;

function mintSeed(): number {
  const [n] = crypto.getRandomValues(new Uint32Array(1));
  return n % 1_000_000;
}

const pct = (p: number) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;

/**
 * La Champions histórica: 36 sides from any era, in today's format.
 *
 * Every side is its club's real ClubElo on 1 June of the year the season ended,
 * so the comparison is "how far above the Europe of its day", the one thing
 * that can be measured across seventy years.
 */
export function HistoricChampionsView({ onReset }: { onReset: () => void }) {
  const t = useTranslations("squadlab");
  const [pool, setPool] = useState<(typeof POOLS)[number]>("campeones");
  const [result, setResult] = useState<HistoricChampions | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  async function play(seed = mintSeed()) {
    setLoading(true);
    setError(false);
    try {
      setResult(await api.historicChampions(pool, seed));
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={onReset}
          className="inline-flex h-9 items-center gap-2 rounded-[10px] border border-border px-3.5 text-[0.83rem] text-muted transition-colors hover:border-border-strong hover:text-text"
        >
          <RotateCcw className="size-3.5" />
          {t("changeSetup")}
        </button>
      </div>

      <div className="mt-6 max-w-2xl">
        <p className="font-display text-[1.6rem] leading-tight">{t("historic.title")}</p>
        <p className="mt-2 text-[0.88rem] leading-relaxed text-muted">{t("historic.lead")}</p>
      </div>

      <div className="mt-6 flex max-w-2xl flex-col gap-2">
        {POOLS.map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => setPool(p)}
            className={cn(
              "rounded-[13px] border px-5 py-3.5 text-left transition-colors duration-200",
              pool === p
                ? "border-brand bg-brand-ghost"
                : "border-border bg-surface hover:border-border-strong",
            )}
          >
            <span className="text-[0.95rem] font-semibold">{t(`historic.pool.${p}`)}</span>
            <span className="mt-0.5 block text-[0.8rem] leading-relaxed text-muted">
              {t(`historic.pool.${p}Lead`)}
            </span>
          </button>
        ))}
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <button
          type="button"
          disabled={loading}
          onClick={() => play()}
          className="inline-flex h-12 items-center gap-2 rounded-[10px] bg-brand px-6 text-[0.95rem] font-medium text-brand-contrast transition-all duration-200 hover:bg-brand-hover disabled:pointer-events-none disabled:opacity-40"
        >
          {loading ? <Loader2 className="size-4 animate-spin" /> : <Dices className="size-4" />}
          {result ? t("historic.again") : t("historic.play")}
        </button>
      </div>
      {error ? <p className="mt-4 text-[0.85rem] text-negative">{t("seasonFailed")}</p> : null}

      {result ? <HistoricResult result={result} /> : null}
    </div>
  );
}

function HistoricResult({ result }: { result: HistoricChampions }) {
  const t = useTranslations("squadlab");
  const champs = new Set(result.entrants.filter((e) => e.europeanChampion).map((e) => e.team));

  return (
    <div className="mt-8 flex flex-col gap-10">
      <div className="overflow-hidden rounded-[18px] border border-border bg-surface">
        <div className="relative px-7 py-9 text-center">
          <div
            aria-hidden
            className="pointer-events-none absolute left-1/2 top-0 size-[22rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-brand opacity-[0.16] blur-[110px]"
          />
          <Trophy className="relative mx-auto size-7 text-warning" />
          <p className="relative mt-4 text-[0.66rem] font-semibold uppercase tracking-[0.14em] text-dim">
            {t("historic.champion")}
          </p>
          <p className="font-display relative mt-2 text-[2rem] leading-tight sm:text-[2.6rem]">
            {result.champion}
          </p>
          <p className="relative mt-3 text-[0.92rem] text-muted">
            {t("historic.beat", { team: result.runnerUp })}
            {result.championOdds != null
              ? ` · ${t("historic.odds", { p: pct(result.championOdds) })}`
              : ""}
          </p>
        </div>
      </div>

      <section>
        <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
          {t("historic.path")}
        </h2>
        <div className="mt-3 flex flex-col gap-4">
          {ROUNDS.map((r) => {
            const ties = result.bracket[r] ?? [];
            if (!ties.length) return null;
            return (
              <div key={r}>
                <p className="text-[0.62rem] font-semibold uppercase tracking-[0.1em] text-dim">
                  {ties[0].roundLabel}
                </p>
                <div className="mt-1.5 flex flex-col gap-1">
                  {ties.map((tie, i) => (
                    <TieRow key={`${r}-${i}`} tie={tie} champion={result.champion} />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section>
        <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
          {t("historic.favourites")}
        </h2>
        <div className="mt-3 flex flex-col gap-1.5">
          {result.favourites.map((f, i) => (
            <div
              key={f.team}
              className={cn(
                "flex items-center gap-3 rounded-[11px] border px-4 py-2",
                f.team === result.champion ? "border-warning/40 bg-warning/5" : "border-border bg-surface",
              )}
            >
              <span className="w-5 text-[0.72rem] tabular-nums text-dim">{i + 1}</span>
              <span className="flex-1 truncate text-[0.85rem]">{f.team}</span>
              <span className="text-[0.7rem] tabular-nums text-dim">Elo {f.elo}</span>
              <span className="w-14 text-right text-[0.85rem] font-semibold tabular-nums">
                {pct(f.pChampion)}
              </span>
            </div>
          ))}
        </div>
        <p className="mt-2 text-[0.72rem] text-dim">{t("historic.favouritesNote")}</p>
      </section>

      <section>
        <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
          {t("leaguePhaseTable")}
        </h2>
        <div className="mt-3 overflow-x-auto rounded-[var(--radius-card)] border border-border">
          <table className="w-full min-w-[20rem] text-[0.82rem]">
            <thead>
              <tr className="border-b border-border bg-bg-elevated text-[0.6rem] uppercase tracking-[0.07em] text-dim">
                <th className="px-3 py-2 text-left font-semibold">#</th>
                <th className="px-3 py-2 text-left font-semibold">{t("colTeam")}</th>
                <th className="px-2 py-2 text-right font-semibold">Elo</th>
                <th className="px-3 py-2 text-right font-semibold">{t("colPoints")}</th>
              </tr>
            </thead>
            <tbody>
              {result.leaguePhase.map((r) => (
                <tr
                  key={r.team}
                  className={cn(
                    "border-b border-border bg-surface last:border-0",
                    r.rank === 9 || r.rank === 25 ? "border-t-2 border-t-border-strong" : "",
                  )}
                >
                  <td className="px-3 py-2 tabular-nums text-dim">{r.rank}</td>
                  <td className="px-3 py-2">
                    {r.team}
                    {champs.has(r.team) ? (
                      <Star
                        aria-label={t("historic.wasChampion")}
                        className="ml-1.5 inline size-3 text-warning"
                        fill="currentColor"
                        strokeWidth={0}
                      />
                    ) : null}
                  </td>
                  <td className="px-2 py-2 text-right tabular-nums text-muted">{r.elo}</td>
                  <td className="px-3 py-2 text-right font-semibold tabular-nums">{r.points}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-[0.7rem] text-dim">{t("tableLegend")}</p>
      </section>
    </div>
  );
}

function TieRow({ tie, champion }: { tie: BracketTie; champion: string }) {
  const name = (x: string) => (
    <span className={cn(x === tie.winner ? "font-semibold text-text" : "text-muted")}>
      {x}
      {x === champion ? " ★" : ""}
    </span>
  );
  return (
    <div className="flex items-center gap-2 rounded-[10px] border border-border bg-surface px-3 py-2 text-[0.82rem]">
      <span className="flex-1 truncate text-right">{name(tie.teamA)}</span>
      <span className="w-20 shrink-0 text-center font-semibold tabular-nums text-brand">
        {tie.agg}
        {tie.note ? <span className="ml-1 text-[0.66rem] text-warning">{tie.note}</span> : null}
      </span>
      <span className="flex-1 truncate">{name(tie.teamB)}</span>
    </div>
  );
}
