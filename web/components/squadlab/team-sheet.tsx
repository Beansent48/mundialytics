"use client";

import { Loader2 } from "lucide-react";
import { useTranslations } from "next-intl";

import { PlayerCard } from "@/components/squadlab/player-card";
import type { SquadPlayer } from "@/lib/api";

const mean = (xs: number[]) =>
  xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;

/**
 * What you signed, while the engine plays it out.
 *
 * Thirty-odd seconds of a spinner is dead time; thirty seconds spent reading
 * your own team sheet is anticipation. Every figure here is computed from the
 * eleven the user picked — nothing is invented to fill the wait.
 */
export function TeamSheet({ squad }: { squad: SquadPlayer[] }) {
  const t = useTranslations("squadlab");

  const overall = Math.round(mean(squad.map((p) => p.overall)));
  const outfield = squad.filter((p) => p.position !== "Goalkeeper");
  const keeper = squad.find((p) => p.position === "Goalkeeper");

  // Four figures that do not overlap. Two of them labelled "Attack" — the axis
  // and the forward line — would just be a puzzle.
  const cells = [
    { key: "attack", label: t("sheetAttack"), value: Math.round(mean(outfield.map((p) => p.attack))) },
    { key: "defense", label: t("sheetDefense"), value: Math.round(mean(outfield.map((p) => p.defense))) },
    { key: "keeper", label: t("sheetKeeper"), value: keeper ? Math.round(keeper.attack) : 0 },
    { key: "clubs", label: t("sheetClubs"), value: new Set(squad.map((p) => p.team)).size },
  ];

  const top = [...squad].sort((a, b) => b.overall - a.overall).slice(0, 3);

  return (
    <div className="overflow-hidden rounded-[18px] border border-border bg-surface">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border px-6 py-5">
        <div>
          <p className="text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-brand">
            {t("sheetEyebrow")}
          </p>
          <p className="font-display mt-1.5 text-[1.5rem] leading-none">
            {t("sheetTitle")}
          </p>
        </div>
        <div className="text-right">
          <p className="text-[0.58rem] font-semibold uppercase tracking-[0.1em] text-dim">
            {t("sheetOverall")}
          </p>
          <p className="font-display text-[2.4rem] leading-none">{overall}</p>
        </div>
      </div>

      <dl className="grid grid-cols-2 gap-px bg-border sm:grid-cols-4">
        {cells.map((cell) => (
          <div key={cell.key} className="bg-bg-elevated px-4 py-3.5">
            <dt className="text-[0.56rem] font-semibold uppercase tracking-[0.09em] text-dim">
              {cell.label}
            </dt>
            <dd className="mt-1 text-[1.15rem] font-semibold tabular-nums">
              {cell.value}
            </dd>
          </div>
        ))}
      </dl>

      <div className="px-6 py-6">
        <p className="text-[0.62rem] font-semibold uppercase tracking-[0.13em] text-muted">
          {t("sheetTop")}
        </p>
        <div className="mt-4 flex gap-3">
          {top.map((p, i) => (
            <div key={p.player} className="w-[6.4rem] sm:w-[7.4rem]">
              <PlayerCard player={p} delay={i * 140} />
            </div>
          ))}
        </div>

        <p className="mt-7 flex items-center gap-2 text-[0.85rem] text-muted">
          <Loader2 className="size-3.5 animate-spin text-brand" />
          {t("playing")}
        </p>
      </div>
    </div>
  );
}
