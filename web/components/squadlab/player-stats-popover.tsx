"use client";

import { useTranslations } from "next-intl";
import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import type { SquadPlayer } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * The depth behind a card: the measured sub-stats the role rating is built
 * from, and the role's own weight formula. Percentiles are 0-100 within the
 * player's (season, position), so 88 means "better than 88% of his position".
 *
 * Rendered in a portal with fixed positioning so it never clips inside the
 * pitch or the candidate grid, both of which can hide overflow.
 */
export function PlayerStatsPanel({
  player,
  roleWeights,
  substatLabels,
  anchor,
}: {
  player: SquadPlayer;
  roleWeights: Record<string, number> | undefined;
  substatLabels: Record<string, string>;
  anchor: DOMRect;
}) {
  const t = useTranslations("squadlab");
  const panelRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);

  // Place below the card, flip above when there is no room, clamp to viewport.
  useLayoutEffect(() => {
    const el = panelRef.current;
    if (!el) return;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    const gap = 8;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    let left = anchor.left + anchor.width / 2 - w / 2;
    left = Math.max(8, Math.min(left, vw - w - 8));
    let top = anchor.bottom + gap;
    if (top + h > vh - 8) {
      const above = anchor.top - gap - h;
      top = above >= 8 ? above : Math.max(8, vh - h - 8);
    }
    setPos({ left, top });
  }, [anchor]);

  const weights = roleWeights ?? {};
  // The role's own stats first, heaviest first — that is the formula. Then any
  // other measured stat, so nothing the engine saw is hidden.
  const weighted = Object.entries(weights).sort((a, b) => b[1] - a[1]);
  const weightedCodes = new Set(weighted.map(([c]) => c));
  const subs = player.substats ?? {};
  // Keeper stats are noise on an outfielder's card and vice versa — the profile
  // file scores every code for everyone, so filter to the player's own kind.
  const GK_CODES = new Set(["SHOTSTOP", "AERIAL_GK", "SWEEP", "DISTRIB"]);
  const isGk = player.position === "Goalkeeper";
  const extra = Object.keys(subs)
    .filter(
      (c) =>
        !weightedCodes.has(c) &&
        substatLabels[c] &&
        (isGk ? GK_CODES.has(c) : !GK_CODES.has(c)),
    )
    .sort((a, b) => (subs[b] ?? 0) - (subs[a] ?? 0));

  const hasProfile = player.substats != null;

  return createPortal(
    <div
      ref={panelRef}
      role="dialog"
      style={{
        position: "fixed",
        left: pos?.left ?? -9999,
        top: pos?.top ?? -9999,
        visibility: pos ? "visible" : "hidden",
      }}
      className="z-[60] w-[19rem] max-w-[calc(100vw-1rem)] overflow-hidden rounded-[14px] border border-border-strong bg-surface shadow-[0_18px_40px_-12px_rgba(0,0,0,0.6)]"
    >
      <div className="border-b border-border px-4 py-3">
        <p className="truncate text-[0.9rem] font-semibold">{player.display}</p>
        <p className="mt-0.5 flex items-center gap-1.5 text-[0.72rem] text-muted">
          <span className="truncate">{player.team}</span>
          {player.role ? (
            <>
              <span className="text-dim">·</span>
              <span className="rounded-[5px] bg-brand-ghost px-1.5 py-0.5 text-[0.66rem] font-semibold text-brand">
                {player.role}
              </span>
            </>
          ) : null}
        </p>
      </div>

      <div className="max-h-[22rem] overflow-y-auto px-4 py-3">
        {weighted.length ? (
          <>
            <p className="text-[0.6rem] font-semibold uppercase tracking-[0.1em] text-dim">
              {t("roleFormula")}
            </p>
            <div className="mt-2 flex flex-col gap-2">
              {weighted.map(([code, weight]) => (
                <StatRow
                  key={code}
                  label={substatLabels[code] ?? code}
                  code={code}
                  weight={weight}
                  value={hasProfile ? (subs[code] ?? null) : null}
                />
              ))}
            </div>
          </>
        ) : null}

        {extra.length ? (
          <>
            <p className="mt-4 text-[0.6rem] font-semibold uppercase tracking-[0.1em] text-dim">
              {t("otherStats")}
            </p>
            <div className="mt-2 flex flex-col gap-2">
              {extra.map((code) => (
                <StatRow
                  key={code}
                  label={substatLabels[code] ?? code}
                  code={code}
                  weight={null}
                  value={subs[code] ?? null}
                />
              ))}
            </div>
          </>
        ) : null}

        {!hasProfile ? (
          <p className="mt-2 text-[0.76rem] leading-relaxed text-dim">
            {t("noAdvancedData")}
          </p>
        ) : null}

        <p className="mt-3 border-t border-border pt-2 text-[0.66rem] leading-relaxed text-dim">
          {t("percentileNote")}
        </p>
      </div>
    </div>,
    document.body,
  );
}

function StatRow({
  label,
  code,
  weight,
  value,
}: {
  label: string;
  code: string;
  weight: number | null;
  value: number | null;
}) {
  const pct = value ?? 0;
  const tone =
    value == null
      ? "bg-surface-3"
      : pct >= 75
        ? "bg-positive"
        : pct >= 45
          ? "bg-brand"
          : "bg-negative/70";
  return (
    <div className="flex items-center gap-2.5">
      <span className="flex w-[8.5rem] shrink-0 items-baseline gap-1">
        <span className="truncate text-[0.76rem]">{label}</span>
        <span className="text-[0.56rem] font-semibold uppercase tracking-[0.04em] text-dim">
          {code}
        </span>
      </span>
      {weight != null ? (
        <span className="w-8 shrink-0 text-right text-[0.62rem] font-semibold tabular-nums text-brand">
          {weight}%
        </span>
      ) : (
        <span className="w-8 shrink-0" />
      )}
      <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-3">
        <span
          className={cn("block h-full rounded-full", tone)}
          style={{ width: `${value == null ? 0 : Math.max(3, pct)}%` }}
        />
      </span>
      <span className="w-6 shrink-0 text-right text-[0.72rem] font-semibold tabular-nums">
        {value == null ? "—" : pct}
      </span>
    </div>
  );
}
