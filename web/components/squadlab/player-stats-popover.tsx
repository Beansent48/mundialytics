"use client";

import { useTranslations } from "next-intl";
import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import type { SquadPlayer } from "@/lib/api";
import { cn } from "@/lib/utils";

function espnPhotoUrl(espnId: number): string {
  return `https://a.espncdn.com/combiner/i?img=/i/headshots/soccer/players/full/${espnId}.png&w=130&h=95`;
}

const POSITION_LABEL: Record<string, string> = {
  Goalkeeper: "GK",
  Defender: "DEF",
  Midfielder: "MID",
  Forward: "FWD",
};

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
  const [imgError, setImgError] = useState(false);

  const isPrime = player.kind === "prime";
  const isIcono = player.kind === "icono";
  const isSpecial = isPrime || isIcono;
  const enrichment = player.enrichment;
  const hasPhoto = !!enrichment?.espnId && !imgError;

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
  const weighted = Object.entries(weights).sort((a, b) => b[1] - a[1]);
  const weightedCodes = new Set(weighted.map(([c]) => c));
  const subs = player.substats ?? {};
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

  const infoChips: string[] = [];
  if (enrichment?.age != null) {
    if (isPrime && enrichment.ageAtSeason != null && player.season) {
      infoChips.push(t("playerAgeAtSeason", { age: enrichment.ageAtSeason, season: player.season }));
    } else {
      infoChips.push(t("playerAge", { age: enrichment.age }));
    }
  }
  if (enrichment?.heightCm != null) {
    infoChips.push(t("playerHeight", { cm: enrichment.heightCm }));
  }
  if (enrichment?.foot) {
    infoChips.push(t("playerFoot", { foot: enrichment.foot }));
  }
  if (enrichment?.nationality) {
    infoChips.push(enrichment.nationality);
  }

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
      className="z-[60] w-[20rem] max-w-[calc(100vw-1rem)] overflow-hidden rounded-[14px] border border-border-strong bg-surface shadow-[0_18px_40px_-12px_rgba(0,0,0,0.6)]"
    >
      {/* ── Header: photo + identity ── */}
      <div className="flex items-start gap-3 border-b border-border px-4 py-3">
        {hasPhoto ? (
          <div className="relative size-14 shrink-0 overflow-hidden rounded-lg bg-surface-2">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={espnPhotoUrl(enrichment!.espnId!)}
              alt={player.display}
              className="absolute inset-0 size-full object-cover object-top"
              onError={() => setImgError(true)}
            />
          </div>
        ) : (
          <div className="flex size-14 shrink-0 items-center justify-center rounded-lg bg-surface-2 text-[1.2rem] font-bold text-dim">
            {player.display.charAt(0)}
          </div>
        )}
        <div className="min-w-0 flex-1">
          <p className="truncate text-[0.92rem] font-semibold leading-tight">
            {player.display}
          </p>
          {player.fullName && player.fullName !== player.display ? (
            <p className="mt-0.5 truncate text-[0.66rem] text-dim">
              {player.fullName}
            </p>
          ) : null}
          <p className="mt-1 flex flex-wrap items-center gap-1.5 text-[0.68rem] text-muted">
            <span className="truncate">{player.team}</span>
            <span className="text-dim">·</span>
            <span>{POSITION_LABEL[player.position] ?? player.position}</span>
            {isSpecial ? (
              <>
                <span className="text-dim">·</span>
                <span className="rounded-[5px] bg-black/25 px-1.5 py-0.5 text-[0.6rem] font-semibold">
                  {isPrime ? `PRIME${player.season ? ` ${player.season}` : ""}` : "ICONO"}
                </span>
              </>
            ) : player.role ? (
              <>
                <span className="text-dim">·</span>
                <span className="rounded-[5px] bg-brand-ghost px-1.5 py-0.5 text-[0.6rem] font-semibold text-brand">
                  {player.role}
                </span>
              </>
            ) : null}
          </p>
        </div>
      </div>

      {/* ── Info chips: age, height, foot, nationality ── */}
      {infoChips.length > 0 ? (
        <div className="flex flex-wrap gap-1.5 border-b border-border px-4 py-2.5">
          {infoChips.map((chip) => (
            <span
              key={chip}
              className="rounded-md bg-surface-2 px-2 py-0.5 text-[0.66rem] font-medium text-muted"
            >
              {chip}
            </span>
          ))}
          <span className="rounded-md bg-surface-2 px-2 py-0.5 text-[0.66rem] font-medium text-muted">
            {t("playerMatches", { n: player.matches })}
          </span>
        </div>
      ) : null}

      {/* ── Rating axes ── */}
      <div className="border-b border-border px-4 py-2.5">
        <div className={cn("grid gap-2", isGk ? "grid-cols-1" : "grid-cols-3")}>
          {isGk ? (
            <AxisBar
              label={t("statsGk")}
              value={player.attack}
              tier={axisTier(player.attack)}
            />
          ) : (
            <>
              <AxisBar
                label={t("statsAttack")}
                value={player.attack}
                tier={axisTier(player.attack)}
              />
              <AxisBar
                label={t("statsDefense")}
                value={player.defense}
                tier={axisTier(player.defense)}
              />
              <AxisBar
                label={t("statsCreation")}
                value={player.creation}
                tier={axisTier(player.creation)}
              />
            </>
          )}
        </div>
      </div>

      {/* ── Substats / highlight ── */}
      <div className="max-h-[18rem] overflow-y-auto px-4 py-3">
        {isSpecial ? (
          player.highlight && (player.highlight.stat || player.highlight.note) ? (
            <>
              <p className="text-[0.6rem] font-semibold uppercase tracking-[0.1em] text-dim">
                {isPrime ? t("primeSeason") : t("careerHighlight")}
              </p>
              {player.highlight.stat ? (
                <p className="mt-2 text-[0.96rem] font-semibold leading-snug text-text">
                  {player.highlight.stat}
                </p>
              ) : null}
              {player.highlight.note ? (
                <p className="mt-2 text-[0.83rem] leading-relaxed text-muted">
                  {player.highlight.note}
                </p>
              ) : null}
            </>
          ) : (
            <p className="text-[0.82rem] leading-relaxed text-muted">
              {t("noHighlight")}
            </p>
          )
        ) : (
          <>
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
          </>
        )}
      </div>
    </div>,
    document.body,
  );
}

function axisTier(value: number | null): "high" | "mid" | "low" {
  if (value == null) return "low";
  if (value >= 80) return "high";
  if (value >= 65) return "mid";
  return "low";
}

function AxisBar({
  label,
  value,
  tier,
}: {
  label: string;
  value: number | null;
  tier: "high" | "mid" | "low";
}) {
  const v = value != null ? Math.round(value) : null;
  const tone =
    tier === "high"
      ? "bg-positive"
      : tier === "mid"
        ? "bg-brand"
        : "bg-negative/70";
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <span className="text-[0.6rem] font-semibold uppercase tracking-[0.06em] text-dim">
          {label}
        </span>
        <span className="text-[0.78rem] font-bold tabular-nums">{v ?? "—"}</span>
      </div>
      <span className="h-1 overflow-hidden rounded-full bg-surface-3">
        <span
          className={cn("block h-full rounded-full transition-all", tone)}
          style={{ width: `${v == null ? 0 : Math.max(3, v)}%` }}
        />
      </span>
    </div>
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
