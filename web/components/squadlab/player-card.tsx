"use client";

import { Info } from "lucide-react";
import { type CSSProperties, useEffect, useRef, useState } from "react";

import { PlayerStatsPanel } from "@/components/squadlab/player-stats-popover";
import type { SquadPlayer } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * Rating tiers.
 *
 * The thresholds are read off the pool itself: 90+ is the handful of players
 * everybody recognises, 85+ the ones you would build around, 80+ solid, the
 * rest squad filler. Four bands is the most a reader can hold at a glance —
 * six would be a legend to memorise rather than a colour to feel.
 *
 * Pass the *rounded* overall, the one printed on the card: 84.6 shows as 85, and
 * a silver 85 sitting beside a gold 85 reads as a rendering fault.
 */
export function tierOf(overall: number) {
  if (overall >= 90) return "elite";
  if (overall >= 85) return "gold";
  if (overall >= 80) return "silver";
  return "bronze";
}

/**
 * Tiers that get the aura: a violet flare as the card lands, then a glow that
 * never quite settles. Only the top band today — a glow every card has is a
 * glow nobody notices — and the set is the hook for the special editions to
 * come, which will join it with their own `--aura`.
 */
const AURA_TIERS = new Set(["elite"]);

const POSITION_ABBR: Record<string, string> = {
  Goalkeeper: "GK",
  Defender: "DF",
  Midfielder: "MF",
  Forward: "FW",
};

/** Every card is the same size, whatever the name — a taller Lucas Vázquez
 *  would break the formation the pitch is trying to draw. */
const SIZE = {
  sm: {
    box: "h-[6.6rem]",
    ovr: "text-[1.3rem]",
    name: "text-[0.72rem]",
    team: "text-[0.58rem]",
  },
  md: {
    // Shorter on a phone, where the stats row is dropped: four defenders have
    // to fit across a 360px screen.
    box: "h-[7rem] sm:h-[8.6rem]",
    ovr: "text-[1.45rem] sm:text-[1.75rem]",
    name: "text-[0.76rem] sm:text-[0.84rem]",
    team: "text-[0.6rem] sm:text-[0.64rem]",
  },
} as const;

export function PlayerCard({
  player,
  onClick,
  delay = 0,
  selected = false,
  size = "md",
  roleWeights,
  substatLabels,
}: {
  player: SquadPlayer;
  onClick?: () => void;
  delay?: number;
  selected?: boolean;
  size?: "sm" | "md";
  /** All role formulas; when given (with labels) the card gets a stats popover. */
  roleWeights?: Record<string, Record<string, number>>;
  substatLabels?: Record<string, string>;
}) {
  const overall = Math.round(player.overall);
  const tier = tierOf(overall);
  const s = SIZE[size];
  const Tag = onClick ? "button" : "div";

  const canInspect = !!substatLabels && !!roleWeights;
  const wrapRef = useRef<HTMLSpanElement>(null);
  const [anchor, setAnchor] = useState<DOMRect | null>(null);
  const [hovered, setHovered] = useState(false);
  const [pinned, setPinned] = useState(false);
  const open = canInspect && (hovered || pinned) && anchor != null;

  const capture = () => {
    if (wrapRef.current) setAnchor(wrapRef.current.getBoundingClientRect());
  };

  // A pinned popover closes on the next click outside the card.
  useEffect(() => {
    if (!pinned) return;
    const onDown = (e: PointerEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setPinned(false);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [pinned]);

  // A keeper's defensive axis *is* their shot-stopping, so showing both would
  // print the same number twice.
  const stats =
    player.position === "Goalkeeper"
      ? ([["GK", player.attack]] as const)
      : ([
          ["ATT", player.attack],
          ["DEF", player.defense],
          ["CRE", player.creation],
        ] as const);

  return (
    <span
      ref={wrapRef}
      className="relative block"
      onMouseEnter={
        canInspect
          ? () => {
              capture();
              setHovered(true);
            }
          : undefined
      }
      onMouseLeave={canInspect ? () => setHovered(false) : undefined}
    >
      {AURA_TIERS.has(tier) ? (
        <span
          aria-hidden
          className="mv-aura"
          style={{ "--deal-delay": `${delay}ms` } as CSSProperties}
        />
      ) : null}
      <Tag
        type={onClick ? "button" : undefined}
        onClick={onClick}
        style={{ animationDelay: `${delay}ms` }}
        className={cn(
          "mv-card animate-deal text-left",
          `mv-card-${tier}`,
          s.box,
          selected && "mv-card-selected",
        )}
      >
        <span className="flex items-start justify-between gap-1.5">
          <span className={cn("font-semibold leading-none", s.ovr)}>{overall}</span>
          <span className="rounded-[5px] bg-black/15 px-1.5 py-0.5 text-[0.55rem] font-bold tracking-[0.06em]">
            {POSITION_ABBR[player.position] ??
              player.position.slice(0, 2).toUpperCase()}
          </span>
        </span>

        <span
          className={cn(
            "mt-auto block w-full truncate font-semibold leading-tight",
            s.name,
          )}
        >
          {player.display}
        </span>
        <span className={cn("block w-full truncate opacity-70", s.team)}>
          {player.team}
        </span>

        {size === "md" ? (
          // Numbers, not bars. On a card this small a bar can only say "about
          // half", and the whole point of a rating is that it is a figure you
          // can compare across two cards side by side.
          <span className="mt-1.5 hidden items-center justify-between gap-1 border-t border-current/20 pt-1.5 sm:flex">
            {stats.map(([label, value]) => (
              <span key={label} className="flex items-baseline gap-0.5">
                <span className="text-[0.5rem] font-bold tracking-[0.04em] opacity-60">
                  {label}
                </span>
                <span className="text-[0.7rem] font-semibold leading-none">
                  {value == null ? "—" : Math.round(value)}
                </span>
              </span>
            ))}
          </span>
        ) : null}
      </Tag>

      {canInspect ? (
        <button
          type="button"
          aria-label="Estadísticas avanzadas"
          onClick={(e) => {
            e.stopPropagation();
            capture();
            setPinned((p) => !p);
          }}
          className={cn(
            "absolute bottom-1 right-1 z-10 flex size-4 items-center justify-center rounded-full bg-black/25 text-white/80 transition-opacity hover:bg-black/45 hover:text-white",
            pinned ? "opacity-100" : "opacity-60",
          )}
        >
          <Info className="size-2.5" />
        </button>
      ) : null}

      {open ? (
        <PlayerStatsPanel
          player={player}
          roleWeights={roleWeights![player.role]}
          substatLabels={substatLabels!}
          anchor={anchor!}
        />
      ) : null}
    </span>
  );
}

export function EmptyCard({
  label,
  onClick,
  active = false,
  delay = 0,
}: {
  label: string;
  onClick?: () => void;
  active?: boolean;
  delay?: number;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{ animationDelay: `${delay}ms` }}
      className={cn(
        "mv-card mv-card-empty animate-deal items-center justify-center",
        SIZE.md.box,
        active && "mv-card-selected",
      )}
    >
      <span className="text-[1.4rem] leading-none opacity-60">+</span>
      <span className="mt-1.5 text-[0.6rem] font-semibold uppercase tracking-[0.08em]">
        {label}
      </span>
    </button>
  );
}
