"use client";

import type { CSSProperties } from "react";

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
export function tierOf(overall: number, kind?: string) {
  // A special edition is not a rating band: a prime and an icon say something
  // about WHICH version of the player this is, and they keep their own face
  // whatever they are rated.
  if (kind === "icono") return "icono";
  if (kind === "prime") return "prime";
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
const AURA_TIERS = new Set(["elite", "prime", "icono"]);

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
}: {
  player: SquadPlayer;
  onClick?: () => void;
  delay?: number;
  selected?: boolean;
  size?: "sm" | "md";
}) {
  const overall = Math.round(player.overall);
  const tier = tierOf(overall, player.kind);
  const s = SIZE[size];
  const Tag = onClick ? "button" : "div";

  // A keeper's defensive axis *is* their shot-stopping, so showing both would
  // print the same number twice.
  const stats =
    player.position === "Goalkeeper"
      ? ([["GK", player.attack]] as const)
      : ([
          ["ATT", player.attack],
          ["DEF", player.defense],
        ] as const);

  return (
    <span className="relative block">
      {AURA_TIERS.has(tier) ? (
        <span
          aria-hidden
          className={cn("mv-aura", tier === "prime" && "mv-aura-prime",
            tier === "icono" && "mv-aura-icono")}
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
        {player.season ? (
          // The year IS the claim a prime makes — this is Suárez in 2015/16,
          // not Suárez — so it goes on the face, not in a tooltip.
          <span className="mv-card-year">{player.season}</span>
        ) : null}

        <span className="flex items-start justify-between gap-1.5">
          <span className={cn("font-semibold leading-none", s.ovr)}>{overall}</span>
          <span
            className={cn(
              "rounded-[5px] px-1.5 py-0.5 text-[0.55rem] font-bold tracking-[0.06em]",
              tier === "icono" ? "bg-black/10" : "bg-black/15",
            )}
          >
            {tier === "icono"
              ? "ICONO"
              : (POSITION_ABBR[player.position] ??
                player.position.slice(0, 2).toUpperCase())}
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
          <span className="mt-1.5 hidden items-center gap-2.5 border-t border-current/20 pt-1.5 sm:flex">
            {stats.map(([label, value]) => (
              <span key={label} className="flex items-baseline gap-1">
                <span className="text-[0.5rem] font-bold tracking-[0.05em] opacity-60">
                  {label}
                </span>
                <span className="text-[0.72rem] font-semibold leading-none">
                  {Math.round(value)}
                </span>
              </span>
            ))}
          </span>
        ) : null}
      </Tag>
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
