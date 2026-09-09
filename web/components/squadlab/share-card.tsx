"use client";

import { Copy, Download, Share2, X } from "lucide-react";
import { useLocale, useTranslations } from "next-intl";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";

import { tierOf } from "@/components/squadlab/player-card";
import type { SquadPlayer, SquadSeason } from "@/lib/api";
import { ordinal } from "@/lib/utils";

const SIZE = 1080;

/** The card gradients, kept in step with the ones in globals.css. */
const TIER_COLORS: Record<string, [string, string, string, string]> = {
  elite: ["#b79bff", "#6d3fff", "#3c1d9e", "#ffffff"],
  gold: ["#ffe08a", "#e0a52e", "#a97016", "#12141c"],
  silver: ["#eef1f6", "#b8c0cf", "#8a93a5", "#12141c"],
  bronze: ["#e6b98d", "#b9793f", "#8a5424", "#12141c"],
};

const mean = (xs: number[]) =>
  xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;

/** Whether the browser can share files never changes within a page's life. */
const subscribeNever = () => () => {};

function shield(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r = 22,
  taper = 34,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - taper);
  ctx.lineTo(x + w / 2, y + h);
  ctx.lineTo(x, y + h - taper);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function fitText(ctx: CanvasRenderingContext2D, text: string, max: number) {
  if (ctx.measureText(text).width <= max) return text;
  let cut = text;
  while (cut.length > 1 && ctx.measureText(`${cut}…`).width > max) {
    cut = cut.slice(0, -1);
  }
  return `${cut}…`;
}

/**
 * The season, as one image.
 *
 * Drawn on a canvas rather than captured from the DOM: the cards are built out
 * of clip-path, drop-shadow filters and color-mix, and every HTML-to-image
 * library mangles exactly those. Drawing the shield by hand costs a hundred
 * lines and gets a file that looks like the page.
 */
export function ShareCard({
  squad,
  season,
  squadLabel,
  onClose,
}: {
  squad: SquadPlayer[];
  season: SquadSeason;
  squadLabel: string;
  onClose: () => void;
}) {
  const t = useTranslations("squadlab");
  const locale = useLocale();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [copied, setCopied] = useState(false);

  // A browser capability, not React state: read it through the store hook so
  // the server renders "no" and the client corrects it without an effect.
  const canShare = useSyncExternalStore(
    subscribeNever,
    () => typeof navigator !== "undefined" && !!navigator.canShare,
    () => false,
  );

  const competitionName = "Champions League";
  const championLabel =
    season.champion === season.teamName ? squadLabel : season.champion;
  const outfield = squad.filter((p) => p.position !== "Goalkeeper");
  const overall = Math.round(mean(squad.map((p) => p.overall)));
  const attack = Math.round(mean(outfield.map((p) => p.attack)));
  const defense = Math.round(mean(outfield.map((p) => p.defense)));
  const top = [...squad].sort((a, b) => b.overall - a.overall).slice(0, 3);
  const scorer = season.scorers.find((s) => s.isSquad) ?? season.scorers[0];

  const summary = [
    `${squadLabel} · ${competitionName}`,
    t("summaryFinish", {
      stage: season.stage,
      rank: ordinal(season.leaguePhaseRank, locale),
    }),
    scorer ? t("summaryScorer", { player: scorer.player, goals: scorer.goals }) : "",
    "mundialytics",
  ]
    .filter(Boolean)
    .join("\n");

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    // The page's own font stack, so the image matches the app rather than
    // falling back to whatever canvas defaults to.
    const family =
      getComputedStyle(document.body).fontFamily || "system-ui, sans-serif";
    const font = (weight: number, size: number) =>
      `${weight} ${size}px ${family}`;

    ctx.clearRect(0, 0, SIZE, SIZE);

    const bg = ctx.createLinearGradient(0, 0, SIZE, SIZE);
    bg.addColorStop(0, "#0a0b10");
    bg.addColorStop(0.55, "#0e1018");
    bg.addColorStop(1, "#140f24");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, SIZE, SIZE);

    const glow = ctx.createRadialGradient(SIZE / 2, 40, 0, SIZE / 2, 40, 620);
    glow.addColorStop(0, "rgba(139,92,255,0.42)");
    glow.addColorStop(1, "rgba(139,92,255,0)");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, SIZE, 700);

    ctx.textAlign = "center";

    ctx.letterSpacing = "7px";
    ctx.font = font(700, 24);
    ctx.fillStyle = "#8b5cff";
    ctx.fillText("MUNDIALYTICS · SQUADLAB", SIZE / 2, 96);
    ctx.letterSpacing = "0px";

    ctx.font = font(600, 40);
    ctx.fillStyle = "#f2f4f8";
    ctx.fillText(`${squadLabel} · ${competitionName}`, SIZE / 2, 168);

    // The stage is a phrase ("Campeón", "Eliminado en octavos"), not a number,
    // so the headline scales down to fit the width instead of clipping.
    let stageSize = 120;
    ctx.font = font(700, stageSize);
    while (ctx.measureText(season.stage).width > SIZE - 120 && stageSize > 48) {
      stageSize -= 6;
      ctx.font = font(700, stageSize);
    }
    ctx.fillStyle = "#ffffff";
    ctx.fillText(season.stage, SIZE / 2, 330);

    ctx.font = font(500, 34);
    ctx.fillStyle = "#99a1b3";
    ctx.fillText(
      t("finishLine", {
        rank: ordinal(season.leaguePhaseRank, locale),
        champion: championLabel,
      }),
      SIZE / 2,
      400,
    );

    // Three figures: what you built, in one row.
    const boxes: [string, number][] = [
      [t("sheetOverall"), overall],
      [t("sheetAttack"), attack],
      [t("sheetDefense"), defense],
    ];
    const bw = 268;
    const gap = 24;
    let bx = (SIZE - (bw * 3 + gap * 2)) / 2;
    for (const [labelText, value] of boxes) {
      ctx.fillStyle = "rgba(255,255,255,0.05)";
      ctx.beginPath();
      ctx.roundRect(bx, 450, bw, 128, 18);
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.09)";
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.letterSpacing = "3px";
      ctx.font = font(700, 20);
      ctx.fillStyle = "#6b7385";
      ctx.fillText(labelText.toUpperCase(), bx + bw / 2, 492);
      ctx.letterSpacing = "0px";

      ctx.font = font(700, 54);
      ctx.fillStyle = "#f2f4f8";
      ctx.fillText(String(value), bx + bw / 2, 552);
      bx += bw + gap;
    }

    ctx.letterSpacing = "4px";
    ctx.font = font(700, 22);
    ctx.fillStyle = "#6b7385";
    ctx.fillText(t("sheetTop").toUpperCase(), SIZE / 2, 648);
    ctx.letterSpacing = "0px";

    const cw = 232;
    const ch = 300;
    const cgap = 30;
    let cx = (SIZE - (cw * 3 + cgap * 2)) / 2;
    for (const p of top) {
      const rounded = Math.round(p.overall);
      const [c1, c2, c3, ink] = TIER_COLORS[tierOf(rounded)] ?? TIER_COLORS.bronze;

      const grad = ctx.createLinearGradient(cx, 690, cx + cw, 690 + ch);
      grad.addColorStop(0, c1);
      grad.addColorStop(0.58, c2);
      grad.addColorStop(1, c3);

      shield(ctx, cx, 690, cw, ch);
      ctx.fillStyle = grad;
      ctx.fill();

      ctx.save();
      shield(ctx, cx, 690, cw, ch);
      ctx.clip();
      const sheen = ctx.createLinearGradient(cx, 690, cx + cw, 690 + ch * 0.5);
      sheen.addColorStop(0, "rgba(255,255,255,0.4)");
      sheen.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = sheen;
      ctx.fillRect(cx, 690, cw, ch);
      ctx.restore();

      ctx.fillStyle = ink;
      ctx.textAlign = "left";
      ctx.font = font(700, 62);
      ctx.fillText(String(rounded), cx + 22, 772);

      ctx.font = font(700, 26);
      ctx.fillText(fitText(ctx, p.display, cw - 44), cx + 22, 862);

      ctx.globalAlpha = 0.72;
      ctx.font = font(500, 21);
      ctx.fillText(fitText(ctx, p.team, cw - 44), cx + 22, 892);
      ctx.globalAlpha = 1;

      ctx.textAlign = "center";
      cx += cw + cgap;
    }

    if (scorer) {
      ctx.font = font(500, 30);
      ctx.fillStyle = "#99a1b3";
      ctx.fillText(
        t("summaryScorer", { player: scorer.player, goals: scorer.goals }),
        SIZE / 2,
        1006,
      );
    }

    ctx.letterSpacing = "5px";
    ctx.font = font(600, 20);
    ctx.fillStyle = "#4c5262";
    ctx.fillText("MUNDIALYTICS.COM", SIZE / 2, 1050);
    ctx.letterSpacing = "0px";
  }, [
    t,
    locale,
    squadLabel,
    competitionName,
    championLabel,
    season.stage,
    season.leaguePhaseRank,
    overall,
    attack,
    defense,
    top,
    scorer,
  ]);

  useEffect(() => {
    // Wait for the webfont, or the first paint uses the fallback and the image
    // does not look like the app.
    let cancelled = false;
    document.fonts?.ready.then(() => {
      if (!cancelled) draw();
    });
    draw();
    return () => {
      cancelled = true;
    };
  }, [draw]);

  const toBlob = () =>
    new Promise<Blob | null>((resolve) =>
      canvasRef.current?.toBlob(resolve, "image/png"),
    );

  async function download() {
    const blob = await toBlob();
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "mundialytics-squadlab.png";
    a.click();
    URL.revokeObjectURL(url);
  }

  async function share() {
    const blob = await toBlob();
    if (!blob) return;
    const file = new File([blob], "mundialytics-squadlab.png", {
      type: "image/png",
    });
    if (navigator.canShare?.({ files: [file] })) {
      await navigator.share({ files: [file], text: summary }).catch(() => undefined);
    }
  }

  return (
    <div className="mt-8 overflow-hidden rounded-[18px] border border-brand bg-surface">
      <div className="flex items-center justify-between gap-3 border-b border-border px-5 py-3">
        <p className="text-[0.62rem] font-semibold uppercase tracking-[0.13em] text-brand">
          {t("shareTitle")}
        </p>
        <button
          type="button"
          onClick={onClose}
          aria-label={t("close")}
          className="flex size-7 items-center justify-center rounded-lg border border-border text-muted transition-colors hover:text-text"
        >
          <X className="size-3.5" />
        </button>
      </div>

      <div className="px-5 py-6">
        <canvas
          ref={canvasRef}
          width={SIZE}
          height={SIZE}
          className="mx-auto block w-full max-w-[26rem] rounded-[14px] border border-border"
        />

        <div className="mt-5 flex flex-wrap justify-center gap-2">
          <button
            type="button"
            onClick={download}
            className="inline-flex h-10 items-center gap-2 rounded-[10px] bg-brand px-4 text-[0.85rem] font-medium text-brand-contrast transition-colors hover:bg-brand-hover"
          >
            <Download className="size-3.5" />
            {t("downloadImage")}
          </button>
          {canShare ? (
            <button
              type="button"
              onClick={share}
              className="inline-flex h-10 items-center gap-2 rounded-[10px] border border-border-strong px-4 text-[0.85rem] font-medium transition-colors hover:border-brand hover:text-brand"
            >
              <Share2 className="size-3.5" />
              {t("shareImage")}
            </button>
          ) : null}
          <button
            type="button"
            onClick={() => {
              navigator.clipboard?.writeText(summary);
              setCopied(true);
              setTimeout(() => setCopied(false), 2000);
            }}
            className="inline-flex h-10 items-center gap-2 rounded-[10px] border border-border px-4 text-[0.85rem] text-muted transition-colors hover:border-border-strong hover:text-text"
          >
            <Copy className="size-3.5" />
            {copied ? t("copied") : t("copySummary")}
          </button>
        </div>
      </div>
    </div>
  );
}
