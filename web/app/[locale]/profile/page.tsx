"use client";

import {
  BookOpen,
  Crown,
  Flag,
  Flame,
  Gem,
  Globe,
  Infinity,
  LayoutGrid,
  Library,
  Lock,
  Medal,
  Repeat,
  RotateCcw,
  Shield,
  ShieldCheck,
  Siren,
  Sparkles,
  Star,
  Swords,
  Target,
  Trophy,
  Zap,
} from "lucide-react";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { Link } from "@/i18n/navigation";
import { useSession } from "@/lib/auth-client";
import { cn } from "@/lib/utils";

type ProfileData = {
  progress: {
    xp: number;
    level: number;
    xpInLevel: number;
    xpNeeded: number;
    title: string;
    titleKey: string;
    formations: string[];
    maxRerolls: number;
  };
  achievements: {
    code: string;
    name: string;
    description: string;
    points: number;
    category: string;
    icon: string;
    hidden: number;
    awardedAt: string | null;
  }[];
  stats: {
    totalRuns: number;
    wins: number;
    winRate: number;
    bestStage: string | null;
    bestSquadRating: number | null;
    totalGoals: number;
    bestMatchGoals: number;
  };
  album: {
    total: number;
    actual: number;
    prime: number;
    icono: number;
  };
};

const ICON_MAP: Record<string, React.ElementType> = {
  trophy: Trophy,
  medal: Medal,
  target: Target,
  swords: Swords,
  "shield-check": ShieldCheck,
  flame: Flame,
  zap: Zap,
  "chevrons-up": Zap,
  shield: Shield,
  "rotate-ccw": RotateCcw,
  star: Star,
  sparkles: Sparkles,
  crown: Crown,
  siren: Siren,
  flag: Flag,
  repeat: Repeat,
  infinity: Infinity,
  gem: Gem,
  library: Library,
  "layout-grid": LayoutGrid,
  globe: Globe,
};

function AchievementIcon({ name, className }: { name: string; className?: string }) {
  const Icon = ICON_MAP[name] ?? Trophy;
  return <Icon className={className} />;
}

export default function ProfilePage() {
  const t = useTranslations("profile");
  const { data: session } = useSession();
  const [data, setData] = useState<ProfileData | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"achievements" | "album">("achievements");

  useEffect(() => {
    if (!session) {
      Promise.resolve().then(() => setLoading(false));
      return;
    }
    let cancelled = false;
    fetch("/api/user/profile", { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : null))
      .then((json) => { if (!cancelled && json) setData(json); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [session]);

  if (!session) {
    return (
      <div className="mx-auto max-w-sm px-5 py-20 text-center sm:px-7 sm:py-28">
        <Trophy className="mx-auto size-8 text-warning" />
        <h1 className="font-display mt-4 text-[1.7rem] leading-tight">
          {t("signedOutTitle")}
        </h1>
        <p className="mt-3 text-[0.9rem] leading-relaxed text-muted">
          {t("signedOutBody")}
        </p>
        <Link
          href="/sign-in"
          className="mt-7 inline-flex h-11 items-center rounded-[10px] bg-brand px-6 text-[0.9rem] font-medium text-brand-contrast transition-colors hover:bg-brand-hover"
        >
          {t("signIn")}
        </Link>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="mx-auto max-w-[780px] px-5 py-14 sm:px-7 sm:py-20">
        <div className="animate-pulse space-y-6">
          <div className="h-8 w-48 rounded bg-surface-2" />
          <div className="h-4 w-64 rounded bg-surface-2" />
          <div className="grid grid-cols-3 gap-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-24 rounded-[12px] bg-surface-2" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const { progress, achievements, stats, album } = data;
  const pct = progress.xpNeeded > 0 ? (progress.xpInLevel / progress.xpNeeded) * 100 : 100;
  const categories = ["tournament", "match", "squad", "collection"] as const;

  return (
    <div className="mx-auto max-w-[780px] px-5 py-14 sm:px-7 sm:py-20">
      {/* ── Level + XP header ──────────────────────────────────────────── */}
      <div className="flex items-center gap-4">
        <div className="flex size-16 items-center justify-center rounded-full border-2 border-brand bg-brand-ghost">
          <span className="text-[1.4rem] font-bold text-brand">{progress.level}</span>
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-2">
            <h1 className="font-display text-[1.7rem] leading-tight">
              {session.user.name}
            </h1>
            <span className="rounded-full bg-brand-ghost px-2.5 py-0.5 text-[0.72rem] font-semibold text-brand">
              {t(`titles.${progress.titleKey}`)}
            </span>
          </div>
          <div className="mt-2 flex items-center gap-3">
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
              <div
                className="h-full rounded-full bg-brand transition-all duration-500"
                style={{ width: `${Math.min(pct, 100)}%` }}
              />
            </div>
            <span className="shrink-0 text-[0.72rem] tabular-nums text-dim">
              {progress.xpInLevel} / {progress.xpNeeded} XP
            </span>
          </div>
        </div>
      </div>

      {/* ── Stats grid ─────────────────────────────────────────────────── */}
      <div className="mt-9 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { label: t("statRuns"), value: stats.totalRuns },
          { label: t("statWins"), value: stats.wins },
          { label: t("statWinRate"), value: `${stats.winRate}%` },
          { label: t("statGoals"), value: stats.totalGoals },
          { label: t("statBestStage"), value: stats.bestStage ?? "—" },
          { label: t("statBestRating"), value: stats.bestSquadRating?.toFixed(1) ?? "—" },
          { label: t("statBestGoals"), value: stats.bestMatchGoals },
          { label: t("statLevel"), value: `Lv. ${progress.level}` },
        ].map(({ label, value }) => (
          <div
            key={label}
            className="rounded-[12px] border border-border bg-surface px-4 py-3.5"
          >
            <p className="text-[0.6rem] font-semibold uppercase tracking-[0.1em] text-dim">
              {label}
            </p>
            <p className="mt-1.5 text-[1.1rem] font-semibold tabular-nums">{value}</p>
          </div>
        ))}
      </div>

      {/* ── Album summary ──────────────────────────────────────────────── */}
      <div className="mt-9 rounded-[14px] border border-border bg-surface px-5 py-5">
        <div className="flex items-center gap-2">
          <BookOpen className="size-4 text-brand" />
          <h2 className="text-[0.88rem] font-semibold">{t("album")}</h2>
          <span className="ml-auto text-[0.82rem] tabular-nums text-muted">
            {album.total} {t("albumCards")}
          </span>
        </div>
        <div className="mt-4 grid grid-cols-3 gap-px overflow-hidden rounded-[10px] border border-border bg-border">
          {[
            { label: "Actual", count: album.actual, cls: "" },
            { label: "Prime", count: album.prime, cls: "text-teal-400" },
            { label: "Icono", count: album.icono, cls: "text-amber-300" },
          ].map(({ label, count, cls }) => (
            <div key={label} className="bg-bg-elevated px-3 py-3 text-center">
              <p className={cn("text-[0.65rem] font-semibold uppercase tracking-[0.08em]", cls || "text-dim")}>
                {label}
              </p>
              <p className="mt-1 text-[1.1rem] font-semibold tabular-nums">{count}</p>
            </div>
          ))}
        </div>
      </div>

      {/* ── Tabs: achievements / album ─────────────────────────────────── */}
      <div className="mt-9 flex gap-1 border-b border-border">
        {(["achievements", "album"] as const).map((key) => (
          <button
            key={key}
            type="button"
            onClick={() => setTab(key)}
            className={cn(
              "relative px-4 py-2.5 text-[0.85rem] font-medium transition-colors",
              tab === key ? "text-text" : "text-muted hover:text-text",
            )}
          >
            {t(`tab.${key}`)}
            {tab === key ? (
              <span className="absolute inset-x-4 bottom-0 h-0.5 rounded-full bg-brand" />
            ) : null}
          </button>
        ))}
      </div>

      {tab === "achievements" ? (
        <div className="mt-6 space-y-8">
          {categories.map((cat) => {
            const inCat = achievements.filter((a) => a.category === cat);
            if (!inCat.length) return null;
            return (
              <section key={cat}>
                <h3 className="text-[0.66rem] font-semibold uppercase tracking-[0.12em] text-dim">
                  {t(`category.${cat}`)}
                </h3>
                <div className="mt-3 grid gap-2 sm:grid-cols-2">
                  {inCat.map((a) => {
                    const earned = Boolean(a.awardedAt);
                    const isHidden = a.hidden && !earned;
                    return (
                      <div
                        key={a.code}
                        className={cn(
                          "flex items-start gap-3 rounded-[12px] border px-4 py-3.5 transition-colors",
                          earned
                            ? "border-brand/30 bg-brand-ghost"
                            : "border-border bg-surface opacity-60",
                        )}
                      >
                        <div
                          className={cn(
                            "flex size-9 shrink-0 items-center justify-center rounded-full",
                            earned ? "bg-brand/15 text-brand" : "bg-surface-2 text-dim",
                          )}
                        >
                          {isHidden ? (
                            <Lock className="size-4" />
                          ) : (
                            <AchievementIcon name={a.icon} className="size-4" />
                          )}
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="text-[0.84rem] font-semibold leading-tight">
                            {isHidden ? "???" : a.name}
                          </p>
                          <p className="mt-0.5 text-[0.76rem] leading-relaxed text-muted">
                            {isHidden ? t("hiddenHint") : a.description}
                          </p>
                        </div>
                        <span className="shrink-0 text-[0.7rem] font-semibold tabular-nums text-dim">
                          {a.points} XP
                        </span>
                      </div>
                    );
                  })}
                </div>
              </section>
            );
          })}
        </div>
      ) : (
        <AlbumView />
      )}

      {/* ── Unlocks ────────────────────────────────────────────────────── */}
      <section className="mt-10">
        <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.12em] text-dim">
          {t("levelRewards")}
        </h2>
        <div className="mt-3 flex flex-col gap-1.5">
          {[
            { level: 1, label: t("titles.novice"), detail: "4-3-3, 4-4-2, 3-5-2 · 3 re-rolls" },
            { level: 3, label: t("titles.aspirant"), detail: "+ 3-4-3" },
            { level: 5, label: t("titles.competitor"), detail: "+ 5-3-2" },
            { level: 7, label: t("titles.strategist"), detail: "+ 5-2-3" },
            { level: 10, label: t("titles.veteran"), detail: "+ 4-5-1 · 4 re-rolls" },
            { level: 15, label: t("titles.legend"), detail: "5 re-rolls" },
            { level: 20, label: t("titles.master"), detail: "" },
          ].map(({ level, label, detail }) => (
            <div
              key={level}
              className={cn(
                "flex items-center gap-3 rounded-[11px] border px-4 py-2.5",
                progress.level >= level
                  ? "border-brand/30 bg-brand-ghost"
                  : "border-border bg-surface opacity-50",
              )}
            >
              <span className="w-10 shrink-0 text-center text-[0.75rem] font-bold tabular-nums text-brand">
                Lv.{level}
              </span>
              <span className="flex-1 text-[0.84rem] font-medium">{label}</span>
              {detail ? (
                <span className="text-[0.72rem] text-dim">{detail}</span>
              ) : null}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function AlbumView() {
  const t = useTranslations("profile");
  const [cards, setCards] = useState<
    { cardId: string; kind: string; display: string; position: string; overall: number; count: number }[]
  >([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/user/album", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : []))
      .then(setCards)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="mt-6 grid grid-cols-3 gap-2 sm:grid-cols-5">
        {Array.from({ length: 10 }).map((_, i) => (
          <div key={i} className="h-24 animate-pulse rounded-[10px] bg-surface-2" />
        ))}
      </div>
    );
  }

  if (!cards.length) {
    return (
      <p className="mt-6 rounded-[12px] border border-border bg-surface px-5 py-10 text-center text-[0.85rem] text-muted">
        {t("albumEmpty")}
      </p>
    );
  }

  const kindColor: Record<string, string> = {
    actual: "border-border",
    prime: "border-teal-400/40",
    icono: "border-amber-300/40 bg-amber-300/5",
  };

  return (
    <div className="mt-6 grid grid-cols-3 gap-2 sm:grid-cols-5 lg:grid-cols-6">
      {cards.map((card) => (
        <div
          key={card.cardId}
          className={cn(
            "rounded-[10px] border bg-surface px-3 py-2.5 text-center",
            kindColor[card.kind] ?? "border-border",
          )}
        >
          <p className="truncate text-[0.78rem] font-semibold leading-tight">
            {card.display}
          </p>
          <p className="mt-0.5 text-[0.65rem] text-dim">{card.position}</p>
          <p className="mt-1 text-[1rem] font-bold tabular-nums">{card.overall}</p>
          {card.count > 1 ? (
            <p className="mt-0.5 text-[0.6rem] text-dim">x{card.count}</p>
          ) : null}
        </div>
      ))}
    </div>
  );
}
