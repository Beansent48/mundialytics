"use client";

import {
  ChevronRight,
  RotateCcw,
  Share2,
  Trophy,
} from "lucide-react";
import { useLocale, useTranslations } from "next-intl";
import { useCallback, useMemo, useState } from "react";

import { LiveMatch } from "@/components/squadlab/live-match";
import { ShareCard } from "@/components/squadlab/share-card";
import type { GamificationResult } from "@/lib/squadlab-runs";
import type {
  LeagueSeason,
  LeagueStandingsRow,
  SquadPlayer,
} from "@/lib/api";
import { cn, ordinal } from "@/lib/utils";

export function LeagueSeasonView({
  season,
  squad,
  revealed,
  setRevealed,
  onReset,
  gamification,
}: {
  season: LeagueSeason;
  squad: SquadPlayer[];
  revealed: number;
  setRevealed: (n: number) => void;
  onReset: () => void;
  gamification: GamificationResult;
}) {
  const t = useTranslations("squadlab");
  const locale = useLocale();
  const [sharing, setSharing] = useState(false);
  const [running, setRunning] = useState(false);
  const [showMatch, setShowMatch] = useState(false);
  const stopRunning = useCallback(() => setRunning(false), []);

  const matches = season.matches;
  const total = matches.length;
  const done = revealed >= total && !running;
  const settled = running ? revealed - 1 : revealed;
  const shown = matches.slice(0, settled);
  const current = revealed > 0 ? matches[revealed - 1] : null;

  const squadLabel = t("sheetTitle");
  const label = (team: string) => (team === season.teamName ? squadLabel : team);

  const progressiveStandings = useMemo(() => {
    if (settled === 0) return [];
    const lastMd = parseInt(matches[settled - 1]?.stage?.replace("J", "") ?? "0", 10);
    const played = season.allResults.filter((r) => r.matchday <= lastMd);

    type Rec = { team: string; played: number; won: number; drawn: number; lost: number;
                 gf: number; ga: number; pts: number; isSquad: boolean };
    const map = new Map<string, Rec>();
    for (const row of season.standings) {
      map.set(row.team, {
        team: row.team, played: 0, won: 0, drawn: 0, lost: 0,
        gf: 0, ga: 0, pts: 0, isSquad: row.isSquad,
      });
    }

    for (const r of played) {
      for (const [team, gf, ga] of [
        [r.home, r.homeGoals, r.awayGoals],
        [r.away, r.awayGoals, r.homeGoals],
      ] as [string, number, number][]) {
        const rec = map.get(team);
        if (!rec) continue;
        rec.played += 1;
        rec.gf += gf;
        rec.ga += ga;
        if (gf > ga) { rec.won += 1; rec.pts += 3; }
        else if (gf === ga) { rec.drawn += 1; rec.pts += 1; }
        else { rec.lost += 1; }
      }
    }

    return [...map.values()].sort((a, b) =>
      b.pts - a.pts || (b.gf - b.ga) - (a.gf - a.ga) || b.gf - a.gf,
    );
  }, [season, settled, matches]);

  const tableRows: LeagueStandingsRow[] = done
    ? season.standings
    : progressiveStandings.map((r, i) => ({
        rank: i + 1,
        team: r.team,
        played: r.played,
        won: r.won,
        drawn: r.drawn,
        lost: r.lost,
        goalsFor: r.gf,
        goalsAgainst: r.ga,
        goalDiff: r.gf - r.ga,
        points: r.pts,
        isSquad: r.isSquad,
      }));

  const squadRow = done
    ? season.standings.find((r) => r.isSquad)
    : null;

  return (
    <div>
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={onReset}
          className="inline-flex h-10 items-center gap-2 rounded-[10px] border border-border px-4 text-[0.85rem] text-muted transition-colors hover:border-border-strong hover:text-text"
        >
          <RotateCcw className="size-3.5" />
          {t("newTeam")}
        </button>
        <span className="text-[0.8rem] text-dim">
          {season.competitionName} · {t("replaced", { team: season.replaced })}
        </span>
      </div>

      {done ? (
        <div className="mt-8 overflow-hidden rounded-[18px] border border-border bg-surface">
          <div className="relative px-7 py-9 text-center">
            <div
              aria-hidden
              className="pointer-events-none absolute left-1/2 top-0 size-[22rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-brand opacity-[0.16] blur-[110px]"
            />
            <Trophy className="relative mx-auto size-7 text-warning" />
            <p className="relative mt-4 text-[0.66rem] font-semibold uppercase tracking-[0.14em] text-dim">
              {t("leagueFinish")}
            </p>
            <p className="font-display relative mt-2 text-[2rem] leading-tight sm:text-[2.6rem]">
              {season.stage}
            </p>
            <p className="relative mt-3 text-[0.95rem] text-muted">
              {t("leagueFinishLine", {
                position: ordinal(season.finalPosition, locale),
                points: squadRow?.points ?? 0,
              })}
              {" · "}
              {t("leagueChampion", { team: label(season.champion) })}
            </p>

            <button
              type="button"
              onClick={() => setSharing((s) => !s)}
              className="relative mt-7 inline-flex h-11 items-center gap-2 rounded-[10px] bg-brand px-5 text-[0.88rem] font-medium text-brand-contrast transition-colors hover:bg-brand-hover"
            >
              <Share2 className="size-4" />
              {t("shareSeason")}
            </button>
          </div>
        </div>
      ) : null}

      {done && gamification ? (
        <div className="mt-5 overflow-hidden rounded-[14px] border border-brand/30 bg-brand-ghost">
          <div className="px-5 py-4">
            <div className="flex items-center gap-3">
              <div className="flex size-10 items-center justify-center rounded-full bg-brand/15">
                <Trophy className="size-5 text-brand" />
              </div>
              <div>
                <p className="text-[0.88rem] font-semibold">
                  +{gamification.xpEarned} XP
                  {gamification.levelUp ? (
                    <span className="ml-2 rounded-full bg-brand px-2 py-0.5 text-[0.68rem] font-semibold text-brand-contrast">
                      Lv. {gamification.newLevel}!
                    </span>
                  ) : null}
                </p>
                {gamification.newAchievements.length > 0 ? (
                  <p className="mt-0.5 text-[0.78rem] text-muted">
                    {gamification.newAchievements.length === 1
                      ? t("achievementUnlocked")
                      : t("achievementsUnlocked", { n: gamification.newAchievements.length })}
                  </p>
                ) : null}
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {done && sharing ? (
        <ShareCard
          squad={squad}
          season={season as any}
          squadLabel={squadLabel}
          onClose={() => setSharing(false)}
        />
      ) : null}

      {showMatch && current ? (
        <div className="mt-8">
          <LiveMatch
            key={revealed}
            fixture={current}
            label={current.stageLabel}
            teamName={season.teamName}
            squadLabel={squadLabel}
            running={running}
            onFinish={stopRunning}
          />
        </div>
      ) : null}

      <div className="mt-8 flex flex-wrap items-center gap-3">
        <button
          type="button"
          disabled={done || running}
          onClick={() => {
            setRevealed(revealed + 1);
            setRunning(true);
            setShowMatch(true);
          }}
          className="inline-flex h-11 items-center gap-2 rounded-[10px] bg-brand px-5 text-[0.9rem] font-medium text-brand-contrast transition-colors hover:bg-brand-hover disabled:pointer-events-none disabled:opacity-40"
        >
          {revealed === 0 ? t("playLeague") : t("nextMatchday")}
          <ChevronRight className="size-4" />
        </button>
        <button
          type="button"
          disabled={done}
          onClick={() => {
            setRevealed(total);
            setRunning(false);
            setShowMatch(false);
          }}
          className="inline-flex h-11 items-center rounded-[10px] border border-border px-4 text-[0.85rem] text-muted transition-colors hover:border-border-strong hover:text-text disabled:pointer-events-none disabled:opacity-40"
        >
          {t("skipToEnd")}
        </button>
        <span className="text-[0.8rem] text-dim">
          {t("matchdayOf", { n: revealed, total })}
        </span>
      </div>

      <div className="mt-8 flex flex-col gap-10">
        {/* ── League table ────────────────────────────────── */}
        <section>
          <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
            {done ? t("finalTable") : settled > 0 ? t("tableSoFar") : t("finalTable")}
          </h2>
          {tableRows.length > 0 ? (
            <div className="mt-3 overflow-x-auto rounded-[var(--radius-card)] border border-border">
              <table className="w-full min-w-[30rem] text-[0.82rem]">
                <thead>
                  <tr className="border-b border-border bg-bg-elevated text-[0.6rem] uppercase tracking-[0.07em] text-dim">
                    <th className="px-3 py-2 text-left font-semibold">#</th>
                    <th className="px-3 py-2 text-left font-semibold">{t("colTeam")}</th>
                    <th className="px-2 py-2 text-right font-semibold">{t("colPlayed")}</th>
                    <th className="px-2 py-2 text-right font-semibold">{t("colWon")}</th>
                    <th className="px-2 py-2 text-right font-semibold">{t("colDrawn")}</th>
                    <th className="px-2 py-2 text-right font-semibold">{t("colLost")}</th>
                    <th className="px-2 py-2 text-right font-semibold">{t("colFor")}</th>
                    <th className="px-2 py-2 text-right font-semibold">{t("colAgainst")}</th>
                    <th className="px-2 py-2 text-right font-semibold">{t("colDiff")}</th>
                    <th className="px-3 py-2 text-right font-semibold">{t("colPoints")}</th>
                  </tr>
                </thead>
                <tbody>
                  {tableRows.map((r) => (
                    <tr
                      key={r.team}
                      className={cn(
                        "border-b border-border last:border-0",
                        r.rank === 5 || r.rank === 7 || r.rank === 18
                          ? "border-t-2 border-t-border-strong"
                          : "",
                        r.isSquad ? "bg-brand-ghost" : "bg-surface",
                      )}
                    >
                      <td className="px-3 py-2 tabular-nums text-dim">{r.rank}</td>
                      <td className={cn("px-3 py-2", r.isSquad ? "font-semibold text-text" : "")}>
                        {label(r.team)}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted">{r.played}</td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted">{r.won}</td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted">{r.drawn}</td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted">{r.lost}</td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted">{r.goalsFor}</td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted">{r.goalsAgainst}</td>
                      <td
                        className={cn(
                          "px-2 py-2 text-right tabular-nums",
                          r.goalDiff > 0
                            ? "text-positive"
                            : r.goalDiff < 0
                              ? "text-negative"
                              : "text-dim",
                        )}
                      >
                        {r.goalDiff > 0 ? `+${r.goalDiff}` : r.goalDiff}
                      </td>
                      <td className="px-3 py-2 text-right font-semibold tabular-nums">
                        {r.points}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="mt-3 text-[0.83rem] text-muted">{t("notStarted")}</p>
          )}
        </section>

        {/* ── Your results ────────────────────────────────── */}
        <section>
          <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
            {t("yourResults")}
          </h2>
          <div className="mt-3 flex flex-col gap-1.5">
            {shown
              .slice()
              .reverse()
              .map((f, i) => {
                const home = f.home === season.teamName;
                const us = home ? f.homeGoals : f.awayGoals;
                const them = home ? f.awayGoals : f.homeGoals;
                return (
                  <div
                    key={`${f.stage}-${i}`}
                    className="flex items-center gap-3 rounded-[11px] border border-border bg-surface px-4 py-2.5"
                  >
                    <span className="w-16 shrink-0 truncate text-[0.66rem] uppercase tracking-[0.06em] text-dim">
                      {f.stageLabel}
                    </span>
                    <span className={cn("w-8 text-right text-[0.78rem] tabular-nums", home ? "font-medium" : "text-muted")}>
                      {f.homeGoals}
                    </span>
                    <span className="text-[0.72rem] text-dim">–</span>
                    <span className={cn("w-8 text-[0.78rem] tabular-nums", !home ? "font-medium" : "text-muted")}>
                      {f.awayGoals}
                    </span>
                    <span className="flex-1 truncate text-[0.85rem]">
                      {label(home ? f.away : f.home)}
                    </span>
                    <span className="text-[0.72rem] text-dim">
                      {home ? t("atHome") : t("away")}
                    </span>
                    <span
                      className={cn(
                        "w-5 text-center text-[0.78rem] font-semibold",
                        us > them ? "text-positive" : us < them ? "text-negative" : "text-dim",
                      )}
                    >
                      {us > them ? "W" : us < them ? "L" : "D"}
                    </span>
                  </div>
                );
              })}
            {shown.length === 0 ? (
              <p className="text-[0.83rem] text-muted">{t("notStarted")}</p>
            ) : null}
          </div>
        </section>

        {/* ── Top scorers ─────────────────────────────────── */}
        {done && season.scorers.length > 0 ? (
          <section>
            <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
              {t("topScorers")}
            </h2>
            <div className="mt-3 overflow-x-auto rounded-[var(--radius-card)] border border-border">
              <table className="w-full min-w-[22rem] text-[0.82rem]">
                <thead>
                  <tr className="border-b border-border bg-bg-elevated text-[0.6rem] uppercase tracking-[0.07em] text-dim">
                    <th className="px-3 py-2 text-left font-semibold">{t("colPlayer")}</th>
                    <th className="px-3 py-2 text-left font-semibold">{t("colTeam")}</th>
                    <th className="px-2 py-2 text-right font-semibold">{t("colGoals")}</th>
                    <th className="px-2 py-2 text-right font-semibold">{t("colAssists")}</th>
                  </tr>
                </thead>
                <tbody>
                  {season.scorers.map((s, i) => (
                    <tr
                      key={`${s.player}-${i}`}
                      className={cn(
                        "border-b border-border last:border-0",
                        s.isSquad ? "bg-brand-ghost" : "bg-surface",
                      )}
                    >
                      <td className={cn("px-3 py-2", s.isSquad ? "font-semibold" : "")}>
                        {s.player}
                      </td>
                      <td className="px-3 py-2 text-muted">{label(s.team)}</td>
                      <td className="px-2 py-2 text-right tabular-nums font-semibold">
                        {s.goals}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted">
                        {s.assists}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ) : null}

        {/* ── Squad totals ────────────────────────────────── */}
        {done && season.squadTotals && season.squadTotals.length > 0 ? (
          <section>
            <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
              {t("squadTotals")}
            </h2>
            <div className="mt-3 overflow-x-auto rounded-[var(--radius-card)] border border-border">
              <table className="w-full min-w-[28rem] text-[0.82rem]">
                <thead>
                  <tr className="border-b border-border bg-bg-elevated text-[0.6rem] uppercase tracking-[0.07em] text-dim">
                    <th className="px-3 py-2 text-left font-semibold">{t("colPlayer")}</th>
                    <th className="px-2 py-2 text-right font-semibold">{t("colPlayed")}</th>
                    <th className="px-2 py-2 text-right font-semibold">{t("colMinutes")}</th>
                    <th className="px-2 py-2 text-right font-semibold">{t("colGoals")}</th>
                    <th className="px-2 py-2 text-right font-semibold">{t("colAssists")}</th>
                    <th className="px-2 py-2 text-right font-semibold">{t("colCards")}</th>
                    <th className="px-3 py-2 text-right font-semibold">{t("colRating")}</th>
                  </tr>
                </thead>
                <tbody>
                  {season.squadTotals.map((p) => (
                    <tr
                      key={p.player}
                      className="border-b border-border bg-surface last:border-0"
                    >
                      <td className="px-3 py-2 font-medium">{p.player}</td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted">{p.apps}</td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted">{p.minutes}</td>
                      <td className="px-2 py-2 text-right tabular-nums font-semibold">
                        {p.goals || "–"}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted">
                        {p.assists || "–"}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted">
                        {p.yellows || "–"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums font-medium">
                        {p.rating != null ? p.rating.toFixed(1) : "–"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ) : null}
      </div>
    </div>
  );
}
