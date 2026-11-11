import { and, desc, eq, sql } from "drizzle-orm";

import {
  achievement,
  squadlabRun,
  userAchievement,
  userCard,
  userProgress,
} from "@/lib/db/app-schema";
import { getDb } from "@/lib/db/client";
import type { SquadSeason } from "@/lib/api";
import {
  detectAchievements,
  levelFromXp,
  rewardsForLevel,
  xpForRun,
  xpProgress,
  ACHIEVEMENT_CATALOGUE,
} from "@/lib/gamification";

// ── Progress ────────────────────────────────────────────────────────────────

export type UserProgressData = {
  xp: number;
  level: number;
  xpInLevel: number;
  xpNeeded: number;
  title: string;
  titleKey: string;
  formations: string[];
  maxRerolls: number;
};

export async function readProgress(userId: string): Promise<UserProgressData> {
  const [row] = await getDb()
    .select({ xp: userProgress.xp, level: userProgress.level })
    .from(userProgress)
    .where(eq(userProgress.userId, userId))
    .limit(1);

  const xp = row?.xp ?? 0;
  const level = row?.level ?? 1;
  const [xpInLevel, xpNeeded] = xpProgress(xp);
  const rewards = rewardsForLevel(level);

  return { xp, level, xpInLevel, xpNeeded, ...rewards };
}

async function ensureProgress(userId: string): Promise<void> {
  await getDb()
    .insert(userProgress)
    .values({ userId, xp: 0, level: 1 })
    .onConflictDoNothing();
}

async function addXp(userId: string, amount: number): Promise<{ xp: number; level: number }> {
  await ensureProgress(userId);
  const [row] = await getDb()
    .update(userProgress)
    .set({
      xp: sql`${userProgress.xp} + ${amount}`,
    })
    .where(eq(userProgress.userId, userId))
    .returning({ xp: userProgress.xp });

  const newXp = row?.xp ?? amount;
  const newLevel = levelFromXp(newXp);

  await getDb()
    .update(userProgress)
    .set({ level: newLevel })
    .where(eq(userProgress.userId, userId));

  return { xp: newXp, level: newLevel };
}

// ── Achievements ────────────────────────────────────────────────────────────

export type UserAchievementRow = {
  code: string;
  name: string;
  description: string;
  points: number;
  category: string;
  icon: string;
  hidden: number;
  awardedAt: string | null;
};

export async function readAchievements(userId: string): Promise<UserAchievementRow[]> {
  const rows = await getDb()
    .select({
      code: achievement.code,
      name: achievement.name,
      description: achievement.description,
      points: achievement.points,
      category: achievement.category,
      icon: achievement.icon,
      hidden: achievement.hidden,
      awardedAt: userAchievement.awardedAt,
    })
    .from(achievement)
    .leftJoin(
      userAchievement,
      and(
        eq(userAchievement.code, achievement.code),
        eq(userAchievement.userId, userId),
      ),
    )
    .orderBy(desc(achievement.points));

  return rows.map((r) => ({
    ...r,
    awardedAt: r.awardedAt?.toISOString() ?? null,
  }));
}

async function awardAchievements(
  userId: string,
  codes: string[],
): Promise<{ newCodes: string[]; xpEarned: number }> {
  if (!codes.length) return { newCodes: [], xpEarned: 0 };

  const existing = await getDb()
    .select({ code: userAchievement.code })
    .from(userAchievement)
    .where(
      and(
        eq(userAchievement.userId, userId),
        sql`${userAchievement.code} = any(${codes})`,
      ),
    );

  const owned = new Set(existing.map((r) => r.code));
  const newCodes = codes.filter((c) => !owned.has(c));
  if (!newCodes.length) return { newCodes: [], xpEarned: 0 };

  await getDb()
    .insert(userAchievement)
    .values(newCodes.map((code) => ({ userId, code })))
    .onConflictDoNothing();

  // Sum up XP from the catalogue
  const catalogue = new Map(ACHIEVEMENT_CATALOGUE.map((a) => [a.code, a.points]));
  const xpEarned = newCodes.reduce((sum, c) => sum + (catalogue.get(c) ?? 0), 0);

  return { newCodes, xpEarned };
}

// ── Album ───────────────────────────────────────────────────────────────────

export type AlbumCard = {
  cardId: string;
  kind: string;
  display: string;
  position: string;
  overall: number;
  count: number;
  firstDraftedAt: string;
};

export async function readAlbum(userId: string): Promise<AlbumCard[]> {
  const rows = await getDb()
    .select()
    .from(userCard)
    .where(eq(userCard.userId, userId))
    .orderBy(desc(userCard.overall));

  return rows.map((r) => ({
    cardId: r.cardId,
    kind: r.kind,
    display: r.display,
    position: r.position,
    overall: r.overall,
    count: r.count,
    firstDraftedAt: r.firstDraftedAt.toISOString(),
  }));
}

export async function albumStats(userId: string): Promise<{
  total: number;
  actual: number;
  prime: number;
  icono: number;
}> {
  const [row] = await getDb()
    .select({
      total: sql<number>`count(*)`,
      actual: sql<number>`count(*) filter (where ${userCard.kind} = 'actual')`,
      prime: sql<number>`count(*) filter (where ${userCard.kind} = 'prime')`,
      icono: sql<number>`count(*) filter (where ${userCard.kind} = 'icono')`,
    })
    .from(userCard)
    .where(eq(userCard.userId, userId));

  return {
    total: Number(row?.total ?? 0),
    actual: Number(row?.actual ?? 0),
    prime: Number(row?.prime ?? 0),
    icono: Number(row?.icono ?? 0),
  };
}

async function collectCards(
  userId: string,
  squad: Record<string, { player: string; kind: string; display: string; position: string; overall: number }>,
): Promise<void> {
  const cards = Object.values(squad);
  if (!cards.length) return;

  for (const card of cards) {
    await getDb()
      .insert(userCard)
      .values({
        userId,
        cardId: card.player,
        kind: card.kind,
        display: card.display,
        position: card.position,
        overall: Math.round(card.overall),
      })
      .onConflictDoUpdate({
        target: [userCard.userId, userCard.cardId],
        set: {
          count: sql`${userCard.count} + 1`,
          // Update display/overall if the card was rebuilt since last draft
          display: card.display,
          overall: Math.round(card.overall),
        },
      });
  }
}

// ── Profile stats (computed from runs) ──────────────────────────────────────

export type ProfileStats = {
  totalRuns: number;
  wins: number;
  winRate: number;
  bestStage: string | null;
  bestSquadRating: number | null;
  totalGoals: number;
  bestMatchGoals: number;
};

export async function readProfileStats(userId: string): Promise<ProfileStats> {
  const runs = await getDb()
    .select({
      status: squadlabRun.status,
      result: squadlabRun.result,
      squad: squadlabRun.squad,
    })
    .from(squadlabRun)
    .where(and(eq(squadlabRun.userId, userId), eq(squadlabRun.status, "finished")));

  let wins = 0;
  let bestStage: string | null = null;
  let bestSquadRating: number | null = null;
  let totalGoals = 0;
  let bestMatchGoals = 0;

  const stageRank: Record<string, number> = {
    "Eliminado en fase de liga": 1,
    "Eliminated in league phase": 1,
    "Eliminado en playoff": 2,
    "Eliminated in play-off": 2,
    "Eliminado en octavos": 3,
    "Eliminated in R16": 3,
    "Eliminado en cuartos": 4,
    "Eliminated in QF": 4,
    "Eliminado en semifinales": 5,
    "Eliminated in SF": 5,
    "Finalista": 6,
    "Runner-up": 6,
    "Campeón": 7,
    "Champion": 7,
  };

  let bestStageRank = 0;

  for (const run of runs) {
    const result = run.result as SquadSeason | null;
    if (!result) continue;

    const isWin = result.champion === result.teamName;
    if (isWin) wins++;

    const rank = stageRank[result.stage] ?? 0;
    if (rank > bestStageRank) {
      bestStageRank = rank;
      bestStage = result.stage;
    }

    const squad = run.squad as Record<string, { overall: number }>;
    const cards = Object.values(squad);
    if (cards.length) {
      const avg = cards.reduce((a, c) => a + (c.overall ?? 0), 0) / cards.length;
      if (bestSquadRating === null || avg > bestSquadRating) {
        bestSquadRating = Math.round(avg * 10) / 10;
      }
    }

    for (const m of result.matches ?? []) {
      const home = m.home === result.teamName;
      const us = home ? m.homeGoals : m.awayGoals;
      totalGoals += us;
      if (us > bestMatchGoals) bestMatchGoals = us;
    }
  }

  return {
    totalRuns: runs.length,
    wins,
    winRate: runs.length > 0 ? Math.round((wins / runs.length) * 100) : 0,
    bestStage,
    bestSquadRating,
    totalGoals,
    bestMatchGoals,
  };
}

// ── Post-run hook ───────────────────────────────────────────────────────────

export type PostRunResult = {
  xpEarned: number;
  newLevel: number;
  levelUp: boolean;
  newAchievements: string[];
  albumNew: number;
};

/**
 * Called after a run finishes. Awards XP, checks achievements, collects cards.
 * Returns the gamification outcome to show in the UI.
 */
export async function processFinishedRun(
  userId: string,
  result: SquadSeason,
  squad: Record<string, { player: string; kind: string; display: string; position: string; overall: number; league?: string }>,
): Promise<PostRunResult> {
  const oldProgress = await readProgress(userId);

  // 1. Collect cards into the album
  await collectCards(userId, squad);

  // 2. Load all finished runs for collection achievements
  const allRunRows = await getDb()
    .select({ result: squadlabRun.result, squad: squadlabRun.squad })
    .from(squadlabRun)
    .where(and(eq(squadlabRun.userId, userId), eq(squadlabRun.status, "finished")));

  const allRuns = allRunRows
    .filter((r) => r.result)
    .map((r) => ({
      result: r.result as SquadSeason,
      squad: r.squad as Record<string, { player: string; kind: string; overall: number; league?: string }>,
    }));

  // 3. Detect achievements
  const detected = detectAchievements(result, squad, allRuns);
  const { newCodes, xpEarned: achievementXp } = await awardAchievements(userId, detected);

  // 4. Award XP (run + achievements)
  const runXp = xpForRun(result.stage);
  const totalXp = runXp + achievementXp;
  const { level: newLevel } = await addXp(userId, totalXp);

  // 5. Count newly collected cards (cards with count=1 that were just inserted)
  const albumCards = await getDb()
    .select({ count: sql<number>`count(*)` })
    .from(userCard)
    .where(and(eq(userCard.userId, userId), eq(userCard.count, 1)));
  // This is approximate — it counts all cards with count=1, not just new ones

  return {
    xpEarned: totalXp,
    newLevel,
    levelUp: newLevel > oldProgress.level,
    newAchievements: newCodes,
    albumNew: 0, // The UI can show new card count from the response
  };
}

// ── Seed achievements ───────────────────────────────────────────────────────

export async function seedAchievements(locale: "es" | "en" = "es"): Promise<void> {
  const rows = ACHIEVEMENT_CATALOGUE.map((a) => ({
    code: a.code,
    name: locale === "es" ? a.nameEs : a.nameEn,
    description: locale === "es" ? a.descEs : a.descEn,
    points: a.points,
    category: a.category,
    icon: a.icon,
    hidden: a.hidden,
  }));

  for (const row of rows) {
    await getDb()
      .insert(achievement)
      .values(row)
      .onConflictDoUpdate({
        target: achievement.code,
        set: {
          name: row.name,
          description: row.description,
          points: row.points,
          category: row.category,
          icon: row.icon,
          hidden: row.hidden,
        },
      });
  }
}
