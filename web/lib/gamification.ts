import type { SquadSeason } from "@/lib/api";

// ── Levels ──────────────────────────────────────────────────────────────────

/** XP required to go from level N to N+1. */
function xpForLevel(n: number): number {
  return 300 * n;
}

/** Total XP needed to reach a given level (cumulative). */
export function totalXpForLevel(level: number): number {
  if (level <= 1) return 0;
  return (300 * (level - 1) * level) / 2;
}

/** The level a given amount of total XP corresponds to. */
export function levelFromXp(xp: number): number {
  if (xp <= 0) return 1;
  // Solve 300 * (L-1) * L / 2 <= xp => L^2 - L - 2xp/300 <= 0
  const L = (1 + Math.sqrt(1 + (8 * xp) / 300)) / 2;
  return Math.floor(L);
}

/** XP progress within the current level, as [current, needed]. */
export function xpProgress(xp: number): [number, number] {
  const level = levelFromXp(xp);
  const base = totalXpForLevel(level);
  const needed = xpForLevel(level);
  return [xp - base, needed];
}

// ── Level rewards ───────────────────────────────────────────────────────────

type LevelReward = {
  title: string;
  titleKey: string;
  formations?: string[];
  maxRerolls?: number;
};

const ALL_FORMATIONS = [
  "4-3-3",
  "4-4-2",
  "3-5-2",
  "3-4-3",
  "5-3-2",
  "5-2-3",
  "4-5-1",
];

const LEVEL_REWARDS: Record<number, LevelReward> = {
  1: { title: "Novato", titleKey: "novice", formations: ["4-3-3", "4-4-2", "3-5-2"] },
  3: { title: "Aspirante", titleKey: "aspirant", formations: ["4-3-3", "4-4-2", "3-5-2", "3-4-3"] },
  5: { title: "Competidor", titleKey: "competitor", formations: ["4-3-3", "4-4-2", "3-5-2", "3-4-3", "5-3-2"] },
  7: { title: "Estratega", titleKey: "strategist", formations: ["4-3-3", "4-4-2", "3-5-2", "3-4-3", "5-3-2", "5-2-3"] },
  10: { title: "Veterano", titleKey: "veteran", formations: ALL_FORMATIONS, maxRerolls: 4 },
  15: { title: "Leyenda", titleKey: "legend", maxRerolls: 5 },
  20: { title: "Maestro", titleKey: "master" },
};

export function rewardsForLevel(level: number): {
  title: string;
  titleKey: string;
  formations: string[];
  maxRerolls: number;
} {
  let title = "Novato";
  let titleKey = "novice";
  let formations = ["4-3-3", "4-4-2", "3-5-2"];
  let maxRerolls = 3;

  for (const [lvl, reward] of Object.entries(LEVEL_REWARDS)) {
    if (level >= Number(lvl)) {
      title = reward.title;
      titleKey = reward.titleKey;
      if (reward.formations) formations = reward.formations;
      if (reward.maxRerolls) maxRerolls = reward.maxRerolls;
    }
  }
  return { title, titleKey, formations, maxRerolls };
}

// ── XP from a run ───────────────────────────────────────────────────────────

const STAGE_XP: Record<string, number> = {
  "Eliminado en fase de liga": 50,
  "Eliminated in league phase": 50,
  "Eliminado en playoff": 100,
  "Eliminated in play-off": 100,
  "Eliminado en octavos": 100,
  "Eliminated in R16": 100,
  "Eliminado en cuartos": 150,
  "Eliminated in QF": 150,
  "Eliminado en semifinales": 200,
  "Eliminated in SF": 200,
  "Finalista": 250,
  "Runner-up": 250,
  "Campeón": 350,
  "Champion": 350,
};

export function xpForRun(stage: string): number {
  return STAGE_XP[stage] ?? 50;
}

// ── Achievement detection ───────────────────────────────────────────────────

export type AchievementCode = string;

type Squad = Record<string, { player: string; kind: string; overall: number; team?: string; league?: string; position?: string }>;

/**
 * Given a finished run's stored data, return the achievement codes it earns.
 * The caller intersects with what the user already has.
 */
export function detectAchievements(
  result: SquadSeason,
  squad: Squad,
  allRuns?: { result: SquadSeason; squad: Squad }[],
): AchievementCode[] {
  const codes: AchievementCode[] = [];
  const matches = result.matches ?? [];
  const teamName = result.teamName;
  const isChampion = result.champion === teamName;

  // ── Tournament ──
  if (isChampion) codes.push("cl_winner");
  if (result.stage === "Finalista" || result.stage === "Runner-up" || isChampion) {
    codes.push("cl_finalist");
  }
  // Reached SF = made it to at least SF
  const reachedSF = ["Finalista", "Runner-up", "Campeón", "Champion",
    "Eliminado en semifinales", "Eliminated in SF"].includes(result.stage);
  if (reachedSF) codes.push("cl_semifinalist");

  const koStages = new Set(["playoff", "r16", "qf", "sf", "final"]);
  const playedKO = matches.some((m) => koStages.has(m.stage));
  if (playedKO) codes.push("cl_knockout");

  // Unbeaten champion
  if (isChampion) {
    const lost = matches.some((m) => {
      const home = m.home === teamName;
      const us = home ? m.homeGoals : m.awayGoals;
      const them = home ? m.awayGoals : m.homeGoals;
      return us < them;
    });
    if (!lost) codes.push("cl_unbeaten");
  }

  // Comeback from 3+ down in a knockout tie
  const bracket = result.bracket;
  if (bracket) {
    for (const round of Object.values(bracket)) {
      for (const tie of round) {
        if (!tie.isSquad || tie.winner !== teamName) continue;
        // Check if squad was down 3+ after leg1
        const [a, b] = tie.leg1.split("-").map(Number);
        const squadIsA = tie.teamA === teamName;
        const leg1For = squadIsA ? a : b;
        const leg1Against = squadIsA ? b : a;
        if (leg1Against - leg1For >= 3) {
          codes.push("cl_comeback_tie");
          break;
        }
      }
    }
  }

  // ── Match ──
  for (const m of matches) {
    const home = m.home === teamName;
    const us = home ? m.homeGoals : m.awayGoals;
    const them = home ? m.awayGoals : m.homeGoals;

    if (us >= 5) codes.push("five_goals");
    if (us - them >= 4) codes.push("win_by_four");
    if (them === 0 && koStages.has(m.stage)) codes.push("clean_sheet_ko");

    // Comeback from 3+ behind in one match (checking events)
    if (m.events && us > them) {
      let running = 0;
      let maxDeficit = 0;
      for (const ev of m.events) {
        if (ev.type !== "goal") continue;
        const isUs = (home && ev.side === "home") || (!home && ev.side === "away");
        running += isUs ? 1 : -1;
        if (running < maxDeficit) maxDeficit = running;
      }
      if (maxDeficit <= -3) codes.push("comeback_match");
    }
  }

  // ── Squad ──
  const cards = Object.values(squad);
  const xi = cards.filter((_, i) => i < 11);
  const icons = cards.filter((c) => c.kind === "icono");

  if (icons.length >= 1) codes.push("draft_icon");
  if (icons.length >= 2) codes.push("draft_two_icons");

  const avgOverall = cards.reduce((a, c) => a + c.overall, 0) / Math.max(cards.length, 1);
  if (avgOverall >= 85) codes.push("squad_85");
  if (isChampion && avgOverall < 75) codes.push("giant_killer");

  // Mono-league: all XI from the same league
  if (xi.length === 11 && xi.every((c) => c.league)) {
    const leagues = new Set(xi.map((c) => c.league));
    if (leagues.size === 1) codes.push("mono_league");
  }

  // ── Collection (need all runs) ──
  if (allRuns) {
    const allSquads = [...allRuns.map((r) => r.squad), squad];
    const totalRuns = allSquads.length;
    if (totalRuns >= 10) codes.push("runs_10");
    if (totalRuns >= 50) codes.push("runs_50");
    if (totalRuns >= 100) codes.push("runs_100");

    const uniqueIcons = new Set<string>();
    for (const s of allSquads) {
      for (const c of Object.values(s)) {
        if (c.kind === "icono") uniqueIcons.add(c.player);
      }
    }
    if (uniqueIcons.size >= 10) codes.push("icons_10");

    const winFormations = new Set<string>();
    for (const r of allRuns) {
      if (r.result.champion === r.result.teamName) {
        const rXi = Object.values(r.squad).slice(0, 11);
        const f = formationFromXi(rXi);
        if (f) winFormations.add(f);
      }
    }
    // Check current run too
    if (isChampion) {
      const f = formationFromXi(xi);
      if (f) winFormations.add(f);
    }
    if (winFormations.size >= 5) codes.push("formations_5");

    const allLeagues = new Set<string>();
    for (const s of allSquads) {
      for (const c of Object.values(s)) {
        if (c.league) allLeagues.add(c.league);
      }
    }
    if (allLeagues.size >= 10) codes.push("leagues_10");
  }

  // Deduplicate
  return [...new Set(codes)];
}

function formationFromXi(
  xi: { position?: string }[],
): string | null {
  const n = (p: string) => xi.filter((x) => x.position === p).length;
  const d = n("Defender");
  const m = n("Midfielder");
  const f = n("Forward");
  if (d + m + f !== 10) return null;
  return `${d}-${m}-${f}`;
}

// ── Achievement catalogue ───────────────────────────────────────────────────

export type AchievementDef = {
  code: string;
  nameEs: string;
  nameEn: string;
  descEs: string;
  descEn: string;
  points: number;
  category: "tournament" | "match" | "squad" | "collection";
  icon: string;
  hidden: number;
};

export const ACHIEVEMENT_CATALOGUE: AchievementDef[] = [
  // ── Tournament ──
  { code: "cl_winner", nameEs: "Campeón", nameEn: "Champion", descEs: "Gana la Champions League", descEn: "Win the Champions League", points: 300, category: "tournament", icon: "trophy", hidden: 0 },
  { code: "cl_finalist", nameEs: "Finalista", nameEn: "Finalist", descEs: "Llega a la final", descEn: "Reach the final", points: 150, category: "tournament", icon: "medal", hidden: 0 },
  { code: "cl_semifinalist", nameEs: "Semifinalista", nameEn: "Semifinalist", descEs: "Llega a semifinales", descEn: "Reach the semifinals", points: 100, category: "tournament", icon: "target", hidden: 0 },
  { code: "cl_knockout", nameEs: "Fase eliminatoria", nameEn: "Knockout Stage", descEs: "Clasifícate para los octavos de final", descEn: "Qualify for the round of 16", points: 50, category: "tournament", icon: "swords", hidden: 0 },
  { code: "cl_unbeaten", nameEs: "Invicto", nameEn: "Unbeaten", descEs: "Gana la Champions sin perder un solo partido", descEn: "Win the Champions League without losing a single match", points: 500, category: "tournament", icon: "shield-check", hidden: 0 },
  { code: "cl_comeback_tie", nameEs: "Noche mágica", nameEn: "Magic Night", descEs: "Remonta una eliminatoria perdiendo 3+ goles en la ida", descEn: "Come back in a knockout tie after trailing 3+ goals on the first leg", points: 200, category: "tournament", icon: "flame", hidden: 1 },

  // ── Match ──
  { code: "five_goals", nameEs: "Goleada", nameEn: "Goal Fest", descEs: "Marca 5 o más goles en un partido", descEn: "Score 5 or more goals in a match", points: 150, category: "match", icon: "zap", hidden: 0 },
  { code: "win_by_four", nameEs: "Paseo", nameEn: "Walkover", descEs: "Gana un partido por 4 o más goles", descEn: "Win a match by 4 or more goals", points: 100, category: "match", icon: "chevrons-up", hidden: 0 },
  { code: "clean_sheet_ko", nameEs: "Muro", nameEn: "Clean Sheet", descEs: "Deja la portería a cero en una eliminatoria", descEn: "Keep a clean sheet in a knockout match", points: 75, category: "match", icon: "shield", hidden: 0 },
  { code: "comeback_match", nameEs: "Remontada", nameEn: "Comeback", descEs: "Remonta un 3-0 en contra y gana el partido", descEn: "Come back from 3-0 down and win the match", points: 200, category: "match", icon: "rotate-ccw", hidden: 1 },

  // ── Squad ──
  { code: "draft_icon", nameEs: "Leyenda en el equipo", nameEn: "A Legend Joins", descEs: "Draftea un ícono", descEn: "Draft an icon card", points: 100, category: "squad", icon: "star", hidden: 0 },
  { code: "draft_two_icons", nameEs: "Panteón", nameEn: "Pantheon", descEs: "Draftea 2 o más íconos en un mismo draft", descEn: "Draft 2 or more icons in the same draft", points: 200, category: "squad", icon: "sparkles", hidden: 0 },
  { code: "squad_85", nameEs: "Dream Team", nameEn: "Dream Team", descEs: "Plantilla con media de 85 o más", descEn: "Build a squad with 85+ average rating", points: 100, category: "squad", icon: "crown", hidden: 0 },
  { code: "giant_killer", nameEs: "Matagigantes", nameEn: "Giant Killer", descEs: "Gana la Champions con una plantilla de media menor a 75", descEn: "Win the Champions League with a squad averaging below 75", points: 500, category: "squad", icon: "siren", hidden: 1 },
  { code: "mono_league", nameEs: "Purista", nameEn: "Purist", descEs: "Titulares de una misma liga", descEn: "All 11 starters from the same league", points: 150, category: "squad", icon: "flag", hidden: 0 },

  // ── Collection ──
  { code: "runs_10", nameEs: "Habitual", nameEn: "Regular", descEs: "Juega 10 partidas de Draft", descEn: "Play 10 Draft runs", points: 100, category: "collection", icon: "repeat", hidden: 0 },
  { code: "runs_50", nameEs: "Adicto", nameEn: "Hooked", descEs: "Juega 50 partidas de Draft", descEn: "Play 50 Draft runs", points: 200, category: "collection", icon: "infinity", hidden: 0 },
  { code: "runs_100", nameEs: "Centenario", nameEn: "Centurion", descEs: "Juega 100 partidas de Draft", descEn: "Play 100 Draft runs", points: 500, category: "collection", icon: "gem", hidden: 0 },
  { code: "icons_10", nameEs: "Coleccionista", nameEn: "Collector", descEs: "Draftea 10 íconos distintos entre todas tus partidas", descEn: "Draft 10 different icons across all your runs", points: 200, category: "collection", icon: "library", hidden: 0 },
  { code: "formations_5", nameEs: "Camaleón", nameEn: "Chameleon", descEs: "Gana la Champions con 5 formaciones distintas", descEn: "Win the Champions League with 5 different formations", points: 300, category: "collection", icon: "layout-grid", hidden: 0 },
  { code: "leagues_10", nameEs: "Cosmopolita", nameEn: "Cosmopolitan", descEs: "Draftea jugadores de 10 ligas distintas", descEn: "Draft players from 10 different leagues", points: 150, category: "collection", icon: "globe", hidden: 0 },
];
