/**
 * Typed client for the Mundialytics API.
 *
 * Every call goes through `request`, which is where the two things that will
 * bite in production live: a timeout (the engine can be cold) and a revalidate
 * window (fixtures change when a match kicks off, not on every page view).
 */
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Probabilities = { home: number; draw: number; away: number };

export type Fixture = {
  slug: string;
  competition: string;
  competitionName: string;
  season: string;
  matchday: number | null;
  kickoff: string | null;
  home: string;
  away: string;
  homeSlug: string;
  awaySlug: string;
  played: boolean;
  score: { home: number; away: number } | null;
  probabilities: Probabilities | null;
};

export type MatchPrediction = {
  probabilities: Probabilities;
  expectedGoals: { home: number; away: number };
  goals: {
    over15: number;
    over25: number;
    over35: number;
    under25: number;
    btts: number;
  };
  scorelines: { score: string; p: number }[];
  scoreMatrix: number[][];
  expectedStats: { key: string; home: number; away: number }[];
};

export type Scorer = { player: string; p: number; minutes: number };

export type TeamMarket = {
  key: string;
  lambdaHome: number | null;
  lambdaAway: number | null;
  lambdaTotal: number | null;
  lines: { line: number; over: number; side: "over" | "under"; p: number }[];
};

export type ActualStats = {
  goals: { home: number; away: number };
  stats: { key: string; home: number; away: number }[];
};

export type Match = Fixture & {
  prediction: MatchPrediction;
  scorers: { home?: Scorer[]; away?: Scorer[] } | null;
  teamProps: TeamMarket[] | null;
  /** Real shots, corners and cards — null until the data refresh ingests them. */
  actual: ActualStats | null;
};

export type Competition = {
  slug: string;
  name: string;
  country: string;
  type: string;
  seasons: string[];
  currentSeason: string;
};

export type Benchmark = {
  sample: number;
  window: string;
  rows: { key: string; rps: number }[];
  byLeague: { league: string; gap: number }[];
};

export type CalibrationBin = { announced: number; observed: number; n: number };

export type TrackRecord = {
  benchmark: Benchmark;
  calibration: CalibrationBin[] | null;
  live: {
    count: number;
    matches: number;
    hitRate: number;
    announced: number;
    byMarket: { key: string; n: number; hitRate: number; announced: number }[];
  } | null;
  dataThrough: string;
};

export type LeagueCard = {
  slug: string;
  name: string;
  country: string;
  season: string;
  played: number;
  leader: { team: string; points: number } | null;
  favourite: string | null;
  pTitle: number | null;
};

export type LeagueRow = {
  rank: number;
  team: string;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  goalsFor: number;
  goalsAgainst: number;
  points: number;
  pTitle: number | null;
  pTop4: number | null;
  pRelegation: number | null;
  expectedPoints: number | null;
};

export type LeagueForecast = {
  competition: string;
  competitionName: string;
  season: string;
  matchday: number | null;
  remaining: number;
  standings: LeagueRow[];
  positionMatrix: { teams: string[]; positions: number[]; values: number[][] };
  upcoming: {
    home: string; away: string; date: string;
    pHome: number; pDraw: number; pAway: number; slug: string;
  }[];
};

export type UefaCard = {
  slug: string;
  name: string;
  season: string;
  favourite: string | null;
  pChampion: number | null;
  teams: number | null;
  played: number | null;
};

export type UefaRow = {
  team: string;
  elo: number;
  leaguePhase: number | null;
  topEight: number | null;
  playoff: number | null;
  roundOf16: number | null;
  quarterFinal: number | null;
  semiFinal: number | null;
  final: number | null;
  champion: number | null;
};

export type UefaFixture = {
  home: string;
  away: string;
  slug: string;
  date: string | null;
  played: boolean;
  result: string | null;
  probabilities: Probabilities | null;
  over25: number | null;
};

export type UefaCompetition = {
  slug: string;
  name: string;
  season: string;
  teams: number;
  played: number;
  leaguePhaseTotal: number;
  preDraw: boolean;
  simulations: number;
  standings: UefaRow[];
  phase: { rounds: number[]; round: number | null; fixtures: UefaFixture[] };
};

export type ScorerRace = {
  competition: string;
  competitionName: string;
  season: string;
  leader: { player: string; goals: number } | null;
  inTheRace: number;
  players: {
    player: string;
    team: string;
    goals: number;
    expected: number;
    p: number;
  }[];
};

export type SquadPlayer = {
  player: string;
  display: string;
  team: string;
  position: string;
  overall: number;
  /** Shot-stopping for keepers, offensive strength for everyone else. */
  attack: number;
  defense: number;
  creation: number;
  matches: number;
};

export type SquadPool = {
  competition: string;
  competitionName: string;
  slots: Record<string, number>;
  positions: string[];
  players: Record<string, SquadPlayer[]>;
};

/** A goal or a booking, with the minute it happened on. */
export type SquadMatchEvent = {
  type: "goal" | "card";
  side: "home" | "away";
  player: string;
  assist: string | null;
  minute: number;
};

export type SquadPlayerRating = {
  player: string;
  rating: number;
  goals: number;
  assists: number;
  cards: number;
};

export type SquadFixture = {
  home: string;
  away: string;
  homeGoals: number;
  awayGoals: number;
  isSquad: boolean;
  /** Narrative detail travels only for the squad's own matches. */
  events: SquadMatchEvent[] | null;
  stats: { key: string; home: number; away: number }[] | null;
  ratings: SquadPlayerRating[] | null;
};

export type SquadSeason = {
  teamName: string;
  replaced: string | null;
  matchdays: { matchday: number; fixtures: SquadFixture[] }[];
  standings: {
    rank: number; team: string; played: number; points: number;
    goalsFor: number; goalsAgainst: number; isSquad: boolean;
  }[];
  scorers: { player: string; team: string; goals: number; assists: number; isSquad: boolean }[];
  odds: {
    title: number | null; top2: number | null; top4: number | null;
    relegation: number | null; expectedPoints?: number; expectedGoals?: number;
  } | null;
  finish: {
    rank: number; team: string; played: number; points: number;
    goalsFor: number; goalsAgainst: number; isSquad: boolean;
  } | null;
};

export type DayFixtures = {
  date: string;
  count: number;
  competitions: {
    slug: string;
    name: string;
    country: string;
    fixtures: Fixture[];
  }[];
};

export type FixtureCalendar = {
  today: string;
  from: string;
  to: string;
  counts: Record<string, number>;
};

export type UpcomingDay = { date: string; fixtures: Fixture[] };

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, revalidate = 300): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20_000);
  try {
    const res = await fetch(`${BASE}${path}`, {
      signal: controller.signal,
      next: { revalidate },
    });
    if (!res.ok) {
      throw new ApiError(`${path} returned ${res.status}`, res.status);
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(timeout);
  }
}

export const api = {
  competitions: () => request<Competition[]>("/catalogue", 3600),
  fixturesDay: (day: string) => request<DayFixtures>(`/fixtures/day?day=${day}`, 300),
  fixturesCalendar: () => request<FixtureCalendar>("/fixtures/calendar", 900),
  upcoming: (days = 8) =>
    request<{ days: UpcomingDay[]; count: number }>(
      `/fixtures/upcoming?days=${days}`,
      300,
    ),
  round: (competition: string, season?: string, matchday?: number) => {
    const q = new URLSearchParams({ competition });
    if (season) q.set("season", season);
    if (matchday != null) q.set("matchday", String(matchday));
    return request<{
      competition: string;
      competitionName: string;
      season: string;
      matchday: number;
      rounds: number[];
      fixtures: Fixture[];
    }>(`/fixtures?${q}`, 300);
  },
  match: (competition: string, fixture: string) =>
    request<Match>(`/match/${competition}/${fixture}`, 300),
  trackRecord: () => request<TrackRecord>("/track-record", 600),
  leagues: () => request<LeagueCard[]>("/leagues", 600),
  league: (slug: string) => request<LeagueForecast>(`/league/${slug}`, 600),
  uefa: () => request<UefaCard[]>("/competitions", 600),
  awards: (slug: string) => request<ScorerRace>(`/awards/${slug}`, 1800),
  squadPool: (competition: string) =>
    request<SquadPool>(`/squadlab/pool?competition=${competition}`, 3600),
  playSeason: async (competition: string, squad: string[], mode: string) => {
    // A POST carrying a squad the user just invented: nothing to cache, since
    // no two visitors send the same body. Playing 380 matches takes real
    // seconds, so this deliberately skips `request`'s 20s abort.
    const res = await fetch(`${BASE}/squadlab/season`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ competition, squad, mode }),
      cache: "no-store",
    });
    if (!res.ok) {
      throw new ApiError(`season returned ${res.status}`, res.status);
    }
    return (await res.json()) as SquadSeason;
  },
  uefaCompetition: (slug: string, matchday?: number) =>
    request<UefaCompetition>(
      `/competition/${slug}${matchday != null ? `?matchday=${matchday}` : ""}`,
      600,
    ),
};
