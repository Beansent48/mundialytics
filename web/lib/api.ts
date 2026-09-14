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

export type Outcome = "home" | "draw" | "away";

export type MatchHeadline = {
  modalOutcome: Outcome;
  modalOutcomeProbability: number;
  expectedScore: string;
  likelyScore: string;
  byOutcome: Record<
    Outcome,
    { score: string; outcomeProbability: number; conditionalProbability: number }
  >;
};

export type MatchPrediction = {
  headline: MatchHeadline;
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
  expectedStats: ExpectedStat[];
};

/** Most-probable integer band [lo, hi] and the probability the count lands in it. */
export type StatRange = { lo: number; hi: number; p: number };

export type ExpectedStat = {
  key: string;
  home: number;
  away: number;
  /** Most-probable range per side; null when the model can't price the stat. */
  homeRange: StatRange | null;
  awayRange: StatRange | null;
};

export type Scorer = { player: string; p: number; minutes: number };

export type TeamMarket = {
  key: string;
  lambdaHome: number | null;
  lambdaAway: number | null;
  lambdaTotal: number | null;
  /** Headline: the range the match total most likely falls in. */
  range: StatRange | null;
  lines: { line: number; over: number; side: "over" | "under"; p: number }[];
};

export type ActualStats = {
  goals: { home: number; away: number };
  stats: { key: string; home: number; away: number }[];
};

/** Who did what, grouped per side. No minutes — the aggregated source has none. */
export type SideEvents = {
  goals: { player: string; count: number }[];
  assists: { player: string; count: number }[];
  yellows: string[];
  /** Goals with no attributed scorer (own goals, unmapped names). */
  unattributed: number;
};

export type TimelineEventType =
  | "goal"
  | "own_goal"
  | "penalty"
  | "yellow"
  | "red";

/** One event on the minute-by-minute timeline, when ESPN supplied the clock. */
export type TimelineEvent = {
  /** Match clock as shown, stoppage included ("45'+2'"). */
  minute: string;
  side: "home" | "away";
  type: TimelineEventType;
  player: string;
  /** Assister, for goals that had one. */
  assist: string | null;
};

export type MatchEvents = {
  home: SideEvents;
  away: SideEvents;
  /** Minute-ordered feed; null when the match has no per-event source. */
  timeline: TimelineEvent[] | null;
};

export type Match = Fixture & {
  prediction: MatchPrediction;
  scorers: { home?: Scorer[]; away?: Scorer[] } | null;
  teamProps: TeamMarket[] | null;
  /** Real shots, corners and cards — null until the data refresh ingests them. */
  actual: ActualStats | null;
  /** Real goals/assists/cards per side — null until the refresh ingests them. */
  events: MatchEvents | null;
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
  role: string;
  /** "actual" | "prime" | "icono" — drives the card's design. */
  kind: string;
  /** Prime season label (e.g. "23/24"); null for actual/icono. */
  season: string | null;
  overall: number;
  /** Shot-stopping for keepers, offensive strength for everyone else. */
  attack: number;
  defense: number;
  creation: number;
  matches: number;
  /** Measured sub-stat percentiles (0-100) behind the rating; ACTUAL only. */
  substats: Record<string, number> | null;
  /** Curated "why this card" line for prime/icono; null for actual. */
  highlight: { stat: string; note: string } | null;
};

export type SquadPool = {
  competition: string;
  competitionName: string;
  slots: Record<string, number>;
  positions: string[];
  players: Record<string, SquadPlayer[]>;
  /** Per-role sub-stat weight formula, /100. */
  roleWeights: Record<string, Record<string, number>>;
  /** Human labels for the sub-stat codes. */
  substatLabels: Record<string, string>;
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

/** One of the squad's own matches. Only the squad's matches carry detail. */
export type SquadFixture = {
  /** "liga" | "playoff" | "r16" | "qf" | "sf" | "final" */
  stage: string;
  stageLabel: string;
  home: string;
  away: string;
  homeGoals: number;
  awayGoals: number;
  isSquad: boolean;
  events: SquadMatchEvent[] | null;
  stats: { key: string; home: number; away: number }[] | null;
  ratings: SquadPlayerRating[] | null;
};

export type LeaguePhaseRow = {
  rank: number;
  team: string;
  played: number;
  points: number;
  goalsFor: number;
  goalsAgainst: number;
  goalDiff: number;
  isSquad: boolean;
};

/** A two-legged knockout tie (the final is a single leg, leg2 empty). */
export type BracketTie = {
  round: string;
  roundLabel: string;
  teamA: string;
  teamB: string;
  leg1: string;
  leg2: string;
  agg: string;
  winner: string;
  /** "" | "pró." (extra time) | "pen." (penalties) */
  note: string;
  isSquad: boolean;
};

export type SquadBracket = {
  playoff: BracketTie[];
  r16: BracketTie[];
  qf: BracketTie[];
  sf: BracketTie[];
  final: BracketTie[];
};

/** The result of one Champions run with the drafted eleven. */
export type SquadSeason = {
  teamName: string;
  /** The real club whose slot in the draw the squad took. */
  replaced: string | null;
  squadElo: number;
  seed: number;
  champion: string;
  runnerUp: string;
  /** e.g. "Campeón", "Finalista", "Eliminado en octavos". */
  stage: string;
  /** Finishing position in the 36-team league phase. */
  leaguePhaseRank: number;
  leaguePhase: LeaguePhaseRow[];
  /** The squad's own matches, chronological: league phase then knockout. */
  matches: SquadFixture[];
  bracket: SquadBracket;
  scorers: { player: string; goals: number; assists: number; isSquad: boolean }[];
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
  // The squad always plays the Champions, so the pool is pan-European — no
  // competition to scope it to.
  squadPool: () => request<SquadPool>(`/squadlab/pool`, 3600),
  playChampions: async (squad: string[], mode: string, seed?: number) => {
    // A POST carrying a squad the user just invented: nothing to cache, since
    // no two visitors send the same body, and each play takes a random slot in
    // the draw. Playing the whole tournament takes real seconds, so this
    // deliberately skips `request`'s 20s abort.
    const res = await fetch(`${BASE}/squadlab/season`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ squad, mode, seed }),
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
