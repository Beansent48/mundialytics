/**
 * Headline figures shown on the landing page.
 *
 * The benchmark is read from `data/benchmark_vs_bet365.json`, which
 * `scripts/benchmark_vs_bet365.py` writes (with the hash of the predictions it
 * scored) and the API's /track-record serves too. Nothing here is typed in by
 * hand any more: the constant this replaced kept saying 0.2025 for months after
 * the deployed engine reached 0.2008.
 *
 * Every number here must be reproducible. If it cannot be pointed at a script,
 * it does not belong on the page.
 */
import benchmark from "@/data/benchmark_vs_bet365.json";

const rpsOf = (key: string) => {
  const row = benchmark.rows.find((r) => r.key === key);
  if (!row) throw new Error(`benchmark_vs_bet365.json has no "${key}" row`);
  return row.rps;
};

export const STATS = {
  /** Offline fallback only; the hero asks the API for the live count. */
  matches: 46_091,
  seasons: 27,
  markets: 30,
  leagues: 5,
  /** Big Five matches in the 2020/21–2025/26 benchmark window. */
  benchmarkSample: benchmark.sample,
} as const;

/** Ranked probability score on 1X2 — lower is better. */
const RPS = {
  uniform: rpsOf("uniform"),
  baseRates: rpsOf("baseRates"),
  engine: rpsOf("engine"),
  market: rpsOf("market"),
} as const;

/**
 * The ladder, ordered best first.
 *
 * Two deliberate choices, because the obvious chart lies about the ranking:
 *
 * 1. Order. RPS is better when lower, so listing it descending puts the engine
 *    third from the top and a reader counts it as third best. Best first.
 * 2. Encoding. The bar is not the RPS — it is the share of the achievable
 *    distance covered, from "no information at all" (0%) to the closing price
 *    (100%). Longer therefore means better, the way a bar is read everywhere
 *    else. The raw RPS stays on the right so nothing is hidden.
 */
const span = RPS.uniform - RPS.market;
const share = (rps: number) => (RPS.uniform - rps) / span;

export const BENCHMARK = [
  { key: "market", rps: RPS.market, share: share(RPS.market), market: true },
  { key: "engine", rps: RPS.engine, share: share(RPS.engine), highlight: true },
  { key: "baseRates", rps: RPS.baseRates, share: share(RPS.baseRates) },
  { key: "uniform", rps: RPS.uniform, share: share(RPS.uniform) },
] as const;

export const COVERAGE = {
  /** RPS between knowing nothing and the closing price. */
  span,
  /** Share of it the engine covers. */
  covered: share(RPS.engine),
} as const;

/**
 * A real engine output, shown on the landing so the product is visible before
 * anyone signs up. Produced by `predict_match("valencia", "barcelona")` on the
 * deployed chain — a screenshot of the model, not an illustration of one.
 * Becomes a live call once the API lands.
 */
export const SAMPLE_FIXTURE = {
  home: "Valencia",
  away: "Barcelona",
  competition: "LaLiga",
  pHome: 0.2686,
  pDraw: 0.2787,
  pAway: 0.4527,
  lambdaHome: 1.22,
  lambdaAway: 1.53,
  pOver25: 0.5205,
  pBtts: 0.574,
  scorelines: [
    { score: "1-1", p: 0.1393 },
    { score: "1-2", p: 0.0913 },
    { score: "0-0", p: 0.0837 },
    { score: "0-1", p: 0.077 },
  ],
} as const;
