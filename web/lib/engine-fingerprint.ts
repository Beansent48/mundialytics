/**
 * What the engine looked like when a run was played.
 *
 * A SquadLab run is reproducible from its seed, but only against the data it
 * was played on: ratings and pools move with the daily refresh, so replaying
 * last week's draft can produce a different tournament. Stamping each finished
 * run with a fingerprint is what lets a later reader tell "this is the result"
 * from "this is a result we could get today" — the same reasoning the
 * pre-kickoff prediction log already uses.
 *
 * It is built from `/health`, which already reports the two things that move
 * when the foundation does: how many matches are loaded and the date they run
 * through. **No change to `api/` was needed**, which matters — the API stays
 * stateless and anonymous, and nothing here can reach a served probability.
 *
 * Read on the server, not in the browser: the server reaches the API directly
 * on the container network, so this costs one internal hop instead of a round
 * trip through the proxy (`lib/api-base.ts` has both routes).
 */
import { apiBase } from "@/lib/api-base";

// It changes once a day at most, and a run being stamped with a fingerprint a
// minute stale is not a problem worth a request per save.
const TTL_MS = 60_000;

let cached: { value: string | null; at: number } | null = null;

export async function engineFingerprint(): Promise<string | null> {
  if (cached && Date.now() - cached.at < TTL_MS) return cached.value;

  let value: string | null = null;
  try {
    const res = await fetch(`${apiBase()}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(2000),
    });
    if (res.ok) {
      const health = (await res.json()) as {
        ready?: boolean;
        matches?: number;
        dataThrough?: string;
      };
      // Only once the models are loaded: before that `/health` answers without
      // `matches`, and "unknown" is a better stamp than a wrong one.
      if (health.ready && health.matches && health.dataThrough) {
        value = `${health.matches}@${health.dataThrough}`;
      }
    }
  } catch {
    // The API being down must not stop somebody saving their own run.
  }

  cached = { value, at: Date.now() };
  return value;
}
