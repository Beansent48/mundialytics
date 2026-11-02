"use client";

import type { SquadPlayer } from "@/lib/api";

/**
 * Saving a SquadLab run, from the browser.
 *
 * Nothing in SquadLab was stored before this: the component carried the draft
 * in React state and warned you on the way out that leaving would lose the
 * team. So this is not a storage swap like favourites — there was no first
 * backend. Signed out, that is still exactly what happens, because no page may
 * become sign-in-only; signed in, the draft and the tournament it becomes are
 * rows.
 *
 * Never through `lib/api.ts`: that caches under a shared `data` tag, and one
 * person's draft must never be served to another.
 */
export type RunSummary = {
  id: string;
  mode: string;
  seed: number;
  status: "in_progress" | "finished";
  stage: string | null;
  teamName: string | null;
  players: number;
  engineFingerprint: string | null;
  createdAt: string;
  updatedAt: string;
};

export type SavedRun = {
  id: string;
  mode: string;
  seed: number;
  squad: Record<string, SquadPlayer>;
  rerolls: Record<string, number>;
  status: "in_progress" | "finished";
  result: unknown | null;
  engineFingerprint: string | null;
  createdAt: string;
  updatedAt: string;
};

export type RunInput = {
  id: string;
  mode: string;
  seed: number;
  /**
   * The chosen cards by slot. The whole card, not just the name: the same
   * footballer is in the pool as an "actual", a "prime" and sometimes an
   * "icono", so a name alone would resume a draft showing one you never held.
   */
  squad: Record<string, SquadPlayer>;
  rerolls: Record<string, number>;
  status: "in_progress" | "finished";
  result: unknown | null;
};

const ENDPOINT = "/api/user/squadlab/runs";

/** A run id the browser mints, so a draft and the season it becomes are one row. */
export function mintRunId(): string {
  return crypto.randomUUID();
}

export async function listRuns(): Promise<RunSummary[]> {
  try {
    const res = await fetch(ENDPOINT, { cache: "no-store" });
    if (!res.ok) return [];
    return (await res.json()) as RunSummary[];
  } catch {
    return [];
  }
}

export async function loadRun(id: string): Promise<SavedRun | null> {
  try {
    const res = await fetch(`${ENDPOINT}/${encodeURIComponent(id)}`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as SavedRun;
  } catch {
    return null;
  }
}

/**
 * Save, and say whether it landed.
 *
 * The caller decides what a failure means. For an in-progress draft it means
 * nothing — the next edit saves again — which is why the autosave ignores it.
 * For a finished tournament it is worth telling the person, because that one
 * cannot be recreated by playing again.
 */
export async function saveRun(input: RunInput): Promise<boolean> {
  try {
    const res = await fetch(ENDPOINT, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(input),
      cache: "no-store",
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function deleteRun(id: string): Promise<boolean> {
  try {
    const res = await fetch(`${ENDPOINT}/${encodeURIComponent(id)}`, {
      method: "DELETE",
      cache: "no-store",
    });
    return res.ok;
  } catch {
    return false;
  }
}
