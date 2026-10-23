"use client";

import { useEffect, useMemo, useSyncExternalStore } from "react";
import { createQuery } from "@/lib/matchday-cache";

type Selection = {
  tab: "day" | "round" | "favorites";
  day: string;
  competition: string;
  round: number | null;
};
const INITIAL: Selection = { tab: "day", day: "", competition: "", round: null };
const STORAGE = "mv.matchday.v1";
let selection: Selection | undefined;
const listeners = new Set<() => void>();

function readSelection(): Selection {
  if (selection) return selection;
  selection = INITIAL;
  try {
    const saved = JSON.parse(window.sessionStorage.getItem(STORAGE) ?? "null");
    if (saved && ["day", "round", "favorites"].includes(saved.tab)) {
      selection = {
        tab: saved.tab,
        day: typeof saved.day === "string" && /^\d{4}-\d{2}-\d{2}$/.test(saved.day) ? saved.day : "",
        competition: typeof saved.competition === "string" ? saved.competition : "",
        round: Number.isInteger(saved.round) && saved.round > 0 ? saved.round : null,
      };
    }
  } catch { /* Storage may be unavailable; memory still survives navigation. */ }
  return selection;
}

function subscribeSelection(notify: () => void) {
  listeners.add(notify);
  return () => { listeners.delete(notify); };
}

export function setMatchdaySelection(patch: Partial<Selection>) {
  selection = { ...readSelection(), ...patch };
  try { window.sessionStorage.setItem(STORAGE, JSON.stringify(selection)); } catch { /* Optional. */ }
  listeners.forEach((notify) => notify());
}

export function useMatchdaySelection() {
  return useSyncExternalStore(subscribeSelection, readSelection, () => INITIAL);
}

const queries = new Map<string, ReturnType<typeof createQuery<unknown>>>();

/** How long to keep asking a cold API, and how often. */
const RETRY_EVERY_MS = 5_000;
const RETRY_FOR_MS = 120_000;

export function useMatchdayQuery<T>(key: string, load: () => Promise<T>, ttl = 300_000) {
  const query = useMemo(() => {
    // Never retain request data in a shared server module.
    if (typeof window === "undefined") return createQuery<T>();
    if (!queries.has(key)) queries.set(key, createQuery<unknown>());
    return queries.get(key)! as ReturnType<typeof createQuery<T>>;
  }, [key]);
  const snapshot = useSyncExternalStore(query.subscribe, query.read, query.server);
  const failing = snapshot.failed && !snapshot.data;
  useEffect(() => {
    const refresh = () => { void query.refresh(load, ttl); };
    refresh();
    window.addEventListener("focus", refresh);
    // A cold API answers 503 (or nothing at all) while it loads its fit, and
    // this screen fetches from the browser, so without a retry the panel stayed
    // on "could not load" until the tab regained focus — for an API that was
    // perfectly healthy twenty seconds later. Keep asking, slowly, for two
    // minutes: long enough to cover a startup, short enough not to hammer an
    // API that is genuinely down.
    let stop: number | undefined;
    let retry: number | undefined;
    if (failing) {
      retry = window.setInterval(refresh, RETRY_EVERY_MS);
      stop = window.setTimeout(() => window.clearInterval(retry), RETRY_FOR_MS);
    }
    return () => {
      window.removeEventListener("focus", refresh);
      if (retry !== undefined) window.clearInterval(retry);
      if (stop !== undefined) window.clearTimeout(stop);
    };
  }, [query, load, ttl, failing]);
  return { ...snapshot, loading: !snapshot.data && !snapshot.failed };
}
