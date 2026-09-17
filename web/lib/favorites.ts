"use client";

import { useCallback, useSyncExternalStore } from "react";

import { useMounted } from "@/lib/use-mounted";

/**
 * Followed teams and competitions.
 *
 * localStorage until accounts exist. That is a deliberate staging post, not a
 * shortcut: the shape below (`{teams, competitions}`) is the shape the account
 * will store, so moving it server-side later swaps the storage and leaves every
 * component untouched.
 */
const KEY = "mv.favorites.v1";

export type Favorites = { teams: string[]; competitions: string[] };

const EMPTY: Favorites = { teams: [], competitions: [] };

function parse(raw: string | null): Favorites {
  if (!raw) return EMPTY;
  try {
    const parsed = JSON.parse(raw) as Partial<Favorites>;
    return {
      teams: Array.isArray(parsed.teams) ? parsed.teams : [],
      competitions: Array.isArray(parsed.competitions) ? parsed.competitions : [],
    };
  } catch {
    // A value someone hand-edited.
    return EMPTY;
  }
}

// Used only when storage refuses a write (full, or blocked in a private
// window), so a toggle still works for the rest of the session.
let memory: Favorites | null = null;
// The store must hand React the same object until the stored value changes.
let cachedRaw: string | null = null;
let cached: Favorites = EMPTY;

function getSnapshot(): Favorites {
  if (memory) return memory;
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(KEY);
  } catch {
    // A private window or cleared site data.
  }
  if (raw !== cachedRaw) {
    cachedRaw = raw;
    cached = parse(raw);
  }
  return cached;
}

const getServerSnapshot = () => EMPTY;

// Writes in this tab don't fire "storage", so toggles notify directly.
const listeners = new Set<() => void>();

function subscribe(onChange: () => void) {
  // Keep two open tabs in step.
  const onStorage = (e: StorageEvent) => {
    if (e.key === KEY) onChange();
  };
  listeners.add(onChange);
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(onChange);
    window.removeEventListener("storage", onStorage);
  };
}

export function useFavorites() {
  // Empty on the server and during hydration so the first paint matches, then
  // the stored list. Reading storage during render would desync hydration.
  const favorites = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const ready = useMounted();

  const toggle = useCallback((kind: keyof Favorites, value: string) => {
    const prev = getSnapshot();
    const list = prev[kind];
    const next = {
      ...prev,
      [kind]: list.includes(value)
        ? list.filter((v) => v !== value)
        : [...list, value],
    };
    try {
      window.localStorage.setItem(KEY, JSON.stringify(next));
    } catch {
      memory = next;
    }
    listeners.forEach((notify) => notify());
  }, []);

  const has = useCallback(
    (kind: keyof Favorites, value: string) => favorites[kind].includes(value),
    [favorites],
  );

  return { favorites, toggle, has, ready };
}
