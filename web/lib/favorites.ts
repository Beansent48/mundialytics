"use client";

import { useCallback, useSyncExternalStore } from "react";

import { useSession } from "@/lib/auth-client";
import { useMounted } from "@/lib/use-mounted";

/**
 * Followed teams and competitions.
 *
 * Two backends, not one replacing the other. Signed out, this is
 * `localStorage` and behaves exactly as it did before accounts existed —
 * nothing on the site was ever gated, and a visitor who never signs in must
 * lose nothing. Signed in, the server is the truth and the local copy becomes
 * an optimistic cache: a toggle paints immediately and is written through, and
 * what comes back from the server wins.
 *
 * The component API — `{favorites, toggle, has, ready}` — is unchanged, which
 * was the whole point of the shape this file has always stored. The one-time
 * merge on first sign-in lives in `components/account/favorites-sync.tsx`.
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

/** Write the local copy and tell every mounted component. */
export function setFavorites(next: Favorites): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(next));
    memory = null;
  } catch {
    memory = next;
  }
  listeners.forEach((notify) => notify());
}

export function readFavorites(): Favorites {
  return getSnapshot();
}

/**
 * Send the whole set to the server, and adopt whatever it says back.
 *
 * Adopting the answer rather than assuming the write landed is what keeps the
 * UI honest: if the request fails, the next read re-adopts the server's real
 * state instead of showing a heart that was never saved.
 */
async function push(next: Favorites): Promise<void> {
  try {
    const res = await fetch("/api/user/favorites", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(next),
      cache: "no-store",
    });
    if (!res.ok) throw new Error(String(res.status));
    setFavorites((await res.json()) as Favorites);
  } catch {
    // Offline, or signed out between the click and the request. The local copy
    // keeps the toggle for this session and the next sync reconciles it.
  }
}

export function useFavorites() {
  // Empty on the server and during hydration so the first paint matches, then
  // the stored list. Reading storage during render would desync hydration.
  const favorites = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const ready = useMounted();
  const { data: session } = useSession();

  const toggle = useCallback(
    (kind: keyof Favorites, value: string) => {
      const prev = getSnapshot();
      const list = prev[kind];
      const next = {
        ...prev,
        [kind]: list.includes(value)
          ? list.filter((v) => v !== value)
          : [...list, value],
      };
      // Paint first: a follow button that waits for a round trip feels broken
      // even when the round trip is fast.
      setFavorites(next);
      if (session) void push(next);
    },
    [session],
  );

  const has = useCallback(
    (kind: keyof Favorites, value: string) => favorites[kind].includes(value),
    [favorites],
  );

  return { favorites, toggle, has, ready };
}
