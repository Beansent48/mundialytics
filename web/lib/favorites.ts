"use client";

import { useCallback, useEffect, useState } from "react";

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

function read(): Favorites {
  if (typeof window === "undefined") return EMPTY;
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return EMPTY;
    const parsed = JSON.parse(raw) as Partial<Favorites>;
    return {
      teams: Array.isArray(parsed.teams) ? parsed.teams : [],
      competitions: Array.isArray(parsed.competitions) ? parsed.competitions : [],
    };
  } catch {
    // A private window, cleared site data, or a value someone hand-edited.
    return EMPTY;
  }
}

export function useFavorites() {
  // Starts empty on both server and client so the first paint matches, then
  // loads on mount. Reading storage during render would desync hydration.
  const [favorites, setFavorites] = useState<Favorites>(EMPTY);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setFavorites(read());
    setReady(true);

    // Keep two open tabs in step.
    const onStorage = (e: StorageEvent) => {
      if (e.key === KEY) setFavorites(read());
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const toggle = useCallback((kind: keyof Favorites, value: string) => {
    setFavorites((prev) => {
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
        // Storage full or blocked: the toggle still works for this session.
      }
      return next;
    });
  }, []);

  const has = useCallback(
    (kind: keyof Favorites, value: string) => favorites[kind].includes(value),
    [favorites],
  );

  return { favorites, toggle, has, ready };
}
