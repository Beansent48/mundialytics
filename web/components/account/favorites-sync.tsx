"use client";

import { useEffect, useRef } from "react";

import { useSession } from "@/lib/auth-client";
import { readFavorites, setFavorites, type Favorites } from "@/lib/favorites";

/**
 * The one moment that is interesting: first sign-in on this browser.
 *
 * §5 of the design — **merge, never replace**. Someone who followed Valencia on
 * their phone and Betis on their laptop signs in on the laptop and must end
 * with both. A replace in either direction loses one of them, and which one
 * depends on the order they happened to open the two browsers in, which is not
 * a rule anybody could predict.
 *
 * So: the first time an account is seen on this browser, the local set is sent
 * up to be unioned and the answer adopted. After that, the server is simply
 * read. The local copy is never deleted — it stays as the offline cache and as
 * what the person keeps if they sign out, so signing out is not data loss.
 *
 * Rendered once, in the locale layout, rather than inside `useFavorites`: that
 * hook is used by many components at once and this must happen exactly once.
 * It renders nothing.
 */
const MIGRATED = "mv.favorites.migrated.v1";

function alreadyMerged(userId: string): boolean {
  try {
    const raw = window.localStorage.getItem(MIGRATED);
    const ids = raw ? (JSON.parse(raw) as unknown) : [];
    return Array.isArray(ids) && ids.includes(userId);
  } catch {
    // A private window: treat it as not merged. Merging twice is a union with
    // itself, which is the same set — this is safe to get wrong in this
    // direction, and losing a favourite is not.
    return false;
  }
}

function rememberMerged(userId: string): void {
  try {
    const raw = window.localStorage.getItem(MIGRATED);
    const ids = raw ? (JSON.parse(raw) as unknown) : [];
    const list = Array.isArray(ids) ? ids.filter((x) => typeof x === "string") : [];
    if (!list.includes(userId)) list.push(userId);
    window.localStorage.setItem(MIGRATED, JSON.stringify(list));
  } catch {
    // Not fatal; the next sign-in merges again, to the same set.
  }
}

export function FavoritesSync() {
  const { data: session } = useSession();
  const userId = session?.user.id ?? null;
  // One sync per account per page load, even though `useSession` re-renders.
  const syncedFor = useRef<string | null>(null);

  useEffect(() => {
    if (!userId || syncedFor.current === userId) return;
    syncedFor.current = userId;

    const merge = !alreadyMerged(userId);
    const local = readFavorites();
    const request: RequestInit = merge
      ? {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(local),
        }
      : { method: "GET" };

    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/user/favorites", {
          ...request,
          cache: "no-store",
        });
        if (!res.ok) throw new Error(String(res.status));
        const server = (await res.json()) as Favorites;
        if (cancelled) return;
        setFavorites(server);
        if (merge) rememberMerged(userId);
      } catch {
        // Offline, or the account layer is not configured. The local copy
        // stays exactly as it was, which is what a signed-out visitor uses,
        // so nothing is lost and the next load tries again.
        syncedFor.current = null;
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [userId]);

  return null;
}
