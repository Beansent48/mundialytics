"use client";

import { useSyncExternalStore } from "react";

const subscribe = () => () => {};

/**
 * False on the server and during hydration, true once running in the browser.
 *
 * For UI that depends on something only the client knows (stored theme,
 * localStorage, a random pick): render the neutral version first so the
 * hydrated markup matches the server's, then the real one.
 */
export function useMounted(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  );
}
