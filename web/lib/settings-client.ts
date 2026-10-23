"use client";

export type Settings = {
  locale: string | null;
  theme: string | null;
  oddsFormat: string | null;
};

/**
 * Read and write the preferences that follow the account.
 *
 * Never through `lib/api.ts`: that client caches under a shared `data` tag, and
 * one person's settings must never be served to another. `no-store` here and
 * `cache-control: no-store, private` on the way back.
 */
export async function fetchSettings(): Promise<Settings | null> {
  try {
    const res = await fetch("/api/user/settings", { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as Settings;
  } catch {
    return null;
  }
}

/**
 * Save the fields that changed, and only those.
 *
 * Fire-and-forget on purpose: this is called from a theme or language switch,
 * where the change has already been applied locally and the person should not
 * wait for a round trip to see it. A failure costs the preference on other
 * devices, not the one in front of them.
 */
export function saveSettings(patch: Partial<Settings>): void {
  void fetch("/api/user/settings", {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(patch),
    cache: "no-store",
  }).catch(() => {
    // Offline, or signed out between the click and the request.
  });
}
