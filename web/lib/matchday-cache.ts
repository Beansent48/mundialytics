// Browser-session cache. Next's fetch revalidation applies on the server,
// not to the requests made by the interactive matchday screen.
export type QuerySnapshot<T> = { data?: T; failed: boolean };
const EMPTY = { failed: false };

export function createQuery<T>() {
  let snapshot: QuerySnapshot<T> = EMPTY;
  let expires = 0;
  let pending: Promise<void> | undefined;
  const listeners = new Set<() => void>();
  return {
    read: () => snapshot,
    server: () => EMPTY as QuerySnapshot<T>,
    subscribe(notify: () => void) {
      listeners.add(notify);
      return () => { listeners.delete(notify); };
    },
    refresh(load: () => Promise<T>, ttl: number) {
      if (pending) return pending;
      if (Date.now() < expires) return Promise.resolve();
      pending = Promise.resolve().then(load).then(
        (data) => {
          snapshot = { data, failed: false };
          expires = Date.now() + ttl;
        },
        () => {
          // Keep an already loaded list available during a temporary outage.
          snapshot = { ...snapshot, failed: true };
        },
      ).finally(() => {
        pending = undefined;
        listeners.forEach((notify) => notify());
      });
      return pending;
    },
  };
}
