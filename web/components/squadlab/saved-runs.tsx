"use client";

import { Loader2, Play, Trash2, Trophy } from "lucide-react";
import { useLocale, useTranslations } from "next-intl";
import { useCallback, useEffect, useState } from "react";

import { deleteRun, listRuns, type RunSummary } from "@/lib/squadlab-runs";
import { cn } from "@/lib/utils";

/**
 * The runs this account has saved, on the setup screen.
 *
 * Shown only to somebody signed in, and shown as nothing at all otherwise —
 * not as a locked panel advertising what an account would buy. The site's rule
 * is that an account unlocks saving, not seeing, and a greyed-out teaser is a
 * gate wearing a disguise.
 *
 * An in-progress draft can be continued; a finished one shows what it achieved
 * and is read-only, because the tournament it recorded cannot be replayed into
 * itself once the data has moved underneath it.
 */
export function SavedRuns({
  signedIn,
  onContinue,
}: {
  signedIn: boolean;
  onContinue: (id: string) => void;
}) {
  const t = useTranslations("squadlab");
  const locale = useLocale();
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  // Bumped to ask for a reload. The fetch and its setState live inside the
  // effect rather than in a callback the effect calls, which is what keeps the
  // result keyed to the request that asked for it — the pattern this codebase
  // adopted when it removed its synchronous resets inside effects.
  const [reload, setReload] = useState(0);
  const refresh = useCallback(() => setReload((n) => n + 1), []);

  useEffect(() => {
    if (!signedIn) return;
    let cancelled = false;
    (async () => {
      const next = await listRuns();
      if (!cancelled) setRuns(next);
    })();
    return () => {
      cancelled = true;
    };
  }, [signedIn, reload]);

  // Signing out is handled here rather than by clearing state in the effect
  // above: a synchronous reset inside an effect is the pattern this codebase
  // removed once already, and the render has the answer anyway.
  if (!signedIn || !runs || runs.length === 0) return null;

  const date = new Intl.DateTimeFormat(locale, {
    day: "numeric",
    month: "short",
  });

  return (
    <div className="mt-10 max-w-xl">
      <p className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-dim">
        {t("saved.title")}
      </p>
      <ul className="mt-3 flex flex-col gap-2">
        {runs.map((run) => {
          const inProgress = run.status === "in_progress";
          return (
            <li
              key={run.id}
              className="flex items-center gap-3 rounded-[12px] border border-border bg-surface px-4 py-3"
            >
              <Trophy
                className={cn(
                  "size-4 shrink-0",
                  inProgress ? "text-dim" : "text-warning",
                )}
              />
              <div className="min-w-0 flex-1">
                <p className="truncate text-[0.88rem] font-medium">
                  {inProgress
                    ? t("saved.inProgress", { players: run.players })
                    : (run.stage ?? t("saved.finished"))}
                </p>
                <p className="truncate text-[0.76rem] text-muted">
                  {t(`mode.${run.mode}`)} · {date.format(new Date(run.createdAt))}
                </p>
              </div>

              {inProgress ? (
                <button
                  type="button"
                  onClick={() => onContinue(run.id)}
                  className="inline-flex items-center gap-1.5 rounded-[9px] border border-border-strong px-2.5 py-1.5 text-[0.78rem] font-medium transition-colors hover:border-brand hover:text-brand"
                >
                  <Play className="size-3" />
                  {t("saved.continue")}
                </button>
              ) : null}

              <button
                type="button"
                aria-label={t("saved.delete")}
                disabled={busy === run.id}
                onClick={async () => {
                  setBusy(run.id);
                  await deleteRun(run.id);
                  setBusy(null);
                  refresh();
                }}
                className="flex size-7 shrink-0 items-center justify-center rounded-lg border border-border text-muted transition-colors hover:border-negative hover:text-negative disabled:opacity-50"
              >
                {busy === run.id ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <Trash2 className="size-3.5" />
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
