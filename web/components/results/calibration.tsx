import { getTranslations } from "next-intl/server";

import { Reveal } from "@/components/ui/reveal";
import type { CalibrationBin } from "@/lib/api";

/**
 * Announced probability against observed frequency.
 *
 * Drawn as paired bars rather than the usual scatter against a diagonal: a
 * reliability diagram is a specialist's chart, and the question it answers —
 * when we say 40%, does it happen 40% of the time — is answered faster by two
 * bars sitting next to each other.
 */
export async function Calibration({ bins }: { bins: CalibrationBin[] }) {
  const t = await getTranslations("results.calibration");
  const total = bins.reduce((s, b) => s + b.n, 0);
  const ece = bins.reduce(
    (s, b) => s + (b.n / total) * Math.abs(b.announced - b.observed),
    0,
  );
  const max = Math.max(...bins.flatMap((b) => [b.announced, b.observed]));

  return (
    <Reveal as="section">
      <h2 className="font-display text-[1.6rem] leading-tight sm:text-[2rem]">
        {t("title")}
      </h2>
      <p className="mt-3 max-w-2xl text-[0.92rem] leading-relaxed text-muted">
        {t("lead")}
      </p>

      <div className="mt-8 rounded-[var(--radius-card)] border border-border bg-surface p-6 sm:p-8">
        <div className="flex items-end gap-2 sm:gap-4">
          {bins.map((b) => (
            <div key={b.announced} className="flex flex-1 flex-col items-center gap-2">
              <div className="flex h-40 w-full items-end justify-center gap-1">
                <span
                  className="w-1/2 rounded-t-[3px] bg-border-strong"
                  style={{ height: `${(b.announced / max) * 100}%` }}
                  title={t("announced")}
                />
                <span
                  className="w-1/2 rounded-t-[3px] bg-brand"
                  style={{ height: `${(b.observed / max) * 100}%` }}
                  title={t("observed")}
                />
              </div>
              <span className="text-[0.62rem] tabular-nums text-dim">
                {Math.round(b.announced * 100)}%
              </span>
            </div>
          ))}
        </div>

        <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2 text-[0.75rem] text-muted">
          <span className="flex items-center gap-2">
            <span className="size-2.5 rounded-sm bg-border-strong" />
            {t("announced")}
          </span>
          <span className="flex items-center gap-2">
            <span className="size-2.5 rounded-sm bg-brand" />
            {t("observed")}
          </span>
        </div>

        <div className="rule-fade my-7" />
        <p className="text-[0.9rem] leading-relaxed">
          {t("ece", { value: ece.toFixed(3) })}
        </p>
      </div>
    </Reveal>
  );
}
