import type { Metadata } from "next";
import { getLocale, getTranslations, setRequestLocale } from "next-intl/server";

import { Calibration } from "@/components/results/calibration";
import { RpsLadder } from "@/components/results/rps-ladder";
import { Reveal } from "@/components/ui/reveal";
import { api, type TrackRecord } from "@/lib/api";
import { cn, formatNumber } from "@/lib/utils";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: "results" });
  return { title: t("title"), description: t("lead") };
}

const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;

export default async function ResultsPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale: paramLocale } = await params;
  setRequestLocale(paramLocale);
  const t = await getTranslations("results");
  const locale = await getLocale();

  let data: TrackRecord | null = null;
  try {
    data = await api.trackRecord();
  } catch {
    data = null;
  }

  return (
    <div className="mx-auto max-w-[1000px] px-5 py-12 sm:px-7 sm:py-16">
      <header className="max-w-2xl">
        <p className="text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-brand">
          {t("eyebrow")}
        </p>
        <h1 className="font-display mt-3 text-[2.2rem] leading-[1.1] sm:text-[2.9rem]">
          {t("title")}
        </h1>
        <p className="mt-4 text-[0.97rem] leading-relaxed text-muted">
          {t("lead")}
        </p>
      </header>

      {!data ? (
        <div className="mt-12 rounded-[14px] border border-border bg-surface px-6 py-12 text-center">
          <p className="text-[1rem] font-semibold">{t("offlineTitle")}</p>
          <p className="mx-auto mt-2 max-w-md text-[0.87rem] text-muted">
            {t("offlineBody")}
          </p>
        </div>
      ) : (
        <div className="mt-14 flex flex-col gap-20">
          {/* 1 · Against the market — the same measurement the landing leads on,
                 so a visitor arriving from there lands on familiar ground. */}
          <RpsLadder benchmark={data.benchmark} />

          {/* 2 · The live log */}
          <Reveal as="section">
            <h2 className="font-display text-[1.6rem] leading-tight sm:text-[2rem]">
              {t("liveTitle")}
            </h2>
            <p className="mt-3 max-w-2xl text-[0.92rem] leading-relaxed text-muted">
              {t("liveLead")}
            </p>

            {data.live ? (
              <>
                <dl className="mt-8 grid grid-cols-2 gap-px overflow-hidden rounded-[var(--radius-card)] border border-border bg-border sm:grid-cols-4">
                  {[
                    {
                      k: t("settled"),
                      v: formatNumber(data.live.count, locale),
                      tone: "",
                    },
                    {
                      k: t("matches"),
                      v: formatNumber(data.live.matches, locale),
                      tone: "",
                    },
                    {
                      k: t("hitRate"),
                      v: pct(data.live.hitRate),
                      // Green only when reality kept up with the claim. A hit
                      // rate far above what was announced is luck, not skill,
                      // and colouring it as success would teach the wrong thing.
                      tone:
                        data.live.hitRate >= data.live.announced - 0.02
                          ? "text-positive"
                          : "text-negative",
                    },
                    { k: t("announced"), v: pct(data.live.announced), tone: "" },
                  ].map((m) => (
                    <div key={m.k} className="bg-surface px-5 py-5">
                      <dt className="text-[0.63rem] font-semibold uppercase tracking-[0.1em] text-dim">
                        {m.k}
                      </dt>
                      <dd
                        className={cn(
                          "mt-2 text-[1.6rem] font-semibold leading-none",
                          m.tone,
                        )}
                      >
                        {m.v}
                      </dd>
                    </div>
                  ))}
                </dl>
                <p className="mt-4 text-[0.82rem] leading-relaxed text-dim">
                  {t("honestNote")}
                </p>

                <h3 className="mt-12 text-[0.68rem] font-semibold uppercase tracking-[0.14em] text-muted">
                  {t("byMarket")}
                </h3>
                <div className="mt-4 overflow-hidden rounded-[var(--radius-card)] border border-border">
                  <table className="w-full text-[0.86rem]">
                    <thead>
                      <tr className="border-b border-border bg-surface-2 text-left text-[0.66rem] uppercase tracking-[0.09em] text-dim">
                        <th className="px-4 py-2.5 font-semibold">{t("market")}</th>
                        <th className="px-4 py-2.5 text-right font-semibold">
                          {t("n")}
                        </th>
                        <th className="px-4 py-2.5 text-right font-semibold">
                          {t("hitRate")}
                        </th>
                        <th className="px-4 py-2.5 text-right font-semibold">
                          {t("announced")}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.live.byMarket.map((m) => (
                        <tr
                          key={m.key}
                          className="border-b border-border last:border-0 bg-surface"
                        >
                          <td className="px-4 py-2.5 font-medium">
                            {t.has(`markets.${m.key}`)
                              ? t(`markets.${m.key}`)
                              : m.key}
                          </td>
                          <td className="px-4 py-2.5 text-right tabular-nums text-muted">
                            {m.n}
                          </td>
                          <td className="px-4 py-2.5 text-right font-semibold tabular-nums">
                            {pct(m.hitRate, 0)}
                          </td>
                          <td className="px-4 py-2.5 text-right tabular-nums text-muted">
                            {pct(m.announced, 0)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <p className="mt-8 rounded-[var(--radius-card)] border border-border bg-surface px-5 py-8 text-[0.88rem] text-muted">
                {t("noSettled")}
              </p>
            )}
          </Reveal>

          {/* 3 · Calibration */}
          {data.calibration?.length ? (
            <Calibration bins={data.calibration} />
          ) : null}
        </div>
      )}
    </div>
  );
}
