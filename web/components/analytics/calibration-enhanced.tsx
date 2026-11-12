"use client";

import { useTranslations } from "next-intl";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import type { EnhancedCalibrationResponse } from "@/lib/api";

export function CalibrationEnhanced({
  data,
}: {
  data: EnhancedCalibrationResponse;
}) {
  const t = useTranslations("analytics.calibration");

  const points = data.bins.map((b) => ({
    announced: b.announced,
    observed: b.observed,
    ci_lo: b.ci_lo,
    ci_hi: b.ci_hi,
    n: b.n,
  }));

  const diagonal = [
    { x: 0, y: 0 },
    { x: 1, y: 1 },
  ] as const;

  return (
    <div className="mt-6 space-y-8">
      {/* Reliability diagram */}
      <div>
        <ResponsiveContainer width="100%" height={360}>
          <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="var(--border)"
            />
            <XAxis
              dataKey="announced"
              type="number"
              domain={[0, 1]}
              tick={{ fontSize: 11, fill: "var(--text-dim)" }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              name={t("announced")}
              label={{
                value: t("announced"),
                position: "insideBottom",
                offset: -16,
                style: { fontSize: 12, fill: "var(--text-muted)" },
              }}
            />
            <YAxis
              dataKey="observed"
              type="number"
              domain={[0, 1]}
              tick={{ fontSize: 11, fill: "var(--text-dim)" }}
              tickLine={false}
              axisLine={false}
              width={40}
              name={t("observed")}
              label={{
                value: t("observed"),
                angle: -90,
                position: "insideLeft",
                offset: 12,
                style: { fontSize: 12, fill: "var(--text-muted)" },
              }}
            />
            <ZAxis dataKey="n" range={[40, 200]} />
            <ReferenceLine
              segment={diagonal}
              stroke="var(--border-strong)"
              strokeDasharray="6 4"
              strokeWidth={1.5}
            />
            <Tooltip
              contentStyle={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={((value: number, name: string) => [
                `${(value * 100).toFixed(1)}%`,
                name === "observed" ? t("observed") : t("announced"),
              ]) as never}
              labelFormatter={() => ""}
            />
            <Scatter
              data={points}
              fill="var(--brand)"
              strokeWidth={2}
              stroke="var(--surface)"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* CI bands as a line chart overlay */}
      <div>
        <h3 className="text-[0.92rem] font-semibold">{t("ciTitle")}</h3>
        <p className="mt-1 text-[0.8rem] text-muted">{t("ciLead")}</p>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={points}
            margin={{ top: 16, right: 16, bottom: 0, left: 8 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="var(--border)"
            />
            <XAxis
              dataKey="announced"
              tick={{ fontSize: 11, fill: "var(--text-dim)" }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "var(--text-dim)" }}
              tickLine={false}
              axisLine={false}
              width={40}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip
              contentStyle={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={((value: number, name: string) => [
                `${(value * 100).toFixed(1)}%`,
                name === "observed"
                  ? t("observed")
                  : name === "ci_lo"
                    ? t("ciLo")
                    : name === "ci_hi"
                      ? t("ciHi")
                      : t("announced"),
              ]) as never}
            />
            <Line
              type="monotone"
              dataKey="ci_hi"
              stroke="var(--border-strong)"
              strokeDasharray="4 3"
              strokeWidth={1}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="ci_lo"
              stroke="var(--border-strong)"
              strokeDasharray="4 3"
              strokeWidth={1}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="observed"
              stroke="var(--brand)"
              strokeWidth={2}
              dot={{ r: 4, fill: "var(--brand)", stroke: "var(--surface)", strokeWidth: 2 }}
            />
            <Line
              type="monotone"
              dataKey="announced"
              stroke="var(--text-dim)"
              strokeDasharray="6 4"
              strokeWidth={1.5}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Brier decomposition */}
      <div className="rounded-[var(--radius-card)] border border-border bg-surface p-5">
        <h3 className="text-[0.92rem] font-semibold">{t("brierTitle")}</h3>
        <p className="mt-1 text-[0.8rem] text-muted">{t("brierLead")}</p>
        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          {(
            [
              ["brier", data.brier],
              ["reliability", data.reliability],
              ["resolution", data.resolution],
              ["uncertainty", data.uncertainty],
            ] as const
          ).map(([key, value]) => (
            <div key={key}>
              <p className="text-[0.7rem] font-medium uppercase tracking-wider text-dim">
                {t(`brier.${key}`)}
              </p>
              <p className="mt-1 font-display text-[1.4rem] tabular-nums">
                {value.toFixed(4)}
              </p>
            </div>
          ))}
        </div>
        <p className="mt-3 text-[0.75rem] text-dim">
          n = {data.n.toLocaleString()}
        </p>
      </div>
    </div>
  );
}
