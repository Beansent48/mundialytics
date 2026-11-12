"use client";

import { useTranslations } from "next-intl";
import { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { DecompositionMatch } from "@/lib/api";

const SERIES = [
  { key: "gl", color: "var(--chart-1)" },
  { key: "ad", color: "var(--chart-2)" },
  { key: "adx", color: "var(--chart-3)" },
  { key: "xr", color: "var(--chart-4)" },
] as const;

export function DecompositionChart({
  matches,
}: {
  matches: DecompositionMatch[];
}) {
  const t = useTranslations("analytics.decomposition");

  const data = useMemo(
    () =>
      matches.map((m) => ({
        date: m.date,
        label: `${m.date} vs ${m.opponent}`,
        gl: m.gl,
        ad: m.ad,
        adx: m.adx,
        xr: m.xr,
        goals: m.goals,
      })),
    [matches],
  );

  if (!data.length) {
    return (
      <p className="py-8 text-center text-[0.87rem] text-muted">{t("empty")}</p>
    );
  }

  return (
    <div className="mt-6">
      <ResponsiveContainer width="100%" height={360}>
        <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -8 }}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="var(--border)"
            vertical={false}
          />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11, fill: "var(--text-dim)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            interval="preserveStartEnd"
            minTickGap={60}
          />
          <YAxis
            tick={{ fontSize: 11, fill: "var(--text-dim)" }}
            tickLine={false}
            axisLine={false}
            width={40}
            label={{
              value: "λ",
              position: "insideTopLeft",
              offset: 6,
              style: { fontSize: 12, fill: "var(--text-muted)" },
            }}
          />
          <Tooltip
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "var(--text)", fontWeight: 600 }}
            labelFormatter={(_, payload) => {
              const p = payload?.[0]?.payload;
              return p ? p.label : "";
            }}
          />
          <Legend
            iconType="square"
            iconSize={10}
            wrapperStyle={{ fontSize: 12, color: "var(--text-muted)" }}
            formatter={(value: string) => t(`series.${value}`)}
          />
          {SERIES.map((s) => (
            <Area
              key={s.key}
              type="monotone"
              dataKey={s.key}
              stackId="1"
              stroke={s.color}
              fill={s.color}
              fillOpacity={0.6}
              strokeWidth={1.5}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
