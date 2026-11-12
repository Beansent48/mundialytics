"use client";

import { useTranslations } from "next-intl";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ErrorAnalysisResponse } from "@/lib/api";

const TIER_COLORS: Record<string, string> = {
  low: "var(--chart-1)",
  mid: "var(--chart-2)",
  high: "var(--chart-3)",
};

export function ErrorAnalysis({ data }: { data: ErrorAnalysisResponse }) {
  const t = useTranslations("analytics.errors");

  return (
    <div className="mt-6 space-y-10">
      {/* RPS by season */}
      <div>
        <h3 className="text-[0.92rem] font-semibold">{t("bySeason")}</h3>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart
            data={data.by_season}
            margin={{ top: 16, right: 8, bottom: 0, left: -8 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="var(--border)"
              vertical={false}
            />
            <XAxis
              dataKey="season"
              tick={{ fontSize: 11, fill: "var(--text-dim)" }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "var(--text-dim)" }}
              tickLine={false}
              axisLine={false}
              width={52}
              domain={["dataMin - 0.002", "dataMax + 0.002"]}
              tickFormatter={(v: number) => v.toFixed(4)}
            />
            <Tooltip
              contentStyle={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(value: number, _name: string) => [
                value.toFixed(6),
                "RPS",
              ]}
              labelStyle={{ color: "var(--text)", fontWeight: 600 }}
            />
            <Bar dataKey="rps" radius={[4, 4, 0, 0]} maxBarSize={40}>
              {data.by_season.map((_, i) => (
                <Cell key={i} fill="var(--brand)" fillOpacity={0.75} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* RPS by league */}
      <div>
        <h3 className="text-[0.92rem] font-semibold">{t("byLeague")}</h3>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart
            data={data.by_league}
            layout="vertical"
            margin={{ top: 8, right: 8, bottom: 0, left: 80 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="var(--border)"
              horizontal={false}
            />
            <XAxis
              type="number"
              tick={{ fontSize: 11, fill: "var(--text-dim)" }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              domain={["dataMin - 0.002", "dataMax + 0.002"]}
              tickFormatter={(v: number) => v.toFixed(4)}
            />
            <YAxis
              type="category"
              dataKey="league"
              tick={{ fontSize: 11, fill: "var(--text-muted)" }}
              tickLine={false}
              axisLine={false}
              width={80}
            />
            <Tooltip
              contentStyle={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(value: number) => [value.toFixed(6), "RPS"]}
              labelStyle={{ color: "var(--text)", fontWeight: 600 }}
            />
            <Bar dataKey="rps" radius={[0, 4, 4, 0]} maxBarSize={28}>
              {data.by_league.map((_, i) => (
                <Cell key={i} fill="var(--chart-2)" fillOpacity={0.75} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* RPS by strength tier */}
      <div>
        <h3 className="text-[0.92rem] font-semibold">{t("byTier")}</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart
            data={data.by_tier}
            margin={{ top: 16, right: 8, bottom: 0, left: -8 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="var(--border)"
              vertical={false}
            />
            <XAxis
              dataKey="tier"
              tick={{ fontSize: 11, fill: "var(--text-dim)" }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              tickFormatter={(v: string) => t(`tier.${v}`)}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "var(--text-dim)" }}
              tickLine={false}
              axisLine={false}
              width={52}
              domain={["dataMin - 0.002", "dataMax + 0.002"]}
              tickFormatter={(v: number) => v.toFixed(4)}
            />
            <Tooltip
              contentStyle={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(value: number) => [value.toFixed(6), "RPS"]}
              labelStyle={{ color: "var(--text)", fontWeight: 600 }}
            />
            <Bar dataKey="rps" radius={[4, 4, 0, 0]} maxBarSize={48}>
              {data.by_tier.map((entry) => (
                <Cell
                  key={entry.tier}
                  fill={TIER_COLORS[entry.tier] ?? "var(--brand)"}
                  fillOpacity={0.75}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Poisson GOF table */}
      {data.scoreline_gof && (
        <div>
          <h3 className="text-[0.92rem] font-semibold">{t("gofTitle")}</h3>
          <p className="mt-1 text-[0.8rem] text-muted">
            χ² = {data.scoreline_gof.chi2}, df = {data.scoreline_gof.df}, p ={" "}
            {data.scoreline_gof.p < 0.001
              ? "< 0.001"
              : data.scoreline_gof.p.toFixed(4)}
          </p>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[400px] text-[0.78rem]">
              <thead>
                <tr className="border-b border-border text-left text-dim">
                  <th className="pb-2 pr-4 font-medium">{t("score")}</th>
                  <th className="pb-2 pr-4 text-right font-medium">
                    {t("observed")}
                  </th>
                  <th className="pb-2 pr-4 text-right font-medium">
                    {t("expected")}
                  </th>
                  <th className="pb-2 text-right font-medium">{t("diff")}</th>
                </tr>
              </thead>
              <tbody>
                {data.scoreline_gof.bins.slice(0, 15).map((bin) => (
                  <tr
                    key={bin.score}
                    className="border-b border-border/50 tabular-nums"
                  >
                    <td className="py-1.5 pr-4 font-medium">{bin.score}</td>
                    <td className="py-1.5 pr-4 text-right">{bin.observed}</td>
                    <td className="py-1.5 pr-4 text-right">{bin.expected}</td>
                    <td className="py-1.5 text-right">
                      {(bin.observed - bin.expected).toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
