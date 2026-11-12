"use client";

import { useTranslations } from "next-intl";
import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { TeamProfileResponse } from "@/lib/api";

function titleCase(s: string) {
  return s.replace(/\b\w/g, (c) => c.toUpperCase());
}

export function TeamProfile({ data }: { data: TeamProfileResponse }) {
  const t = useTranslations("analytics.profile");

  // Downsample Elo trajectory to ~200 points for performance
  const eloData = useMemo(() => {
    const traj = data.elo.trajectory;
    if (traj.length <= 200) return traj;
    const step = Math.ceil(traj.length / 200);
    return traj.filter((_, i) => i % step === 0 || i === traj.length - 1);
  }, [data.elo.trajectory]);

  // Radar data: normalize attack/defense/strength to 0-100 scale using league range
  const radarData = useMemo(() => {
    if (!data.profile) return [];
    const p = data.profile;
    const range = p.league_max_strength - p.league_min_strength;
    const normalize = (val: number, avg: number) => {
      if (range === 0) return 50;
      return Math.max(0, Math.min(100, ((val - avg) / (range / 2)) * 50 + 50));
    };
    return [
      { axis: t("attack"), value: normalize(p.attack, p.league_avg_attack), raw: p.attack },
      { axis: t("defense"), value: normalize(p.defense, p.league_avg_defense), raw: p.defense },
      { axis: t("strength"), value: normalize(p.strength, 0), raw: p.strength },
      { axis: t("balance"), value: Math.abs(p.attack_balance) * 100, raw: p.attack_balance },
    ];
  }, [data.profile, t]);

  return (
    <div className="mt-6 space-y-10">
      {/* Elo trajectory */}
      <div>
        <h3 className="text-[0.92rem] font-semibold">{t("eloTitle")}</h3>
        <div className="mt-2 flex gap-6 text-[0.8rem] text-muted">
          <span>
            {t("current")}: <strong className="text-text">{data.elo.current ?? "–"}</strong>
          </span>
          <span>
            {t("peak")}: <strong className="text-text">{data.elo.peak ?? "–"}</strong>
          </span>
          <span>
            {t("matches")}: <strong className="text-text">{data.elo.trajectory.length}</strong>
          </span>
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart
            data={eloData}
            margin={{ top: 16, right: 8, bottom: 0, left: -8 }}
          >
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
              width={48}
              domain={["dataMin - 20", "dataMax + 20"]}
            />
            <Tooltip
              contentStyle={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                fontSize: 12,
              }}
              labelStyle={{ color: "var(--text)", fontWeight: 600 }}
              formatter={((value: number) => [value.toFixed(1), "Elo"]) as never}
              labelFormatter={(_, payload) => {
                const p = payload?.[0]?.payload;
                return p ? `${p.date} vs ${titleCase(p.opponent)} (${p.venue})` : "";
              }}
            />
            <Line
              type="monotone"
              dataKey="elo"
              stroke="var(--brand)"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: "var(--brand)", stroke: "var(--surface)", strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Attack / defense radar */}
      {data.profile && (
        <div>
          <h3 className="text-[0.92rem] font-semibold">{t("radarTitle")}</h3>
          <p className="mt-1 text-[0.8rem] text-muted">{t("radarLead")}</p>
          <div className="mx-auto max-w-[380px]">
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
                <PolarGrid stroke="var(--border)" />
                <PolarAngleAxis
                  dataKey="axis"
                  tick={{ fontSize: 11, fill: "var(--text-muted)" }}
                />
                <PolarRadiusAxis
                  tick={{ fontSize: 9, fill: "var(--text-dim)" }}
                  domain={[0, 100]}
                  tickCount={4}
                  axisLine={false}
                />
                <Radar
                  dataKey="value"
                  stroke="var(--brand)"
                  fill="var(--brand)"
                  fillOpacity={0.25}
                  strokeWidth={2}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={((_: number, __: string, entry: { payload?: { raw?: number } }) =>
                    [entry.payload?.raw?.toFixed(4) ?? "–", ""]
                  ) as never}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Head to head */}
      {data.h2h.length > 0 && (
        <div>
          <h3 className="text-[0.92rem] font-semibold">{t("h2hTitle")}</h3>
          <div className="mt-4 space-y-4">
            {data.h2h.map((rival) => {
              const wins = rival.matches.filter((m) => m.goals > m.conceded).length;
              const draws = rival.matches.filter((m) => m.goals === m.conceded).length;
              const losses = rival.matches.length - wins - draws;
              return (
                <div
                  key={rival.rival}
                  className="rounded-[var(--radius-card)] border border-border bg-surface p-4"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold">{titleCase(rival.rival)}</span>
                    <span className="text-[0.78rem] text-muted">
                      {rival.total} {t("totalMatches")}
                    </span>
                  </div>
                  <div className="mt-2 flex gap-4 text-[0.8rem]">
                    <span className="text-green-600 dark:text-green-400">
                      {t("wins")}: {wins}
                    </span>
                    <span className="text-dim">
                      {t("draws")}: {draws}
                    </span>
                    <span className="text-red-500 dark:text-red-400">
                      {t("losses")}: {losses}
                    </span>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1">
                    {rival.matches.map((m, i) => {
                      const result =
                        m.goals > m.conceded
                          ? "W"
                          : m.goals === m.conceded
                            ? "D"
                            : "L";
                      const bg =
                        result === "W"
                          ? "bg-green-600/20 text-green-700 dark:text-green-300"
                          : result === "D"
                            ? "bg-border text-dim"
                            : "bg-red-500/20 text-red-600 dark:text-red-300";
                      return (
                        <span
                          key={i}
                          className={`inline-flex items-center rounded px-1.5 py-0.5 text-[0.68rem] font-medium tabular-nums ${bg}`}
                          title={`${m.date}: ${m.goals}-${m.conceded} (${m.venue})`}
                        >
                          {m.goals}-{m.conceded}
                        </span>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
