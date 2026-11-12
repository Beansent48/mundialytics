"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";
import { useSearchParams, useRouter, usePathname } from "next/navigation";

import { CalibrationEnhanced } from "@/components/analytics/calibration-enhanced";
import { DecompositionChart } from "@/components/analytics/decomposition-chart";
import { ErrorAnalysis } from "@/components/analytics/error-analysis";
import { TeamProfile } from "@/components/analytics/team-profile";
import { Reveal } from "@/components/ui/reveal";
import type {
  DecompositionResponse,
  EnhancedCalibrationResponse,
  ErrorAnalysisResponse,
  TeamProfileResponse,
} from "@/lib/api";

function titleCase(s: string) {
  return s.replace(/\b\w/g, (c) => c.toUpperCase());
}

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`/engine${path}`);
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

export function AnalyticsShell({
  teams,
  initialErrors,
  initialCalibration,
}: {
  teams: string[];
  initialErrors: ErrorAnalysisResponse | null;
  initialCalibration: EnhancedCalibrationResponse | null;
}) {
  const t = useTranslations("analytics");
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const defaultTeam = searchParams.get("team") ?? teams[0] ?? "";
  const [team, setTeam] = useState(defaultTeam);
  const [decomposition, setDecomposition] = useState<DecompositionResponse | null>(null);
  const [profile, setProfile] = useState<TeamProfileResponse | null>(null);
  const [loading, setLoading] = useState(!!defaultTeam);

  useEffect(() => {
    if (!team) return;
    let cancelled = false;
    Promise.all([
      fetchJson<DecompositionResponse>(`/analytics/decomposition?team=${encodeURIComponent(team)}`),
      fetchJson<TeamProfileResponse>(`/analytics/team-profile?team=${encodeURIComponent(team)}`),
    ]).then(([dec, prof]) => {
      if (cancelled) return;
      setDecomposition(dec);
      setProfile(prof);
    }).catch(() => {
      if (cancelled) return;
      setDecomposition(null);
      setProfile(null);
    }).finally(() => {
      if (!cancelled) setLoading(false);
    });
    return () => { cancelled = true; };
  }, [team]);

  const onTeamChange = (newTeam: string) => {
    setLoading(true);
    setTeam(newTeam);
    const params = new URLSearchParams(searchParams.toString());
    params.set("team", newTeam);
    router.replace(`${pathname}?${params.toString()}`, { scroll: false });
  };

  return (
    <div className="space-y-16">
      {/* Team selector */}
      <div className="sticky top-16 z-10 -mx-5 border-b border-border bg-bg/80 px-5 py-3 backdrop-blur-sm sm:-mx-7 sm:px-7">
        <div className="flex items-center gap-3">
          <label htmlFor="team-select" className="text-[0.8rem] font-medium text-muted">
            {t("selectTeam")}
          </label>
          <select
            id="team-select"
            value={team}
            onChange={(e) => onTeamChange(e.target.value)}
            className="rounded-lg border border-border bg-surface px-3 py-1.5 text-[0.85rem] text-text focus:border-brand focus:outline-none focus:ring-1 focus:ring-brand-ring"
          >
            {teams.map((name) => (
              <option key={name} value={name}>
                {titleCase(name)}
              </option>
            ))}
          </select>
          {loading && (
            <div className="size-4 animate-spin rounded-full border-2 border-brand border-t-transparent" />
          )}
        </div>
      </div>

      {/* 1. Model decomposition */}
      <Reveal as="section">
        <h2 className="font-display text-[1.6rem] leading-tight sm:text-[2rem]">
          {t("decomposition.title")}
        </h2>
        <p className="mt-3 max-w-2xl text-[0.92rem] leading-relaxed text-muted">
          {t("decomposition.lead")}
        </p>
        {decomposition ? (
          <DecompositionChart matches={decomposition.matches} />
        ) : loading ? (
          <div className="mt-6 h-[360px] animate-pulse rounded-xl bg-surface" />
        ) : null}
      </Reveal>

      {/* 2. Prediction errors */}
      {initialErrors && (
        <Reveal as="section">
          <h2 className="font-display text-[1.6rem] leading-tight sm:text-[2rem]">
            {t("errors.title")}
          </h2>
          <p className="mt-3 max-w-2xl text-[0.92rem] leading-relaxed text-muted">
            {t("errors.lead")}
          </p>
          <div className="mt-2 flex gap-6 text-[0.8rem] text-muted">
            <span>
              RPS: <strong className="text-text">{initialErrors.total_rps?.toFixed(4)}</strong>
            </span>
            <span>
              n: <strong className="text-text">{initialErrors.n.toLocaleString()}</strong>
            </span>
          </div>
          <ErrorAnalysis data={initialErrors} />
        </Reveal>
      )}

      {/* 3. Enhanced calibration */}
      {initialCalibration && (
        <Reveal as="section">
          <h2 className="font-display text-[1.6rem] leading-tight sm:text-[2rem]">
            {t("calibration.title")}
          </h2>
          <p className="mt-3 max-w-2xl text-[0.92rem] leading-relaxed text-muted">
            {t("calibration.lead")}
          </p>
          <CalibrationEnhanced data={initialCalibration} />
        </Reveal>
      )}

      {/* 4. Team strength profile */}
      <Reveal as="section">
        <h2 className="font-display text-[1.6rem] leading-tight sm:text-[2rem]">
          {t("profile.title")}
        </h2>
        <p className="mt-3 max-w-2xl text-[0.92rem] leading-relaxed text-muted">
          {t("profile.lead")}
        </p>
        {profile ? (
          <TeamProfile data={profile} />
        ) : loading ? (
          <div className="mt-6 space-y-6">
            <div className="h-[320px] animate-pulse rounded-xl bg-surface" />
            <div className="mx-auto h-[300px] max-w-[380px] animate-pulse rounded-xl bg-surface" />
          </div>
        ) : null}
      </Reveal>
    </div>
  );
}
