import { getTranslations } from "next-intl/server";

import { Reveal } from "@/components/ui/reveal";
import type { LeagueForecast } from "@/lib/api";

/**
 * Every team against every finishing position.
 *
 * The scale runs from the panel colour to the brand, never from white: a
 * mostly-improbable matrix on a "Blues"-style ramp renders as a pale slab on a
 * dark page and as a dark slab on a light one. Cells below 0.5% are left blank
 * rather than tinted — a wall of near-zeros is noise, and the eye should land
 * on where a team can actually finish.
 */
export async function PositionMatrix({
  matrix,
}: {
  matrix: LeagueForecast["positionMatrix"];
}) {
  const t = await getTranslations("leagues");
  if (!matrix.teams.length) return null;

  const max = Math.max(...matrix.values.flat());

  return (
    <Reveal as="section">
      <h2 className="font-display text-[1.5rem] leading-tight sm:text-[1.9rem]">
        {t("matrixTitle")}
      </h2>
      <p className="mt-3 max-w-2xl text-[0.9rem] leading-relaxed text-muted">
        {t("matrixLead")}
      </p>

      <div className="mt-6 overflow-x-auto rounded-[var(--radius-card)] border border-border">
        <table className="w-full min-w-[720px] border-collapse text-[0.7rem]">
          <thead>
            <tr className="bg-surface-2">
              <th className="sticky left-0 z-10 bg-surface-2 px-3 py-2 text-left text-[0.62rem] font-semibold uppercase tracking-[0.07em] text-dim">
                {t("team")}
              </th>
              {matrix.positions.map((p) => (
                <th
                  key={p}
                  className="px-1 py-2 text-center text-[0.6rem] font-semibold text-dim"
                >
                  {p}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.teams.map((team, i) => (
              <tr key={team}>
                <td className="sticky left-0 z-10 whitespace-nowrap bg-surface px-3 py-1.5 font-medium">
                  {team}
                </td>
                {matrix.values[i].map((v, j) => (
                  <td
                    key={j}
                    className="px-1 py-1.5 text-center tabular-nums"
                    style={{
                      // Cap the tint so the strongest cell stays mid-toned in both
                      // themes, which lets a single theme-aware ink (--text: near
                      // black on light, near white on dark) read on every cell.
                      // A fixed #fff was unreadable on the pale tints of the light
                      // theme.
                      background:
                        v > 0.005
                          ? `color-mix(in oklab, var(--brand) ${Math.round((v / max) * 68)}%, var(--surface))`
                          : "var(--surface)",
                      color: "var(--text)",
                    }}
                    title={`${team} · ${matrix.positions[j]} · ${(v * 100).toFixed(1)}%`}
                  >
                    {v > 0.005 ? Math.round(v * 100) : ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Reveal>
  );
}
